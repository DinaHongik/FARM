"""Fail-closed scoring of the released RecipeGen++ one-shot predictions.

The upstream archive contains ten text beams for three independently prompted
tracks (channel, function, and field).  This module verifies the archive, the
checked-in dataset, the frozen sample, and every selected ID binding before it
computes anything.  Its only output is an aggregate suitable for public
release; queries, targets, predictions, source IDs, and per-case outcomes are
never returned or persisted.

The all-sample field metrics are deliberately named endpoint-gated
schema-label recovery: a field prediction receives no credit when the endpoint
emitted by the same field-track prediction is wrong, while the denominator
remains the full frozen sample.  One separately named cross-track conditional
diagnostic uses only cases where the function-track endpoint pair is correct
and reports that smaller denominator explicitly.  RecipeGen++ supplies field
*names*, not values, ingredient bindings, credentials, resource state, or
executable applets.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import random
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from farm_r9.adapters.common import normalize_label
from farm_r9.adapters.recipegen import parse_target, prepare_recipegen
from farm_r9.artifact_io import (
    canonical_json,
    ordered_ids_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from farm_r9.privacy import DataClassification, export_public_aggregate


ZENODO_RECORD_ID = "6668462"
ZENODO_DOI = "10.5281/zenodo.6668462"
RESULTS_ZIP_SIZE_BYTES = 5_812_340
RESULTS_ZIP_MD5 = "08317eaec43c807b2a61437271c3c5d3"
RESULTS_ZIP_SHA256 = "97bb7f0df99b01b16c076f7c282e5a655d8205ff044c9de4fbac66fccc5b39fa"
PROCESSED_CSV_SHA256 = (
    "6f55a22fac2c71ea6cf6eb29d39f840f534d09eceb576bcc50691130125e357f"
)
EXPECTED_POPULATIONS = {"gold": 261, "noisy": 608}
EXPECTED_SAMPLE_N = 150
EXPECTED_SEED = 9_052_026
BOOTSTRAP_RESAMPLES = 10_000
_TRACKS = ("channel", "function", "field")
_EXTENSIONS = ("src", "gold", "pred")
_TOP_K = (1, 3, 5, 10)
_SOURCE_PROMPTS = {
    "channel": "GENERATE CHANNEL ONLY WITHOUT FUNCTION",
    "function": "GENERATE CHANNEL AND FUNCTION FOR BOTH TRIGGER AND ACTION",
    "field": "GENERATE ON THE FIELD-LEVEL GRANULARITY",
}
_Z_95 = 1.959963984540054


class RecipeGenReleasedError(ValueError):
    """Raised when an upstream or frozen artifact violates its binding."""


def _wilson_95(correct: int, n: int) -> dict[str, float]:
    if (
        isinstance(correct, bool)
        or not isinstance(correct, int)
        or not 0 <= correct <= n
        or n <= 0
    ):
        raise RecipeGenReleasedError(
            "Wilson interval requires 0 <= correct <= n and n > 0"
        )
    proportion = correct / n
    denominator = 1 + _Z_95**2 / n
    centre = (proportion + _Z_95**2 / (2 * n)) / denominator
    radius = (
        _Z_95
        * math.sqrt((proportion * (1 - proportion) + _Z_95**2 / (4 * n)) / n)
        / denominator
    )
    return {
        "low": round(100 * max(0.0, centre - radius), 6),
        "high": round(100 * min(1.0, centre + radius), 6),
        "level": 95.0,
    }


def _type7_quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (
        float(sorted_values[lower]) * (1 - weight)
        + float(sorted_values[upper]) * weight
    )


def _mean_bootstrap_95(values: Sequence[float], *, label: str) -> dict[str, float]:
    if len(values) != EXPECTED_SAMPLE_N or any(
        not math.isfinite(value) for value in values
    ):
        raise RecipeGenReleasedError(
            "continuous metric does not cover the frozen sample"
        )
    seed = int(sha256_text(f"recipegen-released-bootstrap-v1\0{label}")[:16], 16)
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        100 * sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return {
        "low": round(_type7_quantile(draws, 0.025), 6),
        "high": round(_type7_quantile(draws, 0.975), 6),
        "level": 95.0,
    }


def _field_rates(counts: Sequence[tuple[int, int, int]]) -> tuple[float, float, float]:
    true_positive = sum(item[0] for item in counts)
    predicted = sum(item[1] for item in counts)
    gold = sum(item[2] for item in counts)
    # Token-level micro metrics have no positive evidence when both totals
    # are zero. Per-case vacuous correctness additionally requires endpoints
    # and is handled explicitly at the macro aggregation call site.
    if predicted == 0 and gold == 0:
        return 0.0, 0.0, 0.0
    precision = true_positive / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = true_positive / gold if gold else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _micro_bootstrap_95(
    counts: Sequence[tuple[int, int, int]], *, label: str, component: int
) -> dict[str, float]:
    if len(counts) != EXPECTED_SAMPLE_N or component not in {0, 1, 2}:
        raise RecipeGenReleasedError(
            "field micro metric does not cover the frozen sample"
        )
    seed = int(sha256_text(f"recipegen-released-bootstrap-v1\0{label}")[:16], 16)
    rng = random.Random(seed)
    n = len(counts)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = [counts[rng.randrange(n)] for _ in range(n)]
        draws.append(100 * _field_rates(sampled)[component])
    draws.sort()
    return {
        "low": round(_type7_quantile(draws, 0.025), 6),
        "high": round(_type7_quantile(draws, 0.975), 6),
        "level": 95.0,
    }


def _archive_member(split: str, track: str, source_index: int, extension: str) -> str:
    return f"results/oneshot_model/oneshot_{split}_{track}/id{source_index}.{extension}"


def _strict_lines(
    archive: zipfile.ZipFile, member: str, *, expected: int
) -> tuple[list[str], bytes]:
    try:
        raw = archive.read(member)
    except KeyError as error:
        raise RecipeGenReleasedError(
            "released archive is missing a required member"
        ) from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecipeGenReleasedError("released archive member is not UTF-8") from error
    lines = [line.strip() for line in text.splitlines()]
    if len(lines) != expected or any(not line for line in lines):
        raise RecipeGenReleasedError(
            f"released {member.rsplit('.', 1)[-1]} member must contain {expected} nonempty lines"
        )
    return lines, raw


def _validate_archive_inventory(archive: zipfile.ZipFile, split: str) -> None:
    names = archive.namelist()
    duplicate_names = [name for name, count in Counter(names).items() if count != 1]
    if duplicate_names:
        raise RecipeGenReleasedError("released archive contains duplicate member names")
    expected = {
        _archive_member(split, track, source_index, extension)
        for track in _TRACKS
        for source_index in range(EXPECTED_POPULATIONS[split])
        for extension in _EXTENSIONS
    }
    expected_directories = {
        f"results/oneshot_model/oneshot_{split}_{track}" for track in _TRACKS
    }
    relevant = {
        name
        for name in names
        if name.rsplit("/", 1)[0] in expected_directories
        and name.rsplit(".", 1)[-1] in _EXTENSIONS
    }
    if relevant != expected:
        raise RecipeGenReleasedError(
            "released archive one-shot inventory does not match the pinned split population"
        )


def _load_processed_tracks(
    processed_csv: Path, split: str
) -> dict[str, list[dict[str, str]]]:
    if sha256_file(processed_csv) != PROCESSED_CSV_SHA256:
        raise RecipeGenReleasedError(
            "processed.csv SHA-256 does not match the pinned repository artifact"
        )
    with processed_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source", "split", "target", "granularity"]:
            raise RecipeGenReleasedError(
                "processed.csv columns do not match the pinned schema"
            )
        rows = list(reader)
    tracks = {
        track: [
            row for row in rows if row["split"] == split and row["granularity"] == track
        ]
        for track in _TRACKS
    }
    expected = EXPECTED_POPULATIONS[split]
    if any(len(rows_for_track) != expected for rows_for_track in tracks.values()):
        raise RecipeGenReleasedError(
            "processed.csv split/track populations are inconsistent"
        )
    for source_index in range(expected):
        sources = {tracks[track][source_index]["source"] for track in _TRACKS}
        if len(sources) != 1:
            raise RecipeGenReleasedError(
                "processed.csv track occurrence order is not aligned"
            )
    return tracks


def _validate_frozen_sample(
    *,
    cases_path: Path,
    manifest_path: Path,
    processed_csv: Path,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, str]]]]:
    cases = read_jsonl(cases_path)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise RecipeGenReleasedError("sample manifest must be a JSON object")
    if (
        len(cases) != EXPECTED_SAMPLE_N
        or manifest.get("sample_size") != EXPECTED_SAMPLE_N
    ):
        raise RecipeGenReleasedError(
            "released baseline requires the frozen n=150 sample"
        )
    if manifest.get("benchmark") != f"recipegen_{split}":
        raise RecipeGenReleasedError(
            "sample manifest benchmark does not match the requested split"
        )
    if manifest.get("seed") != EXPECTED_SEED:
        raise RecipeGenReleasedError(
            "sample manifest seed does not match the frozen protocol"
        )
    if manifest.get("sampling") != "deterministic-stratified-sha256-v1":
        raise RecipeGenReleasedError(
            "sample manifest sampling method is not the executed frozen method"
        )
    if manifest.get("population_size") != EXPECTED_POPULATIONS[split]:
        raise RecipeGenReleasedError(
            "sample manifest population does not match the pinned split"
        )
    if manifest.get("source_files") != {"processed.csv": PROCESSED_CSV_SHA256}:
        raise RecipeGenReleasedError(
            "sample manifest does not bind the pinned processed.csv"
        )
    if manifest.get("tracks") != ["channel", "function", "field-name"]:
        raise RecipeGenReleasedError("sample manifest track declaration is malformed")
    if sha256_file(cases_path) != manifest.get("case_payload_sha256"):
        raise RecipeGenReleasedError(
            "frozen case payload SHA-256 does not match its manifest"
        )
    ordered_ids = [case.get("case_id") for case in cases]
    if ordered_ids != manifest.get("ordered_case_ids"):
        raise RecipeGenReleasedError("frozen case order does not match its manifest")
    if ordered_ids_sha256(cases) != manifest.get("ordered_case_ids_sha256"):
        raise RecipeGenReleasedError(
            "frozen ordered-case SHA-256 does not match its manifest"
        )
    if len(set(ordered_ids)) != EXPECTED_SAMPLE_N:
        raise RecipeGenReleasedError("frozen sample contains duplicate case IDs")

    # Re-run the executed sampler from the pinned source.  This is stronger
    # than merely trusting the checked-in manifest and catches index drift.
    recomputed_cases, recomputed_manifest = prepare_recipegen(
        processed_csv,
        split=split,
        size=EXPECTED_SAMPLE_N,
        seed=EXPECTED_SEED,
    )
    if recomputed_cases != cases:
        raise RecipeGenReleasedError(
            "frozen cases cannot be reproduced from processed.csv"
        )
    manifest_keys = {
        "benchmark",
        "seed",
        "sampling",
        "population_size",
        "sample_size",
        "population_strata",
        "sample_strata",
        "quotas",
        "ordered_case_ids",
        "source_files",
        "tracks",
        "field_semantics",
    }
    if any(manifest.get(key) != recomputed_manifest.get(key) for key in manifest_keys):
        raise RecipeGenReleasedError(
            "frozen manifest cannot be reproduced by the executed sampler"
        )

    tracks = _load_processed_tracks(processed_csv, split)
    selected_indexes: set[int] = set()
    realized_strata = Counter()
    for case in cases:
        if case.get("benchmark") != f"recipegen_{split}":
            raise RecipeGenReleasedError(
                "frozen case benchmark does not match the requested split"
            )
        audit = case.get("audit")
        if not isinstance(audit, Mapping):
            raise RecipeGenReleasedError(
                "frozen case lacks its upstream occurrence index"
            )
        source_index = audit.get("source_index_within_split_field_track")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or not 0 <= source_index < EXPECTED_POPULATIONS[split]
            or source_index in selected_indexes
        ):
            raise RecipeGenReleasedError(
                "frozen case has an invalid or duplicate source index"
            )
        selected_indexes.add(source_index)
        field_row = tracks["field"][source_index]
        if case.get("input") != {"query": field_row["source"]}:
            raise RecipeGenReleasedError(
                "frozen case query is not bound to its processed.csv occurrence"
            )
        parsed_gold = parse_target(field_row["target"])
        private_gold = case.get("private_gold")
        if not isinstance(private_gold, Mapping) or any(
            private_gold.get(key) != parsed_gold[key]
            for key in (
                "trigger_channel",
                "trigger_function",
                "trigger_fields",
                "action_channel",
                "action_function",
                "action_fields",
            )
        ):
            raise RecipeGenReleasedError(
                "frozen case gold is not bound to its processed.csv occurrence"
            )
        expected_case_id = (
            f"recipegen:{split}:"
            + sha256_text(
                f"{split}\x1f{source_index}\x1f{field_row['source']}\x1f{field_row['target']}"
            )[:20]
        )
        if case.get("case_id") != expected_case_id:
            raise RecipeGenReleasedError("frozen case ID cannot be reproduced")
        stratum = case.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise RecipeGenReleasedError("frozen case has no valid stratum")
        realized_strata[stratum] += 1
    if dict(sorted(realized_strata.items())) != manifest.get("sample_strata"):
        raise RecipeGenReleasedError("frozen sample strata do not match the manifest")
    return cases, dict(manifest), tracks


def _parse_components(value: str, expected: int) -> tuple[str, ...] | None:
    components = tuple(component.strip() for component in value.split(" <sep> "))
    if len(components) != expected or any(not component for component in components):
        return None
    return components


def _same(left: str, right: str) -> bool:
    return normalize_label(left) == normalize_label(right)


def _multiset_counts(
    predicted: Sequence[str], gold: Sequence[str], *, endpoint_correct: bool
) -> tuple[int, int, int]:
    predicted_counter = Counter(normalize_label(item) for item in predicted)
    gold_counter = Counter(normalize_label(item) for item in gold)
    true_positive = (
        sum((predicted_counter & gold_counter).values()) if endpoint_correct else 0
    )
    return true_positive, sum(predicted_counter.values()), sum(gold_counter.values())


def _append_binary(store: dict[str, list[bool]], label: str, value: bool) -> None:
    store.setdefault(label, []).append(bool(value))


def _selected_members_sha256(selected_members: Sequence[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in selected_members:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def evaluate_released_predictions(
    *,
    archive_path: Path,
    processed_csv: Path,
    cases_path: Path,
    manifest_path: Path,
    split: str,
) -> dict[str, Any]:
    """Return one strict aggregate for the released predictions on frozen n=150."""

    if split not in EXPECTED_POPULATIONS:
        raise RecipeGenReleasedError("split must be gold or noisy")
    if archive_path.stat().st_size != RESULTS_ZIP_SIZE_BYTES:
        raise RecipeGenReleasedError(
            "results.zip byte size does not match the pinned Zenodo file"
        )
    if sha256_file(archive_path) != RESULTS_ZIP_SHA256:
        raise RecipeGenReleasedError(
            "results.zip SHA-256 does not match the pinned Zenodo file"
        )

    cases, _, tracks = _validate_frozen_sample(
        cases_path=cases_path,
        manifest_path=manifest_path,
        processed_csv=processed_csv,
        split=split,
    )
    binary: dict[str, list[bool]] = {}
    continuous: dict[str, list[float]] = {}
    conditional_binary: dict[str, list[bool]] = {}
    conditional_eligibility: dict[str, list[bool]] = {}
    field_count_rows: list[tuple[int, int, int]] = []
    failures: Counter[str] = Counter()
    selected_members: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(archive_path, "r") as archive:
        _validate_archive_inventory(archive, split)
        for case in cases:
            source_index = int(case["audit"]["source_index_within_split_field_track"])
            predictions_by_track: dict[str, list[str]] = {}
            for track in _TRACKS:
                source_member = _archive_member(split, track, source_index, "src")
                gold_member = _archive_member(split, track, source_index, "gold")
                prediction_member = _archive_member(split, track, source_index, "pred")
                source_lines, source_raw = _strict_lines(
                    archive, source_member, expected=1
                )
                gold_lines, gold_raw = _strict_lines(archive, gold_member, expected=1)
                prediction_lines, prediction_raw = _strict_lines(
                    archive, prediction_member, expected=10
                )
                selected_members.extend(
                    (
                        (source_member, source_raw),
                        (gold_member, gold_raw),
                        (prediction_member, prediction_raw),
                    )
                )
                row = tracks[track][source_index]
                expected_prefix = _SOURCE_PROMPTS[track] + " <pf> "
                if not source_lines[0].startswith(expected_prefix):
                    raise RecipeGenReleasedError(
                        "released source prompt does not match its track"
                    )
                decoded_source = source_lines[0][len(expected_prefix) :]
                if normalize_label(decoded_source) != normalize_label(row["source"]):
                    raise RecipeGenReleasedError(
                        "released source is not aligned with its processed.csv occurrence"
                    )
                if gold_lines != [row["target"].strip()]:
                    raise RecipeGenReleasedError(
                        "released gold is not aligned with its processed.csv occurrence"
                    )
                predictions_by_track[track] = prediction_lines
                rank = next(
                    (
                        ordinal
                        for ordinal, prediction in enumerate(prediction_lines, start=1)
                        if prediction == gold_lines[0]
                    ),
                    None,
                )
                for top_k in _TOP_K:
                    _append_binary(
                        binary,
                        f"{track}_program_recall_at_{top_k}",
                        rank is not None and rank <= top_k,
                    )
                    continuous.setdefault(f"{track}_program_mrr_at_{top_k}", []).append(
                        1.0 / rank if rank is not None and rank <= top_k else 0.0
                    )

            function_gold = _parse_components(
                tracks["function"][source_index]["target"], 4
            )
            if function_gold is None:  # pragma: no cover - pinned source invariant
                raise RecipeGenReleasedError("processed function target is malformed")
            function_prediction = _parse_components(
                predictions_by_track["function"][0], 4
            )
            if function_prediction is None:
                failures["function_top1_parse_failure"] += 1
                function_components = (False, False, False, False)
            else:
                function_components = tuple(
                    _same(predicted, gold)
                    for predicted, gold in zip(
                        function_prediction, function_gold, strict=True
                    )
                )
            ftc, ftf, fac, faf = function_components
            _append_binary(binary, "function_track_trigger_service_exact", ftc)
            _append_binary(binary, "function_track_action_service_exact", fac)
            _append_binary(binary, "function_track_service_joint_exact", ftc and fac)
            _append_binary(binary, "function_track_trigger_endpoint_exact", ftc and ftf)
            _append_binary(binary, "function_track_action_endpoint_exact", fac and faf)
            _append_binary(
                binary,
                "function_track_endpoint_joint_exact",
                ftc and ftf and fac and faf,
            )
            function_endpoint_joint = ftc and ftf and fac and faf
            continuous.setdefault("function_track_four_component_mean", []).append(
                sum(function_components) / 4
            )

            channel_gold = _parse_components(
                tracks["channel"][source_index]["target"], 2
            )
            channel_prediction = _parse_components(
                predictions_by_track["channel"][0], 2
            )
            if channel_gold is None:  # pragma: no cover - pinned source invariant
                raise RecipeGenReleasedError("processed channel target is malformed")
            if channel_prediction is None:
                failures["channel_top1_parse_failure"] += 1
                channel_components = (False, False)
            else:
                channel_components = tuple(
                    _same(predicted, gold)
                    for predicted, gold in zip(
                        channel_prediction, channel_gold, strict=True
                    )
                )
            _append_binary(
                binary,
                "channel_track_service_joint_exact",
                channel_components[0] and channel_components[1],
            )

            field_gold = parse_target(tracks["field"][source_index]["target"])
            try:
                field_prediction = parse_target(predictions_by_track["field"][0])
            except ValueError:
                failures["field_top1_parse_failure"] += 1
                field_prediction = None
            if field_prediction is None:
                trigger_endpoint = action_endpoint = False
                trigger_fields: Sequence[str] = ()
                action_fields: Sequence[str] = ()
            else:
                trigger_endpoint = _same(
                    field_prediction["trigger_channel"], field_gold["trigger_channel"]
                ) and _same(
                    field_prediction["trigger_function"], field_gold["trigger_function"]
                )
                action_endpoint = _same(
                    field_prediction["action_channel"], field_gold["action_channel"]
                ) and _same(
                    field_prediction["action_function"], field_gold["action_function"]
                )
                trigger_fields = field_prediction["trigger_fields"]
                action_fields = field_prediction["action_fields"]
            trigger_labels_ordered = field_prediction is not None and [
                normalize_label(value) for value in trigger_fields
            ] == [normalize_label(value) for value in field_gold["trigger_fields"]]
            action_labels_ordered = field_prediction is not None and [
                normalize_label(value) for value in action_fields
            ] == [normalize_label(value) for value in field_gold["action_fields"]]
            trigger_ordered = trigger_endpoint and trigger_labels_ordered
            action_ordered = action_endpoint and action_labels_ordered
            _append_binary(
                binary, "field_track_trigger_endpoint_exact", trigger_endpoint
            )
            _append_binary(binary, "field_track_action_endpoint_exact", action_endpoint)
            _append_binary(
                binary,
                "field_track_endpoint_joint_exact",
                trigger_endpoint and action_endpoint,
            )
            _append_binary(
                binary,
                "field_trigger_ordered_exact_endpoint_gated",
                trigger_ordered,
            )
            _append_binary(
                binary,
                "field_action_ordered_exact_endpoint_gated",
                action_ordered,
            )
            _append_binary(
                binary,
                "field_joint_ordered_exact_endpoint_gated",
                trigger_ordered and action_ordered,
            )
            # This is a cross-track diagnostic, not the all-sample field score:
            # eligibility is established solely by the independently prompted
            # function track, while the outcome checks only the two ordered
            # field-name lists emitted by the field track.  The denominator is
            # therefore the count of function-track endpoint-joint successes.
            conditional_label = (
                "field_track_ordered_label_lists_joint_exact_given_"
                "function_track_endpoint_joint_exact"
            )
            _append_binary(
                conditional_eligibility,
                conditional_label,
                function_endpoint_joint,
            )
            _append_binary(
                conditional_binary,
                conditional_label,
                function_endpoint_joint
                and trigger_labels_ordered
                and action_labels_ordered,
            )
            trigger_counts = _multiset_counts(
                trigger_fields,
                field_gold["trigger_fields"],
                endpoint_correct=trigger_endpoint,
            )
            action_counts = _multiset_counts(
                action_fields,
                field_gold["action_fields"],
                endpoint_correct=action_endpoint,
            )
            combined_counts = tuple(
                trigger_counts[index] + action_counts[index] for index in range(3)
            )
            field_count_rows.append(combined_counts)
            precision, recall, f1 = _field_rates([combined_counts])
            if combined_counts[1] == 0 and combined_counts[2] == 0:
                precision = recall = f1 = float(trigger_endpoint and action_endpoint)
            continuous.setdefault(
                "field_label_macro_precision_endpoint_gated", []
            ).append(precision)
            continuous.setdefault("field_label_macro_recall_endpoint_gated", []).append(
                recall
            )
            continuous.setdefault("field_label_macro_f1_endpoint_gated", []).append(f1)

    if any(len(values) != EXPECTED_SAMPLE_N for values in binary.values()):
        raise RecipeGenReleasedError("binary metric does not cover the frozen sample")
    if any(len(values) != EXPECTED_SAMPLE_N for values in continuous.values()):
        raise RecipeGenReleasedError(
            "continuous metric does not cover the frozen sample"
        )
    if any(
        len(values) != EXPECTED_SAMPLE_N
        for store in (conditional_binary, conditional_eligibility)
        for values in store.values()
    ):
        raise RecipeGenReleasedError(
            "conditional metric does not cover the frozen sample"
        )

    raw_numerators = {label: sum(values) for label, values in sorted(binary.items())}
    raw_numerators.update(
        {label: sum(values) for label, values in sorted(conditional_binary.items())}
    )
    raw_denominators = {label: EXPECTED_SAMPLE_N for label in binary}
    raw_denominators.update(
        {label: sum(conditional_eligibility[label]) for label in conditional_binary}
    )
    if any(denominator <= 0 for denominator in raw_denominators.values()):
        raise RecipeGenReleasedError("metric denominator must be positive")
    percentages: dict[str, float] = {
        label: round(100 * sum(values) / EXPECTED_SAMPLE_N, 6)
        for label, values in sorted(binary.items())
    }
    confidence_intervals = {
        label: _wilson_95(sum(values), EXPECTED_SAMPLE_N)
        for label, values in sorted(binary.items())
    }
    for label, values in sorted(conditional_binary.items()):
        denominator = raw_denominators[label]
        numerator = sum(values)
        percentages[label] = round(100 * numerator / denominator, 6)
        confidence_intervals[label] = _wilson_95(numerator, denominator)
    for label, values in sorted(continuous.items()):
        percentages[label] = round(100 * sum(values) / EXPECTED_SAMPLE_N, 6)
        confidence_intervals[label] = _mean_bootstrap_95(values, label=label)
    micro_precision, micro_recall, micro_f1 = _field_rates(field_count_rows)
    for component, (label, value) in enumerate(
        (
            ("field_label_micro_precision_endpoint_gated", micro_precision),
            ("field_label_micro_recall_endpoint_gated", micro_recall),
            ("field_label_micro_f1_endpoint_gated", micro_f1),
        )
    ):
        percentages[label] = round(100 * value, 6)
        confidence_intervals[label] = _micro_bootstrap_95(
            field_count_rows,
            label=label,
            component=component,
        )

    selected_members_sha256 = _selected_members_sha256(selected_members)
    aggregate_input_sha256 = sha256_text(
        canonical_json(
            {
                "archive_sha256": RESULTS_ZIP_SHA256,
                "processed_csv_sha256": PROCESSED_CSV_SHA256,
                "cases_sha256": sha256_file(cases_path),
                "sample_manifest_sha256": sha256_file(manifest_path),
                "selected_members_sha256": selected_members_sha256,
            }
        )
    )
    aggregate = {
        "benchmark_label": f"recipegen_{split}.released_oneshot_archive",
        "n": EXPECTED_SAMPLE_N,
        "raw_numerators": dict(sorted(raw_numerators.items())),
        "raw_denominators": dict(sorted(raw_denominators.items())),
        "percentages": dict(sorted(percentages.items())),
        "confidence_intervals": dict(sorted(confidence_intervals.items())),
        "failure_counts": dict(sorted(failures.items())),
        "model_metadata": {
            "name": "recipegenpp_released_oneshot",
            "provider": "zenodo",
            "version": ZENODO_RECORD_ID,
            "role": "prior_work_baseline",
        },
        "protocol_metadata": {
            "id": "recipegen_released_predictions_v3",
            "version": "3",
            "arm": "released_oneshot_archive",
            "format": "three_tracks_top10_text_beams",
            "semantic_calls_max": 0,
        },
        "hashes": {
            "artifact_sha256": RESULTS_ZIP_SHA256,
            "source_artifact_sha256": PROCESSED_CSV_SHA256,
            "sample_manifest_sha256": sha256_file(manifest_path),
            "aggregate_input_sha256": aggregate_input_sha256,
            "code_sha256": sha256_file(Path(__file__)),
        },
    }
    return export_public_aggregate(
        aggregate,
        source_classification=DataClassification.PUBLIC,
    )


def write_aggregate_exclusive(path: Path, aggregate: Mapping[str, Any]) -> None:
    """Write a mode-0600 aggregate once, refusing every overwrite."""

    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed:
        os.chmod(path.parent, 0o700)
    payload = (canonical_json(dict(aggregate)) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RecipeGenReleasedError(
            "refusing to overwrite an existing aggregate"
        ) from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "EXPECTED_POPULATIONS",
    "EXPECTED_SAMPLE_N",
    "PROCESSED_CSV_SHA256",
    "RESULTS_ZIP_MD5",
    "RESULTS_ZIP_SHA256",
    "RESULTS_ZIP_SIZE_BYTES",
    "RecipeGenReleasedError",
    "ZENODO_DOI",
    "ZENODO_RECORD_ID",
    "evaluate_released_predictions",
    "write_aggregate_exclusive",
]

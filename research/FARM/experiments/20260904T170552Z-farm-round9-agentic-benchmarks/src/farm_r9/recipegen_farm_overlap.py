"""Privacy-safe RecipeGen++ versus FARM-v2 training-overlap audit.

The public RecipeGen++ cases are compared locally with the two FARM-v2
partitions that supply training examples: ``encoder_train`` and
``reranker_train``.  FARM text, identifiers, row-level hashes, and endpoint
programs never leave this module.  The public-facing result is restricted to
denominator-aligned counts, percentages, Wilson intervals, aggregate failure
counts, and hashes of complete immutable artifacts.

The fuzzy check is deliberately narrow.  It reuses Dataset-v2's already-pinned
lexical family heuristic (content-token-set Jaccard >= 0.90, with at least
three content tokens on each side).  It is a deterministic spelling/wording
near-duplicate diagnostic, not semantic paraphrase or model-embedding evidence.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from farm_r9.adapters.common import normalize_label
from farm_r9.artifact_io import read_json, sha256_file
from farm_r9.privacy import DataClassification, export_public_aggregate
from farm_r9.recipegen_released import _validate_frozen_sample


SCHEMA_VERSION = "round9-recipegen-farm-training-overlap-v2"
INPUT_NORMALIZATION_VERSION = (
    "unicode-nfc-collapse-whitespace-preserve-case-punctuation-v1"
)
LEXICAL_INPUT_NORMALIZATION_VERSION = "unicode-nfkc-casefold-alnum-v1"
PROGRAM_NORMALIZATION_VERSION = (
    "recipegen-namespace-strip-farm-declared-aliases-round9-normalize-label-v1"
)
LEXICAL_DIAGNOSTIC_VERSION = "farm-v2-content-token-jaccard-v1"
LEXICAL_THRESHOLD = 0.90
MINIMUM_CONTENT_TOKENS = 3
TRAINING_PARTITIONS = ("encoder_train", "reranker_train")

FARM_DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
FARM_MANIFEST_SHA256 = (
    "002e061c2f4278679129e4eb61741bb01fcf2d8ad3b22b7c65c7f86c78c71b18"
)
FARM_ARTIFACTS = {
    "corpus/actions.json": (
        "e93da4eb3594ffea5f0888920aa35b01d3e17ec4b6b4278911ae42284149ab49"
    ),
    "corpus/triggers.json": (
        "a19d162572dc9b94239a4d2b6caf0d9f03140d4d87b40decee1762ed695b5444"
    ),
    "splits/encoder_train.json": (
        "0ce26a5ed12e98884e26764dcae79a20b27d15b61ee4fca72ef590098aaaa4dd"
    ),
    "splits/reranker_train.json": (
        "8108ea1c757fb2a29c5c25b49d69a9c9b87839d2d2a7e83e002f557a5acf932a"
    ),
}
FARM_ARTIFACT_ROWS = {
    "corpus/actions.json": 1_520,
    "corpus/triggers.json": 1_985,
    "splits/encoder_train.json": 8_018,
    "splits/reranker_train.json": 1_145,
}

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "if",
        "when",
        "your",
        "my",
        "with",
        "from",
        "at",
        "is",
        "it",
        "this",
        "that",
        "be",
        "by",
    }
)
_Z_95 = 1.959963984540054

EndpointKey = tuple[str, str]
ProgramKey = tuple[str, str, str, str]


class RecipeGenFarmOverlapError(ValueError):
    """Raised when an input cannot satisfy the pinned overlap contract."""


@dataclass(frozen=True)
class TrainingExample:
    """The minimum private state required for comparison.

    No FARM identifier, source text, endpoint URL, or record digest is retained.
    """

    input_norm: str
    input_canonical_norm: str
    content_token_set: frozenset[str]
    program_keys: frozenset[ProgramKey]


@dataclass(frozen=True)
class TrainingIndex:
    examples: Mapping[str, tuple[TrainingExample, ...]]
    endpoint_key_counts: Mapping[str, Mapping[EndpointKey, int]]
    source_artifact_sha256: str
    manifest_sha256: str


def normalize_input(value: str) -> str:
    """Normalize representation noise while preserving lexical identity."""
    if not isinstance(value, str):
        raise TypeError("input must be a string")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def canonical_input_key(value: str) -> str:
    """Apply FARM Dataset-v2's lossy canonical grouping normalization."""
    if not isinstance(value, str):
        raise TypeError("input must be a string")
    normalized = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    characters: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        characters.append(
            character if character.isalnum() or category.startswith("M") else " "
        )
    return " ".join("".join(characters).split())


def content_tokens(value: str) -> frozenset[str]:
    """Return the exact content-token set used by FARM Dataset-v2."""
    return frozenset(
        token
        for token in canonical_input_key(value).split()
        if len(token) > 2 and token not in _STOPWORDS
    )


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _nonempty_program_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise RecipeGenFarmOverlapError(f"{label} must be a string")
    normalized = normalize_label(value)
    if not normalized:
        raise RecipeGenFarmOverlapError(f"{label} is empty after normalization")
    return normalized


def _recipe_endpoint_key(channel: Any, function: Any) -> EndpointKey:
    channel_norm = _nonempty_program_component(channel, label="RecipeGen service")
    if not isinstance(function, str) or "." not in function:
        raise RecipeGenFarmOverlapError(
            "RecipeGen function must carry its service namespace"
        )
    namespace, function_name = function.split(".", 1)
    if normalize_label(namespace) != channel_norm:
        raise RecipeGenFarmOverlapError(
            "RecipeGen function namespace does not match its service"
        )
    function_norm = _nonempty_program_component(
        function_name, label="RecipeGen function name"
    )
    return channel_norm, function_norm


def _recipe_program_key(case: Mapping[str, Any]) -> ProgramKey:
    gold = case.get("private_gold")
    if not isinstance(gold, Mapping):
        raise RecipeGenFarmOverlapError("RecipeGen case lacks endpoint gold")
    trigger = _recipe_endpoint_key(
        gold.get("trigger_channel"), gold.get("trigger_function")
    )
    action = _recipe_endpoint_key(
        gold.get("action_channel"), gold.get("action_function")
    )
    return trigger[0], trigger[1], action[0], action[1]


def _json_array(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or len(value) != expected_rows:
        raise RecipeGenFarmOverlapError(
            "pinned FARM artifact has an unexpected row count"
        )
    if any(not isinstance(row, dict) for row in value):
        raise RecipeGenFarmOverlapError("pinned FARM artifact contains a non-object")
    return value


def _endpoint_aliases(
    record: Mapping[str, Any], *, expected_kind: str
) -> set[EndpointKey]:
    if record.get("kind") != expected_kind:
        raise RecipeGenFarmOverlapError("FARM corpus kind is inconsistent")
    service_values = (record.get("channel"), record.get("channel_display"))
    function_values = (record.get("function_name"), record.get("function_slug"))
    if any(not isinstance(value, str) for value in service_values + function_values):
        raise RecipeGenFarmOverlapError("FARM corpus aliases must be strings")
    service_aliases = {
        normalized for value in service_values if (normalized := normalize_label(value))
    }
    function_aliases = {
        normalized
        for value in function_values
        if (normalized := normalize_label(value))
    }
    if not service_aliases or not function_aliases:
        raise RecipeGenFarmOverlapError(
            "FARM corpus record has no comparable declared alias"
        )
    return {
        (service, function)
        for service in service_aliases
        for function in function_aliases
    }


def build_training_index(
    *,
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    trigger_corpus: Sequence[Mapping[str, Any]],
    action_corpus: Sequence[Mapping[str, Any]],
    source_artifact_sha256: str,
    manifest_sha256: str,
) -> TrainingIndex:
    """Build a URL-free in-memory index after validating split/corpus joins."""
    corpora = {"trigger": trigger_corpus, "action": action_corpus}
    aliases_by_url: dict[str, dict[str, set[EndpointKey]]] = {}
    endpoint_key_counts: dict[str, dict[EndpointKey, int]] = {}
    for kind, rows in corpora.items():
        lookup: dict[str, set[EndpointKey]] = {}
        counts: dict[EndpointKey, int] = {}
        for record in rows:
            url = record.get("url")
            if not isinstance(url, str) or not url or url in lookup:
                raise RecipeGenFarmOverlapError(
                    "FARM corpus URLs must be nonempty and unique"
                )
            aliases = _endpoint_aliases(record, expected_kind=kind)
            lookup[url] = aliases
            for key in aliases:
                counts[key] = counts.get(key, 0) + 1
        aliases_by_url[kind] = lookup
        endpoint_key_counts[kind] = counts

    examples: dict[str, tuple[TrainingExample, ...]] = {}
    seen_group_ids: set[str] = set()
    for partition in TRAINING_PARTITIONS:
        rows = split_rows.get(partition)
        if not isinstance(rows, Sequence):
            raise RecipeGenFarmOverlapError("FARM training partition is missing")
        partition_examples: list[TrainingExample] = []
        for row in rows:
            if row.get("split") != partition:
                raise RecipeGenFarmOverlapError(
                    "FARM row is assigned to the wrong training partition"
                )
            group_id = row.get("group_id")
            if (
                not isinstance(group_id, str)
                or not group_id
                or group_id in seen_group_ids
            ):
                raise RecipeGenFarmOverlapError(
                    "FARM training group identifiers are missing or not disjoint"
                )
            seen_group_ids.add(group_id)
            query = row.get("query")
            query_norm = row.get("query_norm")
            if not isinstance(query, str) or canonical_input_key(query) != query_norm:
                raise RecipeGenFarmOverlapError(
                    "FARM training input does not reproduce its normalized key"
                )
            pairs = row.get("valid_pairs")
            if not isinstance(pairs, list) or not pairs:
                raise RecipeGenFarmOverlapError(
                    "FARM training row has no valid endpoint pair"
                )
            program_keys: set[ProgramKey] = set()
            for pair in pairs:
                if not isinstance(pair, Mapping):
                    raise RecipeGenFarmOverlapError(
                        "FARM endpoint pair is not an object"
                    )
                trigger = aliases_by_url["trigger"].get(pair.get("trigger_url"))
                action = aliases_by_url["action"].get(pair.get("action_url"))
                if not trigger or not action:
                    raise RecipeGenFarmOverlapError(
                        "FARM training endpoint is absent from the pinned corpus"
                    )
                program_keys.update(
                    (trigger_key[0], trigger_key[1], action_key[0], action_key[1])
                    for trigger_key in trigger
                    for action_key in action
                )
            partition_examples.append(
                TrainingExample(
                    input_norm=normalize_input(query),
                    input_canonical_norm=query_norm,
                    content_token_set=content_tokens(query),
                    program_keys=frozenset(program_keys),
                )
            )
        examples[partition] = tuple(partition_examples)

    return TrainingIndex(
        examples=examples,
        endpoint_key_counts=endpoint_key_counts,
        source_artifact_sha256=source_artifact_sha256,
        manifest_sha256=manifest_sha256,
    )


def load_pinned_farm_training_index(
    data_root: Path,
    *,
    expected_dataset_id: str = FARM_DATASET_ID,
    expected_manifest_sha256: str = FARM_MANIFEST_SHA256,
    expected_artifacts: Mapping[str, str] = FARM_ARTIFACTS,
    expected_rows: Mapping[str, int] = FARM_ARTIFACT_ROWS,
) -> TrainingIndex:
    """Verify and load the immutable FARM-v2 training sources."""
    manifest_path = data_root / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise RecipeGenFarmOverlapError(
            "FARM manifest SHA-256 is not the pinned artifact"
        )
    manifest = read_json(manifest_path)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("dataset_id") != expected_dataset_id
    ):
        raise RecipeGenFarmOverlapError(
            "FARM Dataset-v2 identity is not the pinned identity"
        )
    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise RecipeGenFarmOverlapError("FARM manifest lacks its normalization config")
    paraphrase = config.get("paraphrase")
    if (
        config.get("normalizer") != LEXICAL_INPUT_NORMALIZATION_VERSION
        or not isinstance(paraphrase, Mapping)
        or paraphrase.get("threshold") != LEXICAL_THRESHOLD
        or paraphrase.get("minimum_content_tokens") != MINIMUM_CONTENT_TOKENS
    ):
        raise RecipeGenFarmOverlapError(
            "FARM manifest does not bind the prespecified lexical protocol"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RecipeGenFarmOverlapError("FARM manifest lacks its artifact ledger")

    loaded: dict[str, list[dict[str, Any]]] = {}
    for relative_path, expected_sha256 in expected_artifacts.items():
        entry = artifacts.get(relative_path)
        row_count = expected_rows.get(relative_path)
        if (
            not isinstance(entry, Mapping)
            or entry.get("sha256") != expected_sha256
            or entry.get("rows") != row_count
            or not isinstance(row_count, int)
        ):
            raise RecipeGenFarmOverlapError(
                "FARM manifest artifact binding does not match the pinned ledger"
            )
        path = data_root / relative_path
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RecipeGenFarmOverlapError(
                "FARM source SHA-256 differs from the pinned manifest"
            )
        loaded[relative_path] = _json_array(path, expected_rows=row_count)

    return build_training_index(
        split_rows={
            partition: loaded[f"splits/{partition}.json"]
            for partition in TRAINING_PARTITIONS
        },
        trigger_corpus=loaded["corpus/triggers.json"],
        action_corpus=loaded["corpus/actions.json"],
        # The manifest is a complete immutable file whose own artifact ledger
        # binds every split and corpus source checked above.
        source_artifact_sha256=manifest_sha256,
        manifest_sha256=manifest_sha256,
    )


def _partition_outcome(
    *,
    input_norm: str,
    input_canonical_norm: str,
    token_set: frozenset[str],
    program_key: ProgramKey,
    examples: Sequence[TrainingExample],
) -> dict[str, Any]:
    input_exact = False
    input_canonical_equivalent = False
    program_exact = False
    input_program_pair_exact = False
    input_program_pair_canonical_equivalent = False
    lexical_near_input = False
    lexical_near_input_program_pair = False
    maximum_similarity = 0.0
    eligible = len(token_set) >= MINIMUM_CONTENT_TOKENS
    for example in examples:
        same_input = input_norm == example.input_norm
        same_canonical_input = input_canonical_norm == example.input_canonical_norm
        same_program = program_key in example.program_keys
        input_exact = input_exact or same_input
        input_canonical_equivalent = input_canonical_equivalent or same_canonical_input
        program_exact = program_exact or same_program
        input_program_pair_exact = input_program_pair_exact or (
            same_input and same_program
        )
        input_program_pair_canonical_equivalent = (
            input_program_pair_canonical_equivalent
            or (same_canonical_input and same_program)
        )
        if (
            not eligible
            or same_input
            or len(example.content_token_set) < MINIMUM_CONTENT_TOKENS
        ):
            continue
        similarity = jaccard(token_set, example.content_token_set)
        maximum_similarity = max(maximum_similarity, similarity)
        if similarity >= LEXICAL_THRESHOLD:
            lexical_near_input = True
            lexical_near_input_program_pair = (
                lexical_near_input_program_pair or same_program
            )
    return {
        "input_exact": input_exact,
        "input_canonical_equivalent": input_canonical_equivalent,
        "program_exact": program_exact,
        "input_program_pair_exact": input_program_pair_exact,
        "input_program_pair_canonical_equivalent": (
            input_program_pair_canonical_equivalent
        ),
        "lexical_near_input": lexical_near_input,
        "lexical_near_input_program_pair": lexical_near_input_program_pair,
        "maximum_lexical_jaccard": round(maximum_similarity, 12),
    }


def audit_cases_against_training(
    *,
    cases: Sequence[Mapping[str, Any]],
    index: TrainingIndex,
    benchmark: str,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Return an access-controlled case ledger and aggregate metric counts."""
    if not cases:
        raise RecipeGenFarmOverlapError("RecipeGen sample cannot be empty")
    case_ids = [case.get("case_id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise RecipeGenFarmOverlapError("RecipeGen case IDs are malformed")
    if len(set(case_ids)) != len(case_ids):
        raise RecipeGenFarmOverlapError("RecipeGen case IDs are not unique")

    records: list[dict[str, Any]] = []
    metric_values: dict[str, list[bool]] = {}
    failures = {
        "trigger_endpoint_catalog_mapping_failure": 0,
        "action_endpoint_catalog_mapping_failure": 0,
        "joint_endpoint_catalog_mapping_failure": 0,
        "ambiguous_endpoint_catalog_mapping": 0,
    }

    def append_metric(label: str, value: bool) -> None:
        metric_values.setdefault(label, []).append(bool(value))

    for case in cases:
        if case.get("benchmark") != benchmark:
            raise RecipeGenFarmOverlapError("RecipeGen benchmark label mismatch")
        request = case.get("input")
        if not isinstance(request, Mapping) or not isinstance(
            request.get("query"), str
        ):
            raise RecipeGenFarmOverlapError("RecipeGen case lacks a string input")
        input_norm = normalize_input(request["query"])
        if not input_norm:
            raise RecipeGenFarmOverlapError(
                "RecipeGen input is empty after normalization"
            )
        input_canonical_norm = canonical_input_key(request["query"])
        token_set = content_tokens(input_norm)
        program_key = _recipe_program_key(case)
        trigger_key = program_key[:2]
        action_key = program_key[2:]
        trigger_matches = index.endpoint_key_counts["trigger"].get(trigger_key, 0)
        action_matches = index.endpoint_key_counts["action"].get(action_key, 0)
        if trigger_matches > 1 or action_matches > 1:
            raise RecipeGenFarmOverlapError(
                "RecipeGen endpoint maps ambiguously in the pinned FARM catalog"
            )
        trigger_mapped = trigger_matches > 0
        action_mapped = action_matches > 0
        joint_mapped = trigger_mapped and action_mapped
        failures["trigger_endpoint_catalog_mapping_failure"] += not trigger_mapped
        failures["action_endpoint_catalog_mapping_failure"] += not action_mapped
        failures["joint_endpoint_catalog_mapping_failure"] += not joint_mapped
        append_metric("catalog_trigger_endpoint_mapped", trigger_mapped)
        append_metric("catalog_action_endpoint_mapped", action_mapped)
        append_metric("catalog_joint_endpoint_mapped", joint_mapped)
        append_metric(
            "lexical_near_input_eligible",
            len(token_set) >= MINIMUM_CONTENT_TOKENS,
        )

        partition_outcomes = {
            partition: _partition_outcome(
                input_norm=input_norm,
                input_canonical_norm=input_canonical_norm,
                token_set=token_set,
                program_key=program_key,
                examples=index.examples[partition],
            )
            for partition in TRAINING_PARTITIONS
        }
        all_training = {
            label: any(outcome[label] for outcome in partition_outcomes.values())
            for label in (
                "input_exact",
                "input_canonical_equivalent",
                "program_exact",
                "input_program_pair_exact",
                "input_program_pair_canonical_equivalent",
                "lexical_near_input",
                "lexical_near_input_program_pair",
            )
        }
        all_training["maximum_lexical_jaccard"] = max(
            outcome["maximum_lexical_jaccard"]
            for outcome in partition_outcomes.values()
        )
        views = {
            "all_training": all_training,
            "encoder_partition": partition_outcomes["encoder_train"],
            "reranker_partition": partition_outcomes["reranker_train"],
        }
        for view_name, outcome in views.items():
            for label in (
                "input_exact",
                "input_canonical_equivalent",
                "program_exact",
                "input_program_pair_exact",
                "input_program_pair_canonical_equivalent",
                "lexical_near_input",
                "lexical_near_input_program_pair",
            ):
                append_metric(f"{view_name}_{label}_overlap", outcome[label])
        append_metric(
            "all_training_exact_or_lexical_near_input_overlap",
            all_training["input_exact"] or all_training["lexical_near_input"],
        )

        # This access-controlled record intentionally has no FARM row identity,
        # text, endpoint, URL, program, or record-level digest.
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": benchmark,
                "case_id": case["case_id"],
                "catalog_mapping": {
                    "trigger_endpoint_count": trigger_matches,
                    "action_endpoint_count": action_matches,
                },
                "lexical_diagnostic": {
                    "eligible": len(token_set) >= MINIMUM_CONTENT_TOKENS,
                    "threshold": LEXICAL_THRESHOLD,
                    "minimum_content_tokens": MINIMUM_CONTENT_TOKENS,
                },
                "overlap": views,
            }
        )

    n = len(cases)
    if any(len(values) != n for values in metric_values.values()):
        raise RecipeGenFarmOverlapError("overlap metric does not cover every case")
    counts = {label: sum(values) for label, values in sorted(metric_values.items())}
    return records, counts, failures


def _wilson_95(correct: int, n: int) -> dict[str, float]:
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


def audit_frozen_recipegen_sample(
    *,
    cases_path: Path,
    sample_manifest_path: Path,
    processed_csv: Path,
    split: str,
    farm_data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate both lineages, compare locally, and return internal summary state."""
    if split not in {"gold", "noisy"}:
        raise RecipeGenFarmOverlapError("split must be gold or noisy")
    cases, _, _ = _validate_frozen_sample(
        cases_path=cases_path,
        manifest_path=sample_manifest_path,
        processed_csv=processed_csv,
        split=split,
    )
    index = load_pinned_farm_training_index(farm_data_root)
    records, counts, failures = audit_cases_against_training(
        cases=cases,
        index=index,
        benchmark=f"recipegen_{split}",
    )
    return records, {
        "n": len(cases),
        "counts": counts,
        "failures": failures,
        # This is the hash of the complete frozen case file, not a digest of a
        # synthetic binding object. The separately verified sample manifest
        # binds it to processed.csv and the executed sampler.
        "sample_artifact_sha256": sha256_file(cases_path),
        "source_artifact_sha256": index.source_artifact_sha256,
        "manifest_sha256": index.manifest_sha256,
    }


def make_public_aggregate(
    *,
    summary: Mapping[str, Any],
    case_audit_sha256: str,
    code_artifact_sha256: str,
) -> dict[str, Any]:
    """Create the strict confidential-source aggregate release contract."""
    n = summary.get("n")
    counts = summary.get("counts")
    failures = summary.get("failures")
    if (
        isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
        or not isinstance(counts, Mapping)
        or not isinstance(failures, Mapping)
    ):
        raise RecipeGenFarmOverlapError("internal overlap summary is malformed")
    denominators = {label: n for label in counts}
    percentages = {
        label: round(100 * int(count) / n, 6) for label, count in counts.items()
    }
    confidence_intervals = {
        label: _wilson_95(int(count), n) for label, count in counts.items()
    }
    hashes = {
        "artifact_sha256": case_audit_sha256,
        "code_artifact_sha256": code_artifact_sha256,
        "sample_artifact_sha256": summary["sample_artifact_sha256"],
        "source_artifact_sha256": summary["source_artifact_sha256"],
    }
    return export_public_aggregate(
        {
            "n": n,
            "raw_numerators": dict(counts),
            "raw_denominators": denominators,
            "percentages": percentages,
            "confidence_intervals": confidence_intervals,
            "failure_counts": dict(failures),
            "hashes": hashes,
        },
        source_classification=DataClassification.CONFIDENTIAL,
    )


__all__ = [
    "FARM_ARTIFACTS",
    "FARM_ARTIFACT_ROWS",
    "FARM_DATASET_ID",
    "FARM_MANIFEST_SHA256",
    "INPUT_NORMALIZATION_VERSION",
    "LEXICAL_INPUT_NORMALIZATION_VERSION",
    "LEXICAL_DIAGNOSTIC_VERSION",
    "LEXICAL_THRESHOLD",
    "MINIMUM_CONTENT_TOKENS",
    "PROGRAM_NORMALIZATION_VERSION",
    "RecipeGenFarmOverlapError",
    "SCHEMA_VERSION",
    "TRAINING_PARTITIONS",
    "TrainingExample",
    "TrainingIndex",
    "audit_cases_against_training",
    "audit_frozen_recipegen_sample",
    "build_training_index",
    "canonical_input_key",
    "content_tokens",
    "jaccard",
    "load_pinned_farm_training_index",
    "make_public_aggregate",
    "normalize_input",
]

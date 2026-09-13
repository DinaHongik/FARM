"""Aggregate-only paired comparison of official BFCL evaluator artifacts.

At pinned BFCL commit ``f7cf7359b7ac615a0b294831c5ba2bc95ee4a000``,
the score JSONL contains an aggregate header followed only by official
evaluator failures.  This module treats those failure rows as the sole source
of per-case correctness; it never parses or re-scores model tool calls.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from farm_r9.artifact_io import read_json, read_jsonl, sha256_file, sha256_text
from farm_r9.bfcl_runner import BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN
from farm_r9.paired_statistics import PairedStatisticsError, paired_delta_bootstrap


EXPECTED_N = 150
OFFICIAL_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
EXPECTED_SAMPLE_SHA256 = (
    "e2848d00c5f4b9781e162a60cc66b1f0c795763de3491b26fce3abbb22ab264e"
)
EXPECTED_CASE_ID_SET_SHA256 = (
    "557d2da1af0c7a54b9cd0d506037ae26452150e87e4a2d210d6e8c302f4f0269"
)
SINGLE_STEP_ARM = "single_step_per_turn_fc"
NATIVE_AGENT_ARM = "native_tool_agent"
_Z_95 = 1.959963984540054
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_BINDING_FIELDS = {
    "schema_version",
    "benchmark",
    "evaluation_scope",
    "registry_name",
    "arm",
    "model_sha256",
    "sample_sha256",
    "official_commit",
    "temperature",
    "seed",
    "thinking_mode",
    "max_physical_calls_per_turn",
    "worker_policy",
    "workers",
}


class BFCLComparisonInputError(ValueError):
    """Raised when official artifacts cannot support a safe paired result."""


@dataclass(frozen=True)
class _OfficialArmOutcome:
    case_ids: frozenset[str]
    correct_ids: frozenset[str]
    protocol_failure_ids: frozenset[str]
    result_sha256: str
    score_sha256: str


@dataclass(frozen=True)
class _RunBinding:
    model_sha256: str
    sample_sha256: str
    temperature: float
    seed: int
    thinking_mode: str | None
    max_physical_calls_per_turn: int
    worker_policy: str
    workers: int
    sha256: str


def _read_stable_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise BFCLComparisonInputError("BFCL artifact is missing")
    try:
        digest_before = sha256_file(path)
        rows = read_jsonl(path)
        digest_after = sha256_file(path)
    except (OSError, ValueError) as error:
        raise BFCLComparisonInputError(
            "BFCL artifact is unreadable or malformed"
        ) from error
    if digest_before != digest_after:
        raise BFCLComparisonInputError("BFCL artifact changed while being read")
    return rows, digest_before


def _read_stable_json(path: Path) -> tuple[Mapping[str, Any], str]:
    if not path.is_file():
        raise BFCLComparisonInputError("BFCL run binding is missing")
    try:
        digest_before = sha256_file(path)
        value = read_json(path)
        digest_after = sha256_file(path)
    except (OSError, ValueError) as error:
        raise BFCLComparisonInputError(
            "BFCL run binding is unreadable or malformed"
        ) from error
    if digest_before != digest_after:
        raise BFCLComparisonInputError("BFCL run binding changed while being read")
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BFCLComparisonInputError("BFCL run binding must be a JSON object")
    return value, digest_before


def _strict_positive_integer(
    value: Any, *, field: str, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BFCLComparisonInputError(f"BFCL run binding has invalid {field}")
    if maximum is not None and value > maximum:
        raise BFCLComparisonInputError(f"BFCL run binding has invalid {field}")
    return value


def _load_run_binding(
    path: Path,
    *,
    expected_arm: str,
    expected_native_call_budget: int = 4,
) -> _RunBinding:
    value, digest = _read_stable_json(path)
    if set(value) != _RUN_BINDING_FIELDS:
        raise BFCLComparisonInputError(
            "BFCL run binding has unexpected or missing fields"
        )
    fixed = {
        "schema_version": "farm-round9-bfcl-run-binding-v2",
        "benchmark": "bfcl_v4_multi_turn_miss_param",
        "evaluation_scope": "partial:150_of_200",
        "arm": expected_arm,
        "sample_sha256": EXPECTED_SAMPLE_SHA256,
        "official_commit": OFFICIAL_COMMIT,
        "worker_policy": "thread-pool-atomic-case-resume-v1",
    }
    if any(value.get(field) != expected for field, expected in fixed.items()):
        raise BFCLComparisonInputError(
            "BFCL run binding differs from the frozen protocol"
        )
    if not isinstance(value["registry_name"], str) or not value["registry_name"]:
        raise BFCLComparisonInputError("BFCL run binding has invalid registry_name")
    model_sha256 = value["model_sha256"]
    if not isinstance(model_sha256, str) or not _SHA256.fullmatch(model_sha256):
        raise BFCLComparisonInputError("BFCL run binding has invalid model_sha256")
    temperature = value["temperature"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
    ):
        raise BFCLComparisonInputError("BFCL run binding has invalid temperature")
    seed = _strict_nonnegative_integer(value["seed"], field="seed")
    thinking_mode = value["thinking_mode"]
    if thinking_mode is not None and (
        not isinstance(thinking_mode, str) or not thinking_mode
    ):
        raise BFCLComparisonInputError("BFCL run binding has invalid thinking_mode")
    call_budget = _strict_positive_integer(
        value["max_physical_calls_per_turn"],
        field="max_physical_calls_per_turn",
        maximum=BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN,
    )
    workers = _strict_positive_integer(value["workers"], field="workers", maximum=3)
    if expected_arm == NATIVE_AGENT_ARM and call_budget != expected_native_call_budget:
        raise BFCLComparisonInputError(
            "BFCL native-agent binding changed its declared call budget"
        )
    return _RunBinding(
        model_sha256=model_sha256,
        sample_sha256=EXPECTED_SAMPLE_SHA256,
        temperature=float(temperature),
        seed=seed,
        thinking_mode=thinking_mode,
        max_physical_calls_per_turn=call_budget,
        worker_policy=str(value["worker_policy"]),
        workers=workers,
        sha256=digest,
    )


def _strict_nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BFCLComparisonInputError(f"official score header has invalid {field}")
    return value


def _load_official_arm(result_path: Path, score_path: Path) -> _OfficialArmOutcome:
    result_rows, result_hash = _read_stable_jsonl(result_path)
    if len(result_rows) != EXPECTED_N:
        raise BFCLComparisonInputError(
            "BFCL result artifact must contain exactly 150 records"
        )

    result_ids: set[str] = set()
    protocol_failure_ids: set[str] = set()
    for row in result_rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise BFCLComparisonInputError("BFCL result artifact has a missing ID")
        if case_id in result_ids:
            raise BFCLComparisonInputError("BFCL result artifact has a duplicate ID")
        result_ids.add(case_id)
        protocol_failure = row.get("protocol_failure", False)
        if not isinstance(protocol_failure, bool):
            raise BFCLComparisonInputError(
                "BFCL result protocol-failure flag is not boolean"
            )
        if protocol_failure:
            protocol_failure_ids.add(case_id)
    case_id_set_sha256 = sha256_text("\n".join(sorted(result_ids)) + "\n")
    if case_id_set_sha256 != EXPECTED_CASE_ID_SET_SHA256:
        raise BFCLComparisonInputError(
            "BFCL result IDs differ from the frozen 150-case sample"
        )

    score_rows, score_hash = _read_stable_jsonl(score_path)
    if not score_rows:
        raise BFCLComparisonInputError("official BFCL score artifact is empty")
    header = score_rows[0]
    total = _strict_nonnegative_integer(header.get("total_count"), field="total_count")
    correct = _strict_nonnegative_integer(
        header.get("correct_count"), field="correct_count"
    )
    accuracy = header.get("accuracy")
    if total != EXPECTED_N or correct > total:
        raise BFCLComparisonInputError(
            "official BFCL score header is inconsistent with n=150"
        )
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not math.isfinite(float(accuracy))
        or not math.isclose(
            float(accuracy), correct / total, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise BFCLComparisonInputError("official BFCL score accuracy is inconsistent")

    failure_rows = score_rows[1:]
    if len(failure_rows) != total - correct:
        raise BFCLComparisonInputError(
            "official BFCL score failure count is inconsistent"
        )
    incorrect_ids: set[str] = set()
    for row in failure_rows:
        case_id = row.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise BFCLComparisonInputError("official BFCL failure row has a missing ID")
        if case_id in incorrect_ids:
            raise BFCLComparisonInputError(
                "official BFCL score has a duplicate failure ID"
            )
        if case_id not in result_ids:
            raise BFCLComparisonInputError("official BFCL score and result IDs differ")
        if row.get("valid") is not False:
            raise BFCLComparisonInputError(
                "official BFCL failure row is not marked invalid"
            )
        incorrect_ids.add(case_id)

    if not protocol_failure_ids.issubset(incorrect_ids):
        raise BFCLComparisonInputError("a BFCL protocol failure was scored correct")
    correct_ids = result_ids - incorrect_ids
    if len(correct_ids) != correct:
        raise BFCLComparisonInputError(
            "official BFCL per-case outcomes disagree with header"
        )
    return _OfficialArmOutcome(
        case_ids=frozenset(result_ids),
        correct_ids=frozenset(correct_ids),
        protocol_failure_ids=frozenset(protocol_failure_ids),
        result_sha256=result_hash,
        score_sha256=score_hash,
    )


def _percentage(count: int) -> float:
    return round(100.0 * count / EXPECTED_N, 6)


def _wilson_95(successes: int) -> dict[str, float | str]:
    proportion = successes / EXPECTED_N
    denominator = 1 + _Z_95 * _Z_95 / EXPECTED_N
    centre = (proportion + _Z_95 * _Z_95 / (2 * EXPECTED_N)) / denominator
    radius = (
        _Z_95
        * math.sqrt(
            (proportion * (1 - proportion) + _Z_95 * _Z_95 / (4 * EXPECTED_N))
            / EXPECTED_N
        )
        / denominator
    )
    return {
        "method": "wilson_95",
        "low": round(100.0 * max(0.0, centre - radius), 6),
        "high": round(100.0 * min(1.0, centre + radius), 6),
        "level": 95.0,
    }


def _exact_mcnemar(native_only: int, single_only: int) -> float:
    discordant = native_only + single_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, index)
        for index in range(min(native_only, single_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def compare_bfcl_arms(
    *,
    single_step_result: Path,
    single_step_score: Path,
    native_agent_result: Path,
    native_agent_score: Path,
    single_step_binding: Path,
    native_agent_binding: Path,
    native_agent_call_budget: int = 4,
) -> dict[str, Any]:
    """Return paired official BFCL outcomes with no case-level material."""

    native_agent_call_budget = _strict_positive_integer(
        native_agent_call_budget,
        field="native_agent_call_budget",
        maximum=BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN,
    )

    paths = (
        single_step_result,
        single_step_score,
        native_agent_result,
        native_agent_score,
        single_step_binding,
        native_agent_binding,
    )
    if len({path.resolve() for path in paths}) != len(paths):
        raise BFCLComparisonInputError("comparison requires six distinct artifacts")

    single_binding = _load_run_binding(
        single_step_binding, expected_arm=SINGLE_STEP_ARM
    )
    native_binding = _load_run_binding(
        native_agent_binding,
        expected_arm=NATIVE_AGENT_ARM,
        expected_native_call_budget=native_agent_call_budget,
    )
    matched_protocol = (
        "model_sha256",
        "sample_sha256",
        "temperature",
        "seed",
        "thinking_mode",
        "workers",
    )
    if any(
        getattr(single_binding, field) != getattr(native_binding, field)
        for field in matched_protocol
    ):
        raise BFCLComparisonInputError(
            "paired BFCL arms differ in model, sample, decoding, or worker protocol"
        )
    single = _load_official_arm(single_step_result, single_step_score)
    native = _load_official_arm(native_agent_result, native_agent_score)
    if single.case_ids != native.case_ids:
        raise BFCLComparisonInputError("paired BFCL arms have different ID sets")

    both_correct = len(single.correct_ids & native.correct_ids)
    native_only = len(native.correct_ids - single.correct_ids)
    single_only = len(single.correct_ids - native.correct_ids)
    neither = EXPECTED_N - both_correct - native_only - single_only
    single_correct = len(single.correct_ids)
    native_correct = len(native.correct_ids)
    ordered_case_ids = tuple(sorted(single.case_ids))
    try:
        delta_uncertainty = paired_delta_bootstrap(
            tuple(case_id in native.correct_ids for case_id in ordered_case_ids),
            tuple(case_id in single.correct_ids for case_id in ordered_case_ids),
            expected_n=EXPECTED_N,
        )
    except PairedStatisticsError as error:
        raise BFCLComparisonInputError(
            "paired BFCL outcomes cannot satisfy the uncertainty protocol"
        ) from error
    cells = {
        "both_correct": both_correct,
        "native_tool_agent_only_rescue": native_only,
        "single_step_only_regression": single_only,
        "neither": neither,
    }
    return {
        "format_version": "round9-bfcl-paired-comparison-v3",
        "benchmark_label": "BFCL_v4_multi_turn_miss_param",
        "evaluation_scope": "partial:150_of_200",
        "leaderboard_comparable": False,
        "official_evaluator_commit": OFFICIAL_COMMIT,
        "official_correctness_source": "official_score_failure_rows",
        "n": EXPECTED_N,
        "arms": {
            SINGLE_STEP_ARM: {
                "correct": single_correct,
                "denominator": EXPECTED_N,
                "accuracy_percent": _percentage(single_correct),
                "wilson_95_percent": _wilson_95(single_correct),
                "protocol_failures": len(single.protocol_failure_ids),
                "protocol_failure_denominator": EXPECTED_N,
                "protocol_failure_percent": _percentage(
                    len(single.protocol_failure_ids)
                ),
            },
            NATIVE_AGENT_ARM: {
                "correct": native_correct,
                "denominator": EXPECTED_N,
                "accuracy_percent": _percentage(native_correct),
                "wilson_95_percent": _wilson_95(native_correct),
                "protocol_failures": len(native.protocol_failure_ids),
                "protocol_failure_denominator": EXPECTED_N,
                "protocol_failure_percent": _percentage(
                    len(native.protocol_failure_ids)
                ),
            },
        },
        "paired_outcomes": {
            "raw_counts": cells,
            "percentages": {name: _percentage(count) for name, count in cells.items()},
        },
        "delta_percentage_points": round(
            100.0 * (native_correct - single_correct) / EXPECTED_N,
            6,
        ),
        "delta_confidence_interval_95": delta_uncertainty,
        "mcnemar_exact_two_sided_p": _exact_mcnemar(native_only, single_only),
        "multiplicity_policy": {
            "confirmatory_metric": "official_accuracy",
            "family_unit": "predeclared_agent_or_model_comparison",
            "rule": "holm_bonferroni_if_complete_family_has_more_than_one_comparison",
            "status": "family_not_inferred_from_a_single_comparison_artifact",
        },
        "paired_protocol_binding": {
            "same_model_sha256": single_binding.model_sha256,
            "same_sample_sha256": single_binding.sample_sha256,
            "temperature": single_binding.temperature,
            "seed": single_binding.seed,
            "thinking_mode": single_binding.thinking_mode,
            "configured_max_physical_calls_per_turn": {
                SINGLE_STEP_ARM: single_binding.max_physical_calls_per_turn,
                NATIVE_AGENT_ARM: native_binding.max_physical_calls_per_turn,
            },
            "worker_policy": {
                SINGLE_STEP_ARM: single_binding.worker_policy,
                NATIVE_AGENT_ARM: native_binding.worker_policy,
            },
            "workers": {
                SINGLE_STEP_ARM: single_binding.workers,
                NATIVE_AGENT_ARM: native_binding.workers,
            },
            "effective_calls_per_turn": {
                SINGLE_STEP_ARM: 1,
                NATIVE_AGENT_ARM: native_binding.max_physical_calls_per_turn,
            },
        },
        "hashes": {
            "single_step_result_sha256": single.result_sha256,
            "single_step_score_sha256": single.score_sha256,
            "native_agent_result_sha256": native.result_sha256,
            "native_agent_score_sha256": native.score_sha256,
            "single_step_binding_sha256": single_binding.sha256,
            "native_agent_binding_sha256": native_binding.sha256,
            "frozen_case_id_set_sha256": EXPECTED_CASE_ID_SET_SHA256,
        },
    }


__all__ = ["BFCLComparisonInputError", "compare_bfcl_arms"]

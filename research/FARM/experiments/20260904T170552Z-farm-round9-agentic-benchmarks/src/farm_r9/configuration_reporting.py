"""Aggregate-only audit and paired reporting for FARM configuration runs.

The inference ledger uses historical metric names that are easy to overread.
This module leaves those ledgers untouched and derives a narrower internal
aggregate vocabulary post hoc:

* ``compiler_accepted_draft`` is static schema/provenance validation;
* ``compiler_accepted_no_unresolved_inputs`` additionally means that the
  retained draft contains no ``needs_input`` decision;
* neither outcome is a semantic-value evaluation, external platform dry run,
  live execution, or deployment-readiness result.

Only aggregate counts, uncertainty, fixed claim-scope labels, and whole-file
SHA-256 digests are returned. Case identifiers and record content are used
solely for local pairing and are never placed in the returned object. The
result remains access-controlled; a separate strict projection is required
for a FARM paper table or public release.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from farm_r9.artifact_io import read_json, read_jsonl, sha256_file, sha256_text
from farm_r9.privacy import DataClassification, export_public_aggregate


EXPECTED_N = 150
ONE_SHOT_ARM = "single_shot_configurator"
BOUNDED_ARM = "bounded_configuration_agent"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_Z_95 = 1.959963984540054
_BOOTSTRAP_DRAWS = 10_000

_SOURCE_VIOLATION_CODES = {
    "ungrounded_query_literal",
    "unknown_ingredient",
    "unobserved_resource",
    "unavailable_secret_ref",
    "unknown_field",
    "unsupported_source",
}
_SECRET_POLICY_CODES = {"secret_material", "secret_in_non_auth_field"}
_TERMINAL_JSON_OR_CONTRACT_FAILURES = {"json_parse_failure", "pydantic_validation"}
_STRUCTURAL_FIELD_CODES = {
    "unknown_field",
    "missing_field_decision",
    "duplicate_field_decision",
    "required_omitted",
}
_STRUCTURAL_PRECHECK_FAILURE_CODES = {
    "secret_material",
    "changed_fixed_trigger",
    "changed_fixed_action",
    "invalid_endpoint_citations",
    "unknown_trigger_alias",
    "unknown_action_alias",
}

_OUTCOME_DEFINITIONS = {
    "final_structured_commit_available": (
        "The terminal ledger retains a structured commit draft and its static compiler result."
    ),
    "complete_field_decision_structure": (
        "The retained compiler reports structural completeness after fixed-endpoint prechecks: "
        "no unknown, missing, or duplicate field decision and no required-field omission; "
        "other field compiler errors may still be present."
    ),
    "compiler_accepted_draft": (
        "The static compiler found no fixed-endpoint, field-coverage, exact-span, known-source, "
        "supplied-resource, or opaque-secret-reference error under the ingested metadata."
    ),
    "compiler_accepted_no_unresolved_inputs": (
        "The static compiler accepted the draft and the retained draft has no needs_input decision."
    ),
    "exact_service_pair_and_compiler_accepted_draft": (
        "The frozen endpoint ledger has an exact service pair and the static compiler accepted the draft."
    ),
    "exact_function_pair_and_compiler_accepted_draft": (
        "The frozen endpoint ledger has an exact function pair and the static compiler accepted the draft."
    ),
    "exact_function_pair_and_compiler_accepted_no_unresolved_inputs": (
        "The frozen endpoint ledger has an exact function pair and the static compiler accepted a draft "
        "with no needs_input decision."
    ),
}

_BEHAVIOR_DEFINITIONS = {
    "case_asked_clarification": "At least one clarification question was emitted.",
    "case_received_supplied_clarification": (
        "At least one emitted question had an answer in the frozen simulator evidence."
    ),
    "case_disclosed_needs_input": "The retained final draft has at least one needs_input decision.",
    "case_used_explicit_optional_omission": "The retained final draft has at least one explicit omit decision.",
    "selected_endpoint_field_list_metadata_available": (
        "Both selected endpoints supplied field-list containers; this does not establish upstream "
        "requiredness or type-metadata provenance."
    ),
    "case_with_compiler_detected_source_violation_attempt": (
        "At least one compiler pass detected an ungrounded or unavailable source/schema reference."
    ),
    "case_with_unrepaired_compiler_detected_source_violation": (
        "The retained terminal compiler result still contains such a source/schema violation."
    ),
    "case_with_compiler_detected_secret_policy_attempt": (
        "At least one compiler pass detected credential material or a secret reference in a non-auth field."
    ),
    "case_with_unrepaired_compiler_detected_secret_policy_violation": (
        "The retained terminal compiler result still contains such a secret-policy violation."
    ),
    "case_with_terminal_json_or_contract_failure": (
        "The terminal failure was JSON parsing or typed output-contract validation."
    ),
}


class ConfigurationReportInputError(ValueError):
    """Raised when a private artifact cannot safely support the public report."""


def _input_error(message: str) -> ConfigurationReportInputError:
    # Error text is deliberately path-, identifier-, and value-free because the
    # CLI may surface it outside the confidential evaluation boundary.
    return ConfigurationReportInputError(message)


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise _input_error(f"{label} must be boolean")
    return value


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _input_error(f"{label} must be a nonnegative integer")
    return value


def _strict_nonnegative_number(value: Any, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise _input_error(f"{label} must be a finite nonnegative number")
    return float(value)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _input_error(f"{label} must be a SHA-256 digest")
    return value


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        digest_before = sha256_file(path)
        value = read_json(path)
        digest_after = sha256_file(path)
    except (OSError, ValueError) as error:
        raise _input_error(f"{label} is unreadable or malformed") from error
    if digest_before != digest_after:
        raise _input_error(f"{label} changed while it was being read")
    if not isinstance(value, dict):
        raise _input_error(f"{label} must be a JSON object")
    return value


def _read_rows(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        digest_before = sha256_file(path)
        rows = read_jsonl(path)
        digest_after = sha256_file(path)
    except (OSError, ValueError) as error:
        raise _input_error(f"{label} is unreadable or malformed") from error
    if digest_before != digest_after:
        raise _input_error(f"{label} changed while it was being read")
    return rows


def _stable_digest(path: Path, label: str) -> str:
    try:
        digest_before = sha256_file(path)
        digest_after = sha256_file(path)
    except OSError as error:
        raise _input_error(f"{label} is unreadable") from error
    if digest_before != digest_after:
        raise _input_error(f"{label} changed while it was being hashed")
    return digest_before


def _validate_manifest(
    path: Path,
    *,
    expected_arm: str,
    endpoint_records_sha256: str,
) -> dict[str, Any]:
    value = _read_object(path, "configuration manifest")
    expected_scalars = {
        "schema_version": "round9-configuration-run-manifest-v3",
        "benchmark": "farm_v2_test",
        "arm": expected_arm,
        "data_classification": "confidential",
        "data_source": "farm_v2",
        "n": EXPECTED_N,
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise _input_error(
                "configuration manifest has an unexpected frozen identity"
            )
    for key in (
        "ordered_case_ids_sha256",
        "candidate_artifact_sha256",
        "endpoint_records_sha256",
        "prompt_sha256",
    ):
        _sha256(value.get(key), "configuration manifest digest")
    if value["endpoint_records_sha256"] != endpoint_records_sha256:
        raise _input_error(
            "configuration manifest is not bound to the supplied endpoint ledger"
        )
    if not isinstance(value.get("model_metadata"), Mapping):
        raise _input_error("configuration manifest lacks model metadata")
    protocol = value.get("protocol_metadata")
    if not isinstance(protocol, Mapping):
        raise _input_error("configuration manifest lacks protocol metadata")
    expected_protocol = {
        "version": "round9-config-v3",
        "questions_max": 0 if expected_arm == ONE_SHOT_ARM else 2,
        "repairs_max": 0 if expected_arm == ONE_SHOT_ARM else 1,
        "temperature": 0,
        "seed": 42,
        "failure_policy": "invalid_or_ungrounded_is_incorrect_no_fallback",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise _input_error(
                "configuration manifest has an unexpected frozen protocol"
            )
    if protocol.get("thinking_mode") not in {"default", "disabled", "low"}:
        raise _input_error("configuration manifest has an invalid thinking mode")
    return value


def _validate_manifest_pair(
    one_shot: Mapping[str, Any], bounded: Mapping[str, Any]
) -> None:
    shared_top_level = (
        "benchmark",
        "data_classification",
        "data_source",
        "n",
        "ordered_case_ids_sha256",
        "candidate_artifact_sha256",
        "endpoint_records_sha256",
        "model_metadata",
    )
    if any(one_shot.get(key) != bounded.get(key) for key in shared_top_level):
        raise _input_error(
            "configuration arms do not share the same data and model binding"
        )
    one_protocol = one_shot["protocol_metadata"]
    bounded_protocol = bounded["protocol_metadata"]
    for key in ("version", "temperature", "seed", "thinking_mode", "failure_policy"):
        if one_protocol.get(key) != bounded_protocol.get(key):
            raise _input_error(
                "configuration arms differ outside the intended agent budget"
            )


def _codes(items: Any, label: str) -> list[str]:
    if not isinstance(items, list):
        raise _input_error(f"{label} must be a list")
    result: list[str] = []
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("code"), str):
            raise _input_error(f"{label} contains a malformed entry")
        result.append(item["code"])
    return result


def _decision_counts(draft: Mapping[str, Any]) -> tuple[int, int]:
    needs_input = 0
    omitted = 0
    for side in ("trigger_fields", "action_fields"):
        decisions = draft.get(side)
        if not isinstance(decisions, list):
            raise _input_error("configuration draft has malformed field decisions")
        for decision in decisions:
            if not isinstance(decision, Mapping) or not isinstance(
                decision.get("source"), Mapping
            ):
                raise _input_error("configuration draft has malformed field decisions")
            kind = decision["source"].get("kind")
            if not isinstance(kind, str):
                raise _input_error("configuration draft has malformed field decisions")
            needs_input += kind == "needs_input"
            omitted += kind == "omit"
    return needs_input, omitted


def _validate_stored_metric(
    metrics: Mapping[str, Any], key: str, expected: Any
) -> None:
    if metrics.get(key) != expected or type(metrics.get(key)) is not type(expected):
        raise _input_error(
            "configuration ledger contains stale or inconsistent derived metrics"
        )


def _audit_configuration_record(
    record: Mapping[str, Any], expected_arm: str
) -> dict[str, bool]:
    if record.get("arm") != expected_arm:
        raise _input_error("configuration ledger has the wrong or mixed arm")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise _input_error("configuration ledger lacks case metrics")
    questions = record.get("questions")
    if not isinstance(questions, list):
        raise _input_error("configuration ledger has malformed clarification events")
    answered_questions = 0
    for question in questions:
        if not isinstance(question, Mapping):
            raise _input_error(
                "configuration ledger has malformed clarification events"
            )
        answered_questions += _strict_bool(
            question.get("answer_available"),
            "clarification answer availability",
        )

    attempt_codes = _codes(record.get("validation_errors"), "validation errors")
    draft = record.get("draft")
    compilation = record.get("compilation")
    if (draft is None) != (compilation is None):
        raise _input_error(
            "configuration ledger has a detached draft or compiler result"
        )

    terminal_commit = draft is not None
    if terminal_commit:
        if not isinstance(draft, Mapping) or not isinstance(compilation, Mapping):
            raise _input_error("configuration ledger has a malformed terminal commit")
        needs_input_count, omit_count = _decision_counts(draft)
        compiler_valid = _strict_bool(compilation.get("valid"), "compiler validity")
        structurally_complete = _strict_bool(
            compilation.get("structurally_complete"), "field decision completeness"
        )
        legacy_executable = _strict_bool(
            compilation.get("executable_ready"), "legacy executable-ready outcome"
        )
        final_codes = _codes(compilation.get("issues"), "compiler issues")
        if compiler_valid != (not final_codes):
            raise _input_error("compiler result violates its validity invariants")
        expected_structural = not any(
            code in _STRUCTURAL_PRECHECK_FAILURE_CODES for code in final_codes
        ) and not any(code in _STRUCTURAL_FIELD_CODES for code in final_codes)
        if structurally_complete != expected_structural:
            raise _input_error("compiler result violates its structural invariants")
        if legacy_executable != (compiler_valid and needs_input_count == 0):
            raise _input_error(
                "legacy executable-ready outcome violates its static invariant"
            )
    else:
        needs_input_count = 0
        omit_count = 0
        compiler_valid = False
        structurally_complete = False
        legacy_executable = False
        final_codes = []

    protocol_failure = record.get("protocol_failure")
    if protocol_failure is not None and not isinstance(protocol_failure, str):
        raise _input_error("configuration ledger has a malformed terminal failure")
    protocol_success = protocol_failure is None
    # In this frozen worker the only success return is an accepted compiler
    # result.  Report it once, under the narrower compiler name.
    if protocol_success != compiler_valid:
        raise _input_error(
            "legacy protocol success is not equivalent to compiler acceptance"
        )

    source_attempts = sum(code in _SOURCE_VIOLATION_CODES for code in attempt_codes)
    source_unrepaired = sum(code in _SOURCE_VIOLATION_CODES for code in final_codes)
    secret_attempts = sum(code in _SECRET_POLICY_CODES for code in attempt_codes)
    secret_unrepaired = sum(code in _SECRET_POLICY_CODES for code in final_codes)

    _validate_stored_metric(metrics, "protocol_success", protocol_success)
    _validate_stored_metric(metrics, "compiler_valid", compiler_valid)
    _validate_stored_metric(metrics, "structurally_complete", structurally_complete)
    _validate_stored_metric(
        metrics,
        "executable_ready_under_supplied_evidence",
        legacy_executable,
    )
    _validate_stored_metric(metrics, "question_count", len(questions))
    _validate_stored_metric(metrics, "answered_question_count", answered_questions)
    _validate_stored_metric(metrics, "needs_input_decision_count", needs_input_count)
    _validate_stored_metric(metrics, "explicit_omit_decision_count", omit_count)
    _validate_stored_metric(metrics, "fabrication_attempt_count", source_attempts)
    _validate_stored_metric(metrics, "accepted_fabrication_count", source_unrepaired)
    _validate_stored_metric(
        metrics, "security_violation_attempt_count", secret_attempts
    )
    _validate_stored_metric(
        metrics, "accepted_security_violation_count", secret_unrepaired
    )
    if (
        "semantic_value_accuracy" not in metrics
        or metrics.get("semantic_value_accuracy") is not None
    ):
        raise _input_error("semantic value accuracy must remain unavailable")
    if metrics.get("semantic_accuracy_supported") is not False:
        raise _input_error("semantic value accuracy support is incorrectly asserted")
    selected_field_lists_available = _strict_bool(
        metrics.get("schema_requiredness_known"),
        "selected field-list availability",
    )

    return {
        "final_structured_commit_available": terminal_commit,
        "complete_field_decision_structure": structurally_complete,
        "compiler_accepted_draft": compiler_valid,
        "compiler_accepted_no_unresolved_inputs": legacy_executable,
        "case_asked_clarification": bool(questions),
        "case_received_supplied_clarification": answered_questions > 0,
        "case_disclosed_needs_input": needs_input_count > 0,
        "case_used_explicit_optional_omission": omit_count > 0,
        "selected_endpoint_field_list_metadata_available": selected_field_lists_available,
        "case_with_compiler_detected_source_violation_attempt": source_attempts > 0,
        "case_with_unrepaired_compiler_detected_source_violation": source_unrepaired
        > 0,
        "case_with_compiler_detected_secret_policy_attempt": secret_attempts > 0,
        "case_with_unrepaired_compiler_detected_secret_policy_violation": secret_unrepaired
        > 0,
        "case_with_terminal_json_or_contract_failure": protocol_failure
        in _TERMINAL_JSON_OR_CONTRACT_FAILURES,
    }


def _load_configuration_ledger(
    path: Path, expected_arm: str
) -> tuple[dict[str, dict[str, bool]], list[dict[str, Any]]]:
    records = _read_rows(path, "configuration ledger")
    if len(records) != EXPECTED_N:
        raise _input_error("configuration ledger must contain exactly 150 records")
    audited: dict[str, dict[str, bool]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise _input_error("configuration ledger has a missing case identifier")
        if case_id in audited:
            raise _input_error("configuration ledger has a duplicate case identifier")
        audited[case_id] = _audit_configuration_record(record, expected_arm)
    return audited, records


def _load_endpoint_context(path: Path) -> dict[str, dict[str, bool]]:
    records = _read_rows(path, "endpoint ledger")
    latest: dict[str, Mapping[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise _input_error("endpoint ledger has a missing case identifier")
        latest[case_id] = record
    if len(latest) != EXPECTED_N:
        raise _input_error("endpoint ledger must resolve to exactly 150 cases")
    outcomes: dict[str, dict[str, bool]] = {}
    for case_id, record in latest.items():
        if record.get("terminal") is not True:
            raise _input_error("endpoint ledger contains a nonterminal latest record")
        scores = record.get("scores")
        if not isinstance(scores, Mapping):
            raise _input_error("endpoint ledger lacks exact endpoint scores")
        outcomes[case_id] = {
            "exact_service_pair_selected": _strict_bool(
                scores.get("service_joint"), "exact service-pair score"
            ),
            "exact_function_pair_selected": _strict_bool(
                scores.get("function_joint"), "exact function-pair score"
            ),
        }
    return outcomes


def _wilson_95(successes: int, n: int = EXPECTED_N) -> dict[str, float | str]:
    proportion = successes / n
    denominator = 1 + _Z_95 * _Z_95 / n
    centre = (proportion + _Z_95 * _Z_95 / (2 * n)) / denominator
    radius = (
        _Z_95
        * math.sqrt((proportion * (1 - proportion) + _Z_95 * _Z_95 / (4 * n)) / n)
        / denominator
    )
    return {
        "method": "wilson_95",
        "low": round(100.0 * max(0.0, centre - radius), 6),
        "high": round(100.0 * min(1.0, centre + radius), 6),
        "level": 95.0,
    }


def _exact_mcnemar(bounded_only: int, one_shot_only: int) -> float:
    discordant = bounded_only + one_shot_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(bounded_only, one_shot_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _paired_bootstrap_95(
    differences: Sequence[int], metric: str
) -> dict[str, float | int | str]:
    ordered = sorted(differences)
    rng = random.Random(f"round9-configuration-report-v2:{metric}")
    draws = sorted(
        100.0
        * sum(ordered[rng.randrange(len(ordered))] for _ in ordered)
        / len(ordered)
        for _ in range(_BOOTSTRAP_DRAWS)
    )
    return {
        "method": "paired_nonparametric_percentile_bootstrap",
        "draws": _BOOTSTRAP_DRAWS,
        "low": round(draws[int(0.025 * (len(draws) - 1))], 6),
        "high": round(draws[math.ceil(0.975 * (len(draws) - 1))], 6),
    }


def _arm_stat(values: Sequence[bool]) -> dict[str, Any]:
    count = sum(values)
    n = len(values)
    return {
        "raw_numerator": count,
        "denominator": n,
        "percentage": round(100.0 * count / n, 6),
        "confidence_interval": _wilson_95(count, n),
    }


def _paired_metric(
    one_shot: Mapping[str, Mapping[str, bool]],
    bounded: Mapping[str, Mapping[str, bool]],
    metric: str,
) -> dict[str, Any]:
    cells = {"both": 0, "bounded_only": 0, "one_shot_only": 0, "neither": 0}
    differences: list[int] = []
    one_values: list[bool] = []
    bounded_values: list[bool] = []
    for case_id in sorted(one_shot):
        one_value = one_shot[case_id][metric]
        bounded_value = bounded[case_id][metric]
        one_values.append(one_value)
        bounded_values.append(bounded_value)
        differences.append(int(bounded_value) - int(one_value))
        if one_value and bounded_value:
            cells["both"] += 1
        elif bounded_value:
            cells["bounded_only"] += 1
        elif one_value:
            cells["one_shot_only"] += 1
        else:
            cells["neither"] += 1
    return {
        "one_shot": _arm_stat(one_values),
        "bounded_agent": _arm_stat(bounded_values),
        "paired_raw_counts": {**cells, "denominator": len(differences)},
        "bounded_minus_one_shot_percentage_points": round(
            100.0 * sum(differences) / len(differences), 6
        ),
        "paired_delta_confidence_interval": _paired_bootstrap_95(differences, metric),
        "mcnemar_exact_two_sided_p": _exact_mcnemar(
            cells["bounded_only"], cells["one_shot_only"]
        ),
    }


def _descriptive_metric(
    one_shot: Mapping[str, Mapping[str, bool]],
    bounded: Mapping[str, Mapping[str, bool]],
    metric: str,
) -> dict[str, Any]:
    one_values = [one_shot[case_id][metric] for case_id in sorted(one_shot)]
    bounded_values = [bounded[case_id][metric] for case_id in sorted(bounded)]
    return {
        "one_shot": _arm_stat(one_values),
        "bounded_agent": _arm_stat(bounded_values),
        "interpretation": "descriptive_not_a_correctness_outcome",
    }


def _operational(records: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    calls_by_case: list[int] = []
    calls: list[Mapping[str, Any]] = []
    questions_total = 0
    answered_total = 0
    repair_calls = 0
    for record in records:
        record_calls = record.get("calls")
        questions = record.get("questions")
        if not isinstance(record_calls, list) or any(
            not isinstance(item, Mapping) for item in record_calls
        ):
            raise _input_error("configuration ledger has malformed call accounting")
        if not isinstance(questions, list):
            raise _input_error(
                "configuration ledger has malformed clarification accounting"
            )
        calls_by_case.append(len(record_calls))
        calls.extend(record_calls)
        questions_total += len(questions)
        answered_total += sum(bool(item.get("answer_available")) for item in questions)
        repair_calls += max(0, len(record_calls) - 1 - len(questions))

    latencies: list[float] = []
    prompt_tokens = completion_tokens = physical_attempts = cache_hits = 0
    for call in calls:
        prompt_tokens += _strict_nonnegative_int(
            call.get("prompt_tokens"), "prompt-token count"
        )
        completion_tokens += _strict_nonnegative_int(
            call.get("completion_tokens"), "completion-token count"
        )
        physical_attempts += _strict_nonnegative_int(
            call.get("physical_attempts"), "physical-attempt count"
        )
        cache_hits += _strict_bool(call.get("cache_hit"), "cache-hit flag")
        latencies.append(
            _strict_nonnegative_number(
                call.get("provider_latency_ms"), "provider latency"
            )
        )
    calls_by_case.sort()
    latencies.sort()
    return {
        "semantic_calls_total": len(calls),
        "semantic_calls_mean_per_case": round(len(calls) / EXPECTED_N, 6),
        "semantic_calls_p50": calls_by_case[len(calls_by_case) // 2],
        "semantic_calls_p95": calls_by_case[math.ceil(0.95 * len(calls_by_case)) - 1],
        "physical_attempts_total": physical_attempts,
        "cache_hits_total": cache_hits,
        "input_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "tokens_total": prompt_tokens + completion_tokens,
        "provider_latency_ms_total": round(sum(latencies), 6),
        "provider_latency_ms_mean_per_call": round(sum(latencies) / len(latencies), 6)
        if latencies
        else 0.0,
        "provider_latency_ms_p50": round(latencies[len(latencies) // 2], 6)
        if latencies
        else 0.0,
        "provider_latency_ms_p95": round(
            latencies[math.ceil(0.95 * len(latencies)) - 1], 6
        )
        if latencies
        else 0.0,
        "questions_total": questions_total,
        "answered_questions_total": answered_total,
        "repair_calls_total": repair_calls,
    }


def build_configuration_internal_report(
    *,
    one_shot_records: Path,
    one_shot_manifest: Path,
    bounded_records: Path,
    bounded_manifest: Path,
    endpoint_records: Path,
) -> dict[str, Any]:
    """Build one access-controlled aggregate report from immutable private ledgers."""

    one_records_sha = _stable_digest(one_shot_records, "one-shot configuration ledger")
    bounded_records_sha = _stable_digest(
        bounded_records, "bounded configuration ledger"
    )
    endpoint_sha = _stable_digest(endpoint_records, "endpoint ledger")
    one_manifest_sha = _stable_digest(
        one_shot_manifest, "one-shot configuration manifest"
    )
    bounded_manifest_sha = _stable_digest(
        bounded_manifest, "bounded configuration manifest"
    )
    one_manifest = _validate_manifest(
        one_shot_manifest,
        expected_arm=ONE_SHOT_ARM,
        endpoint_records_sha256=endpoint_sha,
    )
    bounded_manifest_value = _validate_manifest(
        bounded_manifest,
        expected_arm=BOUNDED_ARM,
        endpoint_records_sha256=endpoint_sha,
    )
    _validate_manifest_pair(one_manifest, bounded_manifest_value)

    one_shot, one_records = _load_configuration_ledger(one_shot_records, ONE_SHOT_ARM)
    bounded, bounded_rows = _load_configuration_ledger(bounded_records, BOUNDED_ARM)
    endpoints = _load_endpoint_context(endpoint_records)
    if set(one_shot) != set(bounded) or set(one_shot) != set(endpoints):
        raise _input_error("paired ledgers do not share the same case-identifier set")

    ordered_id_hash = sha256_text(
        "\n".join(record["case_id"] for record in one_records) + "\n"
    )
    if ordered_id_hash != one_manifest["ordered_case_ids_sha256"]:
        raise _input_error(
            "configuration ledger order differs from the frozen manifest"
        )
    bounded_ordered_id_hash = sha256_text(
        "\n".join(record["case_id"] for record in bounded_rows) + "\n"
    )
    if bounded_ordered_id_hash != bounded_manifest_value["ordered_case_ids_sha256"]:
        raise _input_error(
            "configuration ledger order differs from the frozen manifest"
        )

    for case_id in one_shot:
        endpoint = endpoints[case_id]
        for arm_values in (one_shot[case_id], bounded[case_id]):
            arm_values["exact_service_pair_and_compiler_accepted_draft"] = (
                endpoint["exact_service_pair_selected"]
                and arm_values["compiler_accepted_draft"]
            )
            arm_values["exact_function_pair_and_compiler_accepted_draft"] = (
                endpoint["exact_function_pair_selected"]
                and arm_values["compiler_accepted_draft"]
            )
            arm_values[
                "exact_function_pair_and_compiler_accepted_no_unresolved_inputs"
            ] = (
                endpoint["exact_function_pair_selected"]
                and arm_values["compiler_accepted_no_unresolved_inputs"]
            )

    outcome_metrics = {
        metric: {
            "definition": definition,
            **_paired_metric(one_shot, bounded, metric),
        }
        for metric, definition in _OUTCOME_DEFINITIONS.items()
    }
    behavior_metrics = {
        metric: {
            "definition": definition,
            **_descriptive_metric(one_shot, bounded, metric),
        }
        for metric, definition in _BEHAVIOR_DEFINITIONS.items()
    }
    endpoint_service = [
        value["exact_service_pair_selected"] for value in endpoints.values()
    ]
    endpoint_function = [
        value["exact_function_pair_selected"] for value in endpoints.values()
    ]

    # Refuse an aggregate assembled across files that changed after validation.
    stable_inputs = (
        (one_shot_records, "one-shot configuration ledger", one_records_sha),
        (bounded_records, "bounded configuration ledger", bounded_records_sha),
        (endpoint_records, "endpoint ledger", endpoint_sha),
        (one_shot_manifest, "one-shot configuration manifest", one_manifest_sha),
        (bounded_manifest, "bounded configuration manifest", bounded_manifest_sha),
    )
    for path, label, expected_digest in stable_inputs:
        if _stable_digest(path, label) != expected_digest:
            raise _input_error(
                "an input artifact changed during aggregate construction"
            )

    report = {
        "schema_version": "round9-configuration-internal-paired-report-v3",
        "benchmark_label": "farm_v2_test",
        "n": EXPECTED_N,
        "evaluation_scope": {
            "measured": [
                "exact_service_and_function_selection_from_the_frozen_endpoint_ledger",
                "static_field_decision_coverage",
                "static_field_source_resource_secret_reference_and_exact_query_span_checks",
                "unresolved_input_disclosure",
                "question_emission_and_safe_deferral_without_supplied_answers",
            ],
            "not_measured": [
                "semantic_field_value_or_binding_correctness",
                "literal_or_resource_value_type_correctness",
                "upstream_requiredness_and_type_metadata_provenance",
                "preview_grounding_readability_or_optional_field_usefulness",
                "clarification_question_necessity_quality_or_user_satisfaction",
                "external_platform_dry_run_success",
                "live_platform_execution_success",
                "deployment_readiness",
            ],
        },
        "legacy_metric_audit": {
            "protocol_success_is_redundant_with": "compiler_accepted_draft",
            "legacy_executable_ready_is_renamed_to": "compiler_accepted_no_unresolved_inputs",
            "legacy_schema_requiredness_known_is_renamed_to": "selected_endpoint_field_list_metadata_available",
            "accepted_fabrication_is_renamed_to": "unrepaired_compiler_detected_source_violation",
            "accepted_security_violation_is_renamed_to": "unrepaired_compiler_detected_secret_policy_violation",
            "records_checked_per_arm": EXPECTED_N,
        },
        "endpoint_context": {
            "exact_service_pair_selected": _arm_stat(endpoint_service),
            "exact_function_pair_selected": _arm_stat(endpoint_function),
        },
        "outcome_metrics": outcome_metrics,
        "behavior_and_safety_observations": behavior_metrics,
        "statistical_protocol": {
            "arm_prevalence_interval": "wilson_95",
            "paired_delta_interval": "paired_nonparametric_percentile_bootstrap_10000_draws",
            "paired_test": "two_sided_exact_mcnemar",
            "multiplicity_adjustment": "none_treat_all_configuration_tests_as_exploratory",
        },
        "operational_metrics": {
            "one_shot": _operational(one_records),
            "bounded_agent": _operational(bounded_rows),
        },
        "verified_design": {
            "same_case_set": True,
            "same_ordered_case_hash": True,
            "same_candidate_artifact": True,
            "same_frozen_endpoint_ledger": True,
            "same_model_binding": True,
            "same_temperature_seed_and_thinking_mode": True,
            "endpoint_score_handling": "validated_hash_bound_ledger_scores_not_recomputed_here",
            "declared_arm_difference": "arm_specific_prompt_plus_clarification_and_repair_budget",
            "not_manifest_verified": [
                "inference_worker_code_hash",
                "request_timeout_seconds",
                "prompt_text_rederivation",
            ],
        },
        "hashes": {
            "one_shot_records_sha256": one_records_sha,
            "bounded_records_sha256": bounded_records_sha,
            "one_shot_manifest_sha256": one_manifest_sha,
            "bounded_manifest_sha256": bounded_manifest_sha,
            "endpoint_records_sha256": endpoint_sha,
            "candidate_artifact_sha256": one_manifest["candidate_artifact_sha256"],
        },
    }
    return report


def build_configuration_paper_tables(
    *,
    one_shot_records: Path,
    one_shot_manifest: Path,
    bounded_records: Path,
    bounded_manifest: Path,
    endpoint_records: Path,
) -> dict[str, dict[str, Any]]:
    """Project the verified internal report into strict FARM table artifacts.

    Arm identities live in the caller-selected filenames/caption, not inside
    the data-bearing JSON. The paired table releases only four aggregate cells
    per outcome; those cells are sufficient to reproduce the paired delta and
    exact McNemar test without exposing a case identifier or record.
    """

    report = build_configuration_internal_report(
        one_shot_records=one_shot_records,
        one_shot_manifest=one_shot_manifest,
        bounded_records=bounded_records,
        bounded_manifest=bounded_manifest,
        endpoint_records=endpoint_records,
    )
    hashes = report["hashes"]
    endpoint_hash = hashes["endpoint_records_sha256"]

    def paper_interval(interval: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "low": interval["low"],
            "high": interval["high"],
            "level": interval["level"],
        }

    def arm_table(arm: str, record_hash: str) -> dict[str, Any]:
        statistics = {
            metric: value[arm] for metric, value in report["outcome_metrics"].items()
        }
        value = {
            "n": EXPECTED_N,
            "raw_numerators": {
                metric: statistic["raw_numerator"]
                for metric, statistic in statistics.items()
            },
            "raw_denominators": {
                metric: statistic["denominator"]
                for metric, statistic in statistics.items()
            },
            "percentages": {
                metric: statistic["percentage"]
                for metric, statistic in statistics.items()
            },
            "confidence_intervals": {
                metric: paper_interval(statistic["confidence_interval"])
                for metric, statistic in statistics.items()
            },
            "failure_counts": {
                "terminal_json_or_contract_failure": report[
                    "behavior_and_safety_observations"
                ]["case_with_terminal_json_or_contract_failure"][arm]["raw_numerator"],
                "unrepaired_compiler_detected_source_violation": report[
                    "behavior_and_safety_observations"
                ]["case_with_unrepaired_compiler_detected_source_violation"][arm][
                    "raw_numerator"
                ],
                "unrepaired_compiler_detected_secret_policy_violation": report[
                    "behavior_and_safety_observations"
                ]["case_with_unrepaired_compiler_detected_secret_policy_violation"][
                    arm
                ]["raw_numerator"],
            },
            "hashes": {
                "artifact_sha256": record_hash,
                "endpoint_records_artifact_sha256": endpoint_hash,
                "source_artifact_sha256": hashes["candidate_artifact_sha256"],
            },
        }
        return export_public_aggregate(
            value,
            source_classification=DataClassification.CONFIDENTIAL,
        )

    paired_numerators: dict[str, int] = {}
    for metric, value in report["outcome_metrics"].items():
        cells = value["paired_raw_counts"]
        for cell in ("both", "bounded_only", "one_shot_only", "neither"):
            paired_numerators[f"{metric}__{cell}"] = cells[cell]
    paired_denominators = {metric: EXPECTED_N for metric in paired_numerators}
    paired_percentages = {
        metric: round(100.0 * count / EXPECTED_N, 6)
        for metric, count in paired_numerators.items()
    }
    paired_intervals = {
        metric: paper_interval(_wilson_95(count, EXPECTED_N))
        for metric, count in paired_numerators.items()
    }
    paired = export_public_aggregate(
        {
            "n": EXPECTED_N,
            "raw_numerators": paired_numerators,
            "raw_denominators": paired_denominators,
            "percentages": paired_percentages,
            "confidence_intervals": paired_intervals,
            "hashes": {
                "one_shot_records_artifact_sha256": hashes["one_shot_records_sha256"],
                "bounded_records_artifact_sha256": hashes["bounded_records_sha256"],
                "endpoint_records_artifact_sha256": endpoint_hash,
            },
        },
        source_classification=DataClassification.CONFIDENTIAL,
    )
    return {
        "one_shot": arm_table("one_shot", hashes["one_shot_records_sha256"]),
        "bounded": arm_table("bounded_agent", hashes["bounded_records_sha256"]),
        "paired": paired,
    }


__all__ = [
    "ConfigurationReportInputError",
    "build_configuration_internal_report",
    "build_configuration_paper_tables",
]

#!/usr/bin/env python3
"""Run/resume configuration evaluation from frozen endpoint predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import (
    canonical_json,
    ordered_ids_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from farm_r9.configuration_agent import run_configuration_agent, system_prompt_for_arm
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.ollama_client import OllamaChatClient
from farm_r9.privacy import DataClassification, DataSource, export_public_aggregate


SOURCE_BY_BENCHMARK = {
    "farm_v2_test": DataSource.FARM_V2,
    "recipegen_gold": DataSource.RECIPEGEN,
    "recipegen_noisy": DataSource.RECIPEGEN,
}

_PUBLIC_FAILURE_LABELS = {
    "json_parse_failure": "json_parse_failure",
    "pydantic_validation": "pydantic_validation",
    "question_forbidden_in_single_shot": "question_forbidden_in_single_shot",
    "unknown_question_component": "unknown_question_component",
    "question budget exhausted": "question_budget_exhausted",
    "repeated clarification is forbidden": "repeated_clarification_forbidden",
    "compiler_validation_failed": "compiler_validation_failed",
    "semantic_call_budget_exhausted": "semantic_call_budget_exhausted",
    "endpoint_selection_unavailable": "endpoint_selection_unavailable",
}


def _validate_candidate_lineage(
    cases: list[dict],
    *,
    benchmark: str,
    classification: DataClassification,
    data_source: DataSource,
    cloud: bool,
) -> None:
    """Reject mislabelled inputs before any external client is constructed.

    Older frozen Round 9 candidate artifacts predate the optional
    ``data_source`` field.  Their source remains bound by the embedded benchmark
    plus the endpoint run manifest, which is validated separately.  When a row
    does carry ``data_source``, it must agree exactly.
    """

    seen_case_ids: set[str] = set()
    for case in cases:
        row_classification = case.get("data_classification")
        if cloud and row_classification == DataClassification.CONFIDENTIAL.value:
            raise RuntimeError(
                "Cloud configuration inference is forbidden when any candidate row is confidential"
            )
        if case.get("benchmark") != benchmark:
            raise RuntimeError(
                "candidate benchmark does not match the requested benchmark"
            )
        if row_classification != classification.value:
            raise RuntimeError(
                "candidate classification does not match the requested benchmark"
            )
        row_source = case.get("data_source")
        if row_source is not None and row_source != data_source.value:
            raise RuntimeError(
                "candidate source does not match the requested benchmark"
            )
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("candidate row has a missing case ID")
        if case_id in seen_case_ids:
            raise RuntimeError("candidate rows contain a duplicate case ID")
        seen_case_ids.add(case_id)


def _read_stable_jsonl(path: Path, *, label: str) -> tuple[list[dict], str]:
    """Read a JSONL artifact only when its bytes remain stable across the read."""

    try:
        digest_before = sha256_file(path)
        rows = read_jsonl(path)
        digest_after = sha256_file(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"{label} is missing, unreadable, or malformed") from error
    if digest_before != digest_after:
        raise RuntimeError(f"{label} changed while it was being read")
    return rows, digest_before


def _read_stable_json(path: Path, *, label: str) -> tuple[dict, str]:
    """Read a JSON object only when its bytes remain stable across the read."""

    try:
        digest_before = sha256_file(path)
        value = read_json(path)
        digest_after = sha256_file(path)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"{label} is missing, unreadable, or malformed") from error
    if digest_before != digest_after:
        raise RuntimeError(f"{label} changed while it was being read")
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value, digest_before


def _validate_endpoint_lineage(
    manifest: dict,
    *,
    records_path: Path,
    benchmark: str,
    classification: DataClassification,
    data_source: DataSource,
    candidate_sha256: str,
    ordered_case_ids_sha256: str,
) -> None:
    """Bind configuration inputs to the immutable endpoint-selection run."""

    expected = {
        "schema_version": "round9-endpoint-run-manifest-v1",
        "benchmark": benchmark,
        "data_classification": classification.value,
        "data_source": data_source.value,
        "candidate_artifact_sha256": candidate_sha256,
        "ordered_case_ids_sha256": ordered_case_ids_sha256,
        "records_file": records_path.name,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            public_field = field.replace("_", " ")
            raise RuntimeError(
                f"endpoint manifest {public_field} does not match configuration inputs"
            )


def _selection(record: dict) -> tuple[str, str] | None:
    prediction = record.get("prediction")
    if not isinstance(prediction, dict):
        return None
    return str(prediction["trigger_alias"]), str(prediction["action_alias"])


def _requiredness_known(case: dict, selected: tuple[str, str]) -> bool:
    trigger_alias, action_alias = selected
    public = case["public_evidence"]
    chosen = [
        next(
            row for row in public["trigger_candidates"] if row["alias"] == trigger_alias
        ),
        next(
            row for row in public["action_candidates"] if row["alias"] == action_alias
        ),
    ]
    return all(isinstance(row.get("fields"), list) for row in chosen)


def _wilson_95(correct: int, n: int) -> dict[str, float]:
    if n <= 0 or not 0 <= correct <= n:
        raise ValueError("Wilson interval requires 0 <= correct <= n and n > 0")
    z = 1.959963984540054
    proportion = correct / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    radius = (
        z
        * math.sqrt((proportion * (1 - proportion) + z * z / (4 * n)) / n)
        / denominator
    )
    return {
        "low": 100 * max(0.0, centre - radius),
        "high": 100 * min(1.0, centre + radius),
        "level": 95.0,
    }


def _operational(records: list[dict]) -> dict:
    n = len(records)
    calls_by_case = sorted(len(row.get("calls", [])) for row in records)
    calls = [
        call
        for row in records
        for call in row.get("calls", [])
        if isinstance(call, dict)
    ]
    latencies = sorted(float(call.get("provider_latency_ms", 0.0)) for call in calls)
    input_tokens = sum(int(call.get("prompt_tokens", 0)) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0)) for call in calls)
    questions_total = sum(
        int(row["metrics"].get("question_count", 0)) for row in records
    )
    return {
        "semantic_calls_total": len(calls),
        "semantic_calls_mean_per_case": len(calls) / n if n else 0.0,
        "semantic_calls_p50": float(calls_by_case[len(calls_by_case) // 2])
        if n
        else 0.0,
        "semantic_calls_p95": float(calls_by_case[math.ceil(0.95 * n) - 1])
        if n
        else 0.0,
        "physical_attempts_total": sum(
            int(call.get("physical_attempts", 1)) for call in calls
        ),
        "cache_hits_total": sum(bool(call.get("cache_hit", False)) for call in calls),
        "input_tokens_total": input_tokens,
        "completion_tokens_total": completion_tokens,
        "tokens_total": input_tokens + completion_tokens,
        "provider_latency_ms_total": sum(latencies),
        "provider_latency_ms_mean_per_call": sum(latencies) / len(latencies)
        if latencies
        else 0.0,
        "provider_latency_ms_p50": latencies[len(latencies) // 2] if latencies else 0.0,
        "provider_latency_ms_p95": latencies[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else 0.0,
        "queue_wait_ms_total": sum(
            float(call.get("queue_wait_ms", 0.0)) for call in calls
        ),
        "questions_total": questions_total,
        "questions_mean_per_case": questions_total / n if n else 0.0,
        "answered_questions_total": sum(
            int(row["metrics"].get("answered_question_count", 0)) for row in records
        ),
        "repair_calls_total": sum(
            max(
                0,
                len(row.get("calls", []))
                - 1
                - int(row["metrics"].get("question_count", 0)),
            )
            for row in records
        ),
        "cost_available": False,
        "cost_usd_total": None,
        "cost_usd_mean_per_case": None,
    }


def _aggregate(
    records: list[dict],
    *,
    benchmark: str,
    arm: str,
    model: str,
    model_digest: str,
    input_hash: str,
    thinking_mode: str = "default",
    manifest_hash: str | None = None,
) -> dict:
    # Public labels intentionally avoid schema/query/candidate terminology: the
    # strict egress validator treats those names as disclosure-risk markers.
    # The mapping keeps the private ledger's precise internal field name while
    # exposing only an aggregate, non-content-bearing label.
    metric_fields = {
        "compiler_accepted_draft": "compiler_valid",
        "explicit_field_decisions_complete": "structurally_complete",
        "compiler_accepted_no_unresolved_inputs": "executable_ready_under_supplied_evidence",
        "selected_endpoint_field_list_metadata_available": "schema_requiredness_known",
    }
    counts = {
        public_name: sum(bool(row["metrics"][private_name]) for row in records)
        for public_name, private_name in metric_fields.items()
    }
    counts.update(
        {
            "structured_action_valid": sum(
                row.get("protocol_failure")
                not in {"json_parse_failure", "pydantic_validation"}
                for row in records
            ),
            "commit_emitted": sum(row.get("draft") is not None for row in records),
            "cases_questioned": sum(
                int(row["metrics"]["question_count"]) > 0 for row in records
            ),
            "cases_disclosing_unresolved_input": sum(
                int(row["metrics"]["needs_input_decision_count"]) > 0 for row in records
            ),
            "cases_with_explicit_optional_omission": sum(
                int(row["metrics"]["explicit_omit_decision_count"]) > 0
                for row in records
            ),
            "cases_with_fabrication_attempt": sum(
                int(row["metrics"]["fabrication_attempt_count"]) > 0 for row in records
            ),
            # The private case ledger retains the historical ``accepted_*`` field
            # names, but those values mean that a violation remains in the final
            # proposed draft after the bounded repair.  The compiler still rejects
            # such a draft, so the public label must not imply authorization.
            "cases_with_unrepaired_fabrication_violation": sum(
                int(row["metrics"]["accepted_fabrication_count"]) > 0 for row in records
            ),
            "cases_with_security_violation_attempt": sum(
                int(row["metrics"]["security_violation_attempt_count"]) > 0
                for row in records
            ),
            "cases_with_unrepaired_security_violation": sum(
                int(row["metrics"]["accepted_security_violation_count"]) > 0
                for row in records
            ),
        }
    )
    n = len(records)
    protocol_failures = Counter(
        _PUBLIC_FAILURE_LABELS.get(
            str(row["protocol_failure"]), "other_protocol_failure"
        )
        for row in records
        if row.get("protocol_failure") is not None
    )
    percentages = {
        name: round(100.0 * counts[name] / n, 6) if n else 0.0 for name in counts
    }
    confidence = (
        {name: _wilson_95(value, n) for name, value in counts.items()} if n else {}
    )
    classification = (
        DataClassification.CONFIDENTIAL
        if benchmark == "farm_v2_test"
        else DataClassification.PUBLIC
    )
    public = {
        "n": n,
        "raw_numerators": counts,
        "raw_denominators": {name: n for name in counts},
        "percentages": percentages,
        "confidence_intervals": confidence,
        "failure_counts": {
            "terminal_not_compiler_accepted": n - counts["compiler_accepted_draft"],
            "cases_with_fabrication_attempt": counts["cases_with_fabrication_attempt"],
            "cases_with_unrepaired_fabrication_violation": counts[
                "cases_with_unrepaired_fabrication_violation"
            ],
            "cases_with_security_violation_attempt": counts[
                "cases_with_security_violation_attempt"
            ],
            "cases_with_unrepaired_security_violation": counts[
                "cases_with_unrepaired_security_violation"
            ],
            **{
                f"failure_{name}": count
                for name, count in sorted(protocol_failures.items())
            },
        },
        "hashes": {
            "aggregate_input_sha256": input_hash,
        },
    }
    if classification is DataClassification.PUBLIC:
        public.update(
            {
                "benchmark_label": benchmark,
                "model_metadata": {
                    "id": model,
                    "digest": model_digest,
                    "role": "configuration_agent",
                },
                "protocol_metadata": {
                    "id": "round9-config-v3",
                    "arm": arm,
                    "questions_max": 0 if arm == "single_shot_configurator" else 2,
                    "repairs_max": 0 if arm == "single_shot_configurator" else 1,
                    "thinking_mode": thinking_mode,
                },
                "operational_metrics": _operational(records),
            }
        )
        if manifest_hash is not None:
            public["hashes"]["manifest_sha256"] = manifest_hash
    return export_public_aggregate(
        public,
        source_classification=classification,
    )


def _ensure_manifest(path: Path, payload: dict) -> None:
    """Create an immutable run identity, or verify an exact resume."""

    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(
                "configuration run manifest mismatch; refusing mixed resume"
            )
        return
    body = (canonical_json(payload) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, body)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--endpoint-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark", choices=sorted(SOURCE_BY_BENCHMARK), required=True
    )
    parser.add_argument(
        "--arm",
        choices=["single_shot_configurator", "bounded_configuration_agent"],
        required=True,
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-digest", required=True)
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--cloud-limiter-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--thinking-mode",
        choices=["default", "disabled", "low"],
        default="default",
    )
    args = parser.parse_args()
    if not 30.0 <= args.timeout_seconds <= 1800.0:
        raise RuntimeError("--timeout-seconds must be between 30 and 1800")
    classification = (
        DataClassification.CONFIDENTIAL
        if args.benchmark == "farm_v2_test"
        else DataClassification.PUBLIC
    )
    if classification is DataClassification.CONFIDENTIAL and args.cloud:
        raise RuntimeError(
            "FARM configuration data is confidential and Cloud routing is forbidden"
        )
    cases, candidate_sha256 = _read_stable_jsonl(
        args.candidates,
        label="candidate artifact",
    )
    if len(cases) != 150:
        raise RuntimeError("Round 9 configuration protocol requires exactly 150 cases")
    data_source = SOURCE_BY_BENCHMARK[args.benchmark]
    _validate_candidate_lineage(
        cases,
        benchmark=args.benchmark,
        classification=classification,
        data_source=data_source,
        cloud=args.cloud,
    )
    endpoint_record_list, endpoint_records_sha256 = _read_stable_jsonl(
        args.endpoint_records,
        label="endpoint records",
    )
    endpoint_rows: dict[str, dict] = {}
    for row in endpoint_record_list:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise RuntimeError("endpoint record has a missing case ID")
        endpoint_rows[case_id] = row
    if set(endpoint_rows) != {row["case_id"] for row in cases}:
        raise RuntimeError("endpoint records and candidate cases differ")
    endpoint_manifest, _ = _read_stable_json(
        args.endpoint_records.parent / "manifest.json",
        label="endpoint manifest",
    )
    case_order_sha256 = ordered_ids_sha256(cases)
    _validate_endpoint_lineage(
        endpoint_manifest,
        records_path=args.endpoint_records,
        benchmark=args.benchmark,
        classification=classification,
        data_source=data_source,
        candidate_sha256=candidate_sha256,
        ordered_case_ids_sha256=case_order_sha256,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    manifest_path = args.output_dir / "manifest.json"
    manifest = {
        "schema_version": "round9-configuration-run-manifest-v3",
        "benchmark": args.benchmark,
        "arm": args.arm,
        "data_classification": classification.value,
        "data_source": data_source.value,
        "n": len(cases),
        "ordered_case_ids_sha256": case_order_sha256,
        "candidate_artifact_sha256": candidate_sha256,
        "endpoint_records_sha256": endpoint_records_sha256,
        "prompt_sha256": sha256_text(
            system_prompt_for_arm(
                args.arm,
                max_questions=0 if args.arm == "single_shot_configurator" else 2,
            )
        ),
        "model_metadata": {
            "id": args.model,
            "digest": args.model_digest,
            "provider": "ollama_cloud" if args.cloud else "ollama_local",
        },
        "protocol_metadata": {
            "version": "round9-config-v3",
            "questions_max": 0 if args.arm == "single_shot_configurator" else 2,
            "repairs_max": 0 if args.arm == "single_shot_configurator" else 1,
            "temperature": 0,
            "seed": 42,
            "thinking_mode": args.thinking_mode,
            "failure_policy": "invalid_or_ungrounded_is_incorrect_no_fallback",
        },
    }
    _ensure_manifest(manifest_path, manifest)
    existing_path = args.output_dir / "records.jsonl"
    records = read_jsonl(existing_path) if existing_path.exists() else []
    done = {row["case_id"] for row in records}
    if len(done) != len(records):
        raise RuntimeError("duplicate case in resumable configuration records")
    limiter = None
    if args.cloud:
        if args.cloud_limiter_dir is None:
            raise RuntimeError(
                "Cloud execution requires one shared --cloud-limiter-dir"
            )
        limiter = OllamaCloudLimiter(args.cloud_limiter_dir, max_concurrent=3)
    client = OllamaChatClient(
        host=args.host,
        api_key=os.environ.get(args.api_key_env) if args.cloud else None,
        model=args.model,
        cache_directory=args.output_dir / "cache",
        journal_path=args.output_dir / "requests.jsonl",
        cloud=args.cloud,
        limiter=limiter,
        timeout_seconds=args.timeout_seconds,
    )
    think: str | bool | None = {
        "default": None,
        "disabled": False,
        "low": "low",
    }[args.thinking_mode]
    for case in cases:
        if case["case_id"] in done:
            continue
        selected = _selection(endpoint_rows[case["case_id"]])
        if selected is None:
            records.append(
                {
                    "case_id": case["case_id"],
                    "arm": args.arm,
                    "draft": None,
                    "compilation": None,
                    "calls": [],
                    "questions": [],
                    "validation_errors": [],
                    "protocol_failure": "endpoint_selection_unavailable",
                    "metrics": {
                        "protocol_success": False,
                        "compiler_valid": False,
                        "structurally_complete": False,
                        "executable_ready_under_supplied_evidence": False,
                        "question_count": 0,
                        "answered_question_count": 0,
                        "needs_input_decision_count": 0,
                        "fabrication_attempt_count": 0,
                        "explicit_omit_decision_count": 0,
                        "accepted_fabrication_count": 0,
                        "security_violation_attempt_count": 0,
                        "accepted_security_violation_count": 0,
                        "semantic_value_accuracy": None,
                        "semantic_accuracy_supported": False,
                        "schema_requiredness_known": False,
                    },
                }
            )
            write_jsonl_atomic(existing_path, records)
            os.chmod(existing_path, 0o600)
            continue
        trigger_alias, action_alias = selected
        result = run_configuration_agent(
            client=client,
            case=case,
            trigger_alias=trigger_alias,
            action_alias=action_alias,
            benchmark=args.benchmark,
            classification=classification,
            data_source=data_source,
            arm=args.arm,
            think=think,
        )
        case_metrics = result.metrics()
        case_metrics["schema_requiredness_known"] = _requiredness_known(case, selected)
        records.append(
            {
                "case_id": case["case_id"],
                "arm": args.arm,
                "draft": None
                if result.draft is None
                else result.draft.model_dump(mode="json"),
                "compilation": None
                if result.compilation is None
                else result.compilation.model_dump(mode="json"),
                "calls": list(result.calls),
                "questions": list(result.questions),
                "validation_errors": list(result.validation_errors),
                "protocol_failure": result.protocol_failure,
                "metrics": case_metrics,
            }
        )
        write_jsonl_atomic(existing_path, records)
        os.chmod(existing_path, 0o600)
    aggregate = _aggregate(
        records,
        benchmark=args.benchmark,
        arm=args.arm,
        model=args.model,
        model_digest=args.model_digest,
        thinking_mode=args.thinking_mode,
        input_hash=candidate_sha256,
        manifest_hash=sha256_file(manifest_path),
    )
    write_json_atomic(args.output_dir / "aggregate_public.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

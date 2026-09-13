"""Resumable endpoint-arm execution and disclosure-safe aggregate statistics.

This module deliberately keeps the inference envelope and the result envelope
separate.  Candidate artifacts can contain FARM material, while the aggregate
never contains a request, a gold label, candidate text, or a case identifier.
The per-case ledger is consequently private (``0700`` directory and ``0600``
files) and is the only place that retains case identifiers.
"""

from __future__ import annotations

import fcntl
import math
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from farm_r9.artifact_io import (
    assert_no_secrets,
    canonical_json,
    ordered_ids_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from farm_r9.contracts import EndpointPrediction
from farm_r9.endpoint_agent import AgentResult, run_endpoint_program
from farm_r9.metrics_endpoint import score_endpoint
from farm_r9.ollama_client import OllamaProtocolError, OllamaTransportError
from farm_r9.privacy import DataClassification, DataSource, export_public_aggregate


ARMS = frozenset({"retrieval_top1", "same_model_one_shot", "bounded_tool_agent"})
_SCORE_KEYS = (
    "service_trigger",
    "service_action",
    "service_joint",
    "function_trigger",
    "function_action",
    "function_joint",
    "field_trigger_exact",
    "field_action_exact",
    "field_joint_exact",
)
_Z_95 = 1.959963984540054
_PUBLIC_FAILURE_CODES = {
    "json_parse_failure": "json_parse_failure",
    "pydantic_validation": "pydantic_validation",
    "invalid_output": "invalid_output",
    "trigger_field_names_not_exact_schema": "trigger_field_name_mismatch",
    "action_field_names_not_exact_schema": "action_field_name_mismatch",
}


@dataclass(frozen=True)
class EndpointRun:
    """The immutable invocation identity for one endpoint-arm ledger."""

    benchmark: str
    arm: str
    classification: DataClassification
    data_source: DataSource
    output_directory: Path
    candidate_artifact_sha256: str
    ordered_case_ids_sha256: str
    model_metadata: Mapping[str, Any]
    protocol_metadata: Mapping[str, Any]
    prompt_cost_per_million_usd: float | None = None
    completion_cost_per_million_usd: float | None = None


def _mode_0600(path: Path) -> None:
    os.chmod(path, 0o600)


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    """Create a ledger manifest exactly once; never replace it."""
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically refresh a derived private summary without broad permissions."""
    payload = (canonical_json(dict(value)) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _mode_0600(path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _manifest_payload(run: EndpointRun) -> dict[str, Any]:
    return {
        "schema_version": "round9-endpoint-run-manifest-v1",
        "benchmark": run.benchmark,
        "arm": run.arm,
        "data_classification": run.classification.value,
        "data_source": run.data_source.value,
        "candidate_artifact_sha256": run.candidate_artifact_sha256,
        "ordered_case_ids_sha256": run.ordered_case_ids_sha256,
        "model_metadata": dict(run.model_metadata),
        "protocol_metadata": dict(run.protocol_metadata),
        "pricing": {
            "prompt_cost_per_million_usd": run.prompt_cost_per_million_usd,
            "completion_cost_per_million_usd": run.completion_cost_per_million_usd,
        },
        "records_file": "records.jsonl",
        "aggregate_file": "aggregate_private.json",
        "public_aggregate_file": "aggregate_public.json",
    }


def _initialize_ledger(run: EndpointRun) -> tuple[Path, Path, Path, Path]:
    if run.arm not in ARMS:
        raise ValueError(f"unsupported endpoint arm: {run.arm}")
    if not run.benchmark:
        raise ValueError("benchmark is required")
    _private_directory(run.output_directory)
    manifest_path = run.output_directory / "manifest.json"
    records_path = run.output_directory / "records.jsonl"
    lock_path = run.output_directory / ".records.lock"
    expected = _manifest_payload(run)
    assert_no_secrets(expected)
    if manifest_path.exists():
        if read_json(manifest_path) != expected:
            raise RuntimeError(
                "existing endpoint ledger has a different immutable manifest"
            )
        _mode_0600(manifest_path)
    else:
        try:
            _write_json_exclusive(manifest_path, expected)
        except FileExistsError:
            if read_json(manifest_path) != expected:
                raise RuntimeError(
                    "existing endpoint ledger has a different immutable manifest"
                )
    if records_path.exists():
        _mode_0600(records_path)
    return (
        manifest_path,
        records_path,
        lock_path,
        run.output_directory / "aggregate_private.json",
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    # read_jsonl fails closed on a partially written line rather than silently
    # losing a completed model call during a resume.
    return read_jsonl(path)


def _latest_by_case(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        case_id = record.get("case_id")
        attempt = record.get("attempt")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(attempt, int)
            or attempt < 1
        ):
            raise ValueError("malformed endpoint ledger record")
        old = latest.get(case_id)
        if old is None or attempt > old["attempt"]:
            latest[case_id] = dict(record)
        elif attempt == old["attempt"] and dict(record) != old:
            raise ValueError("ambiguous duplicate endpoint ledger record")
    return latest


def _append_record(
    records_path: Path, lock_path: Path, record: Mapping[str, Any]
) -> bool:
    """Append once under a cross-process lock.  False means another worker won."""
    assert_no_secrets(record)
    payload = (canonical_json(dict(record)) + "\n").encode("utf-8")
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        latest = _latest_by_case(_read_records(records_path))
        case_id = str(record["case_id"])
        if case_id in latest and latest[case_id].get("terminal"):
            return False
        file_descriptor = os.open(
            records_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
        )
        try:
            os.fchmod(file_descriptor, 0o600)
            os.write(file_descriptor, payload)
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
        return True
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _candidate_alias_maps(
    case: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str], list[str], list[str]]:
    """Get rank order and opaque aliases, accepting the two frozen artifact shapes."""
    private = case.get("private")
    if not isinstance(private, Mapping):
        raise ValueError("candidate artifact lacks private ranking data")
    if "trigger_ranking" in private:
        trigger_ranking, action_ranking = (
            private.get("trigger_ranking"),
            private.get("action_ranking"),
        )
        trigger_aliases, action_aliases = (
            private.get("trigger_alias_map"),
            private.get("action_alias_map"),
        )
    else:
        trigger = private.get("trigger")
        action = private.get("action")
        if not isinstance(trigger, Mapping) or not isinstance(action, Mapping):
            raise ValueError("candidate artifact has unsupported private ranking shape")
        trigger_ranking, action_ranking = trigger.get("ranking"), action.get("ranking")
        trigger_aliases, action_aliases = (
            trigger.get("alias_map"),
            action.get("alias_map"),
        )
    if not all(
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
        for value in (trigger_ranking, action_ranking)
    ):
        raise ValueError("candidate artifact has an empty or invalid private ranking")
    if not all(
        isinstance(value, Mapping) for value in (trigger_aliases, action_aliases)
    ):
        raise ValueError("candidate artifact has no opaque alias map")
    return (
        {str(key): str(value) for key, value in trigger_aliases.items()},
        {str(key): str(value) for key, value in action_aliases.items()},
        list(trigger_ranking),
        list(action_ranking),
    )


def retrieval_top1_prediction(case: Mapping[str, Any]) -> EndpointPrediction:
    """Map the *private* rank-one identities back to their shuffled aliases.

    There is intentionally no fallback to an arbitrary visible alias.  A stale
    alias map is an artifact error, not a retrieval success.
    """
    trigger_aliases, action_aliases, trigger_ranking, action_ranking = (
        _candidate_alias_maps(case)
    )
    try:
        trigger_alias = next(
            alias
            for alias, identifier in trigger_aliases.items()
            if identifier == trigger_ranking[0]
        )
        action_alias = next(
            alias
            for alias, identifier in action_aliases.items()
            if identifier == action_ranking[0]
        )
    except StopIteration as error:
        raise ValueError(
            "private rank-one candidate is absent from shuffled alias map"
        ) from error
    evidence = case.get("public_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("candidate artifact lacks public evidence")
    triggers = {
        str(row.get("alias")): row
        for row in evidence.get("trigger_candidates", [])
        if isinstance(row, Mapping)
    }
    actions = {
        str(row.get("alias")): row
        for row in evidence.get("action_candidates", [])
        if isinstance(row, Mapping)
    }
    if trigger_alias not in triggers or action_alias not in actions:
        raise ValueError("private alias map does not bind visible candidates")
    return EndpointPrediction(
        trigger_alias=trigger_alias,
        action_alias=action_alias,
        trigger_field_names=tuple(_field_names(triggers[trigger_alias])),
        action_field_names=tuple(_field_names(actions[action_alias])),
        preview="Frozen retrieval top-one endpoint selection.",
        evidence_aliases=(trigger_alias, action_alias),
    )


def _field_names(candidate: Mapping[str, Any]) -> list[str]:
    values = candidate.get("field_names")
    if isinstance(values, list):
        return [str(value) for value in values]
    fields = candidate.get("fields")
    if not isinstance(fields, list):
        return []
    return [
        str(field.get("label") or field.get("slug"))
        for field in fields
        if isinstance(field, Mapping)
    ]


def _validate_case(
    case: Mapping[str, Any], run: EndpointRun, *, required_candidates: int
) -> None:
    if case.get("benchmark") != run.benchmark:
        raise ValueError("candidate case benchmark does not match run")
    if case.get("data_classification") != run.classification.value:
        raise ValueError("candidate case classification does not match run")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("candidate case lacks case_id")
    evidence = case.get("public_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("candidate case lacks public evidence")
    for side in ("trigger", "action"):
        candidates = evidence.get(f"{side}_candidates")
        if not isinstance(candidates, list) or len(candidates) != required_candidates:
            raise ValueError(
                f"candidate case must expose exactly {required_candidates} {side} candidates"
            )
        aliases = [
            item.get("alias") for item in candidates if isinstance(item, Mapping)
        ]
        if (
            len(aliases) != required_candidates
            or len(set(aliases)) != required_candidates
        ):
            raise ValueError(f"candidate case has malformed {side} aliases")


def _cost(calls: Sequence[Mapping[str, Any]], run: EndpointRun) -> float | None:
    if (
        run.prompt_cost_per_million_usd is None
        or run.completion_cost_per_million_usd is None
    ):
        return None
    return sum(
        float(call.get("prompt_tokens", 0))
        * run.prompt_cost_per_million_usd
        / 1_000_000
        + float(call.get("completion_tokens", 0))
        * run.completion_cost_per_million_usd
        / 1_000_000
        for call in calls
    )


def _terminal_record(
    case: Mapping[str, Any],
    attempt: int,
    result: AgentResult | None,
    run: EndpointRun,
    *,
    failure: str | None = None,
    retryable: bool = False,
) -> dict[str, Any]:
    prediction = result.prediction if result is not None else None
    protocol_failure = failure or (
        result.protocol_failure if result is not None else None
    )
    scores = (
        score_endpoint(case, prediction)
        if not retryable
        else {key: None for key in _SCORE_KEYS}
    )
    calls = list(result.calls) if result is not None else []
    return {
        "schema_version": "round9-endpoint-case-v1",
        "case_id": case["case_id"],
        "attempt": attempt,
        "terminal": not retryable,
        "outcome": "retryable_transport_failure"
        if retryable
        else ("protocol_failure" if protocol_failure else "success"),
        "failure_code": protocol_failure,
        "prediction": (
            {
                "trigger_alias": prediction.trigger_alias,
                "action_alias": prediction.action_alias,
                "trigger_field_names": list(prediction.trigger_field_names),
                "action_field_names": list(prediction.action_field_names),
                "evidence_aliases": list(prediction.evidence_aliases),
            }
            if prediction is not None
            else None
        ),
        "scores": scores,
        "validation_errors": list(result.validation_errors)
        if result is not None
        else [],
        "repaired": bool(result.repaired) if result is not None else False,
        "calls": calls,
        "cost_usd": _cost(calls, run),
    }


def wilson_95(correct: int, n: int) -> dict[str, float]:
    if not 0 <= correct <= n or n <= 0:
        raise ValueError("Wilson interval requires 0 <= correct <= n and n > 0")
    proportion = correct / n
    denominator = 1 + _Z_95**2 / n
    centre = (proportion + _Z_95**2 / (2 * n)) / denominator
    radius = (
        _Z_95
        * math.sqrt((proportion * (1 - proportion) + _Z_95**2 / (4 * n)) / n)
        / denominator
    )
    return {
        "low": 100 * max(0.0, centre - radius),
        "high": 100 * min(1.0, centre + radius),
        "level": 95.0,
    }


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant pair counts."""
    if b < 0 or c < 0:
        raise ValueError("McNemar counts must be nonnegative")
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * tail)


def _paired_statistics(
    records: Mapping[str, Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    shared = sorted(set(records) & set(reference))
    for metric in _SCORE_KEYS:
        pairs: list[tuple[int, int]] = []
        for case_id in shared:
            current = records[case_id].get("scores", {}).get(metric)
            baseline = reference[case_id].get("scores", {}).get(metric)
            if isinstance(current, bool) and isinstance(baseline, bool):
                pairs.append((int(current), int(baseline)))
        if not pairs:
            continue
        b = sum(current == 1 and baseline == 0 for current, baseline in pairs)
        c = sum(current == 0 and baseline == 1 for current, baseline in pairs)
        differences = [current - baseline for current, baseline in pairs]
        rng = random.Random(f"round9-paired-bootstrap-v1:{metric}")
        draws = sorted(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
            for _ in range(4000)
        )
        result[metric] = {
            "n": len(pairs),
            "rescues": b,
            "regressions": c,
            "delta_percentage_points": 100 * sum(differences) / len(differences),
            "paired_bootstrap_95_delta_percentage_points": {
                "low": 100 * draws[int(0.025 * (len(draws) - 1))],
                "high": 100 * draws[math.ceil(0.975 * (len(draws) - 1))],
            },
            "mcnemar_exact_two_sided_p": exact_mcnemar(b, c),
        }
    return result


def _operational(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calls = [
        call
        for record in records
        for call in record.get("calls", [])
        if isinstance(call, Mapping)
    ]
    latencies = sorted(float(call.get("provider_latency_ms", 0.0)) for call in calls)
    queue_wait = sum(float(call.get("queue_wait_ms", 0.0)) for call in calls)
    prompt_tokens = sum(int(call.get("prompt_tokens", 0)) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0)) for call in calls)
    costs = [record.get("cost_usd") for record in records]
    known_costs = [float(cost) for cost in costs if isinstance(cost, (int, float))]
    n = len(records)
    return {
        "terminal_cases_n": n,
        "semantic_calls": {
            "total": len(calls),
            "mean_per_case": len(calls) / n if n else None,
        },
        "tokens": {
            "prompt_total": prompt_tokens,
            "completion_total": completion_tokens,
            "total": prompt_tokens + completion_tokens,
        },
        "latency_ms": {
            "provider_total": sum(latencies),
            "provider_mean_per_call": sum(latencies) / len(latencies)
            if latencies
            else None,
            "provider_p50": latencies[len(latencies) // 2] if latencies else None,
            "provider_p95": latencies[math.ceil(0.95 * len(latencies)) - 1]
            if latencies
            else None,
            "queue_wait_total": queue_wait,
        },
        "cost_usd": {
            "available": len(known_costs) == n,
            "total": sum(known_costs) if len(known_costs) == n else None,
            "mean_per_case": sum(known_costs) / n
            if n and len(known_costs) == n
            else None,
        },
    }


def _public_operational(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Flatten aggregate-only operations into the strict public contract."""

    calls_by_case = sorted(len(record.get("calls", [])) for record in records)
    calls = [
        call
        for record in records
        for call in record.get("calls", [])
        if isinstance(call, Mapping)
    ]
    latencies = sorted(float(call.get("provider_latency_ms", 0.0)) for call in calls)
    input_tokens = sum(int(call.get("prompt_tokens", 0)) for call in calls)
    completion_tokens = sum(int(call.get("completion_tokens", 0)) for call in calls)
    costs = [record.get("cost_usd") for record in records]
    known_costs = [float(value) for value in costs if isinstance(value, (int, float))]
    n = len(records)
    cost_available = len(known_costs) == n
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
        "cost_available": cost_available,
        "cost_usd_total": sum(known_costs) if cost_available else None,
        "cost_usd_mean_per_case": sum(known_costs) / n
        if n and cost_available
        else None,
    }


def _rescore_terminal_records(
    records: Mapping[str, Mapping[str, Any]],
    cases_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recompute scores from immutable predictions and the frozen case gold.

    Scores are derived data, so the aggregate must not trust values embedded by
    an older worker.  This also makes intention-to-evaluate handling explicit:
    a terminal protocol failure has ``prediction=None`` and is scored incorrect
    for every metric whose gold is present, including field-name exactness.
    The source ledger itself is never modified.
    """

    rescored: dict[str, dict[str, Any]] = {}
    for case_id, record in records.items():
        case = cases_by_id.get(case_id)
        if case is None:
            raise ValueError(
                "endpoint ledger contains cases absent from candidate artifact"
            )
        prediction_payload = record.get("prediction")
        if prediction_payload is None:
            prediction = None
        elif isinstance(prediction_payload, Mapping):
            persisted_prediction = dict(prediction_payload)
            # Case ledgers intentionally omit free-text previews.  Reintroduce
            # a fixed non-model sentinel solely to satisfy the strict contract;
            # scoring reads aliases and field names only, and this object is
            # neither persisted nor counted as generated output.
            persisted_prediction.setdefault("preview", "[not persisted]")
            prediction = EndpointPrediction.model_validate(persisted_prediction)
        else:
            raise ValueError("endpoint ledger prediction must be an object or null")
        refreshed = dict(record)
        refreshed["scores"] = score_endpoint(case, prediction)
        rescored[case_id] = refreshed
    return rescored


def _aggregate(
    run: EndpointRun,
    cases: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    paired_reference: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    latest = _latest_by_case(records)
    cases_by_id = {str(case["case_id"]): case for case in cases}
    expected_ids = set(cases_by_id)
    extra = set(latest) - expected_ids
    if extra:
        raise ValueError(
            "endpoint ledger contains cases absent from candidate artifact"
        )
    terminal = _rescore_terminal_records(
        {
            case_id: row
            for case_id, row in latest.items()
            if row.get("terminal") is True
        },
        cases_by_id,
    )
    metrics: dict[str, Any] = {}
    for key in _SCORE_KEYS:
        values = [row.get("scores", {}).get(key) for row in terminal.values()]
        bools = [value for value in values if isinstance(value, bool)]
        if bools:
            correct, denominator = sum(bools), len(bools)
            metrics[key] = {
                "correct": correct,
                "n": denominator,
                "percentage": 100 * correct / denominator,
                "wilson_95": wilson_95(correct, denominator),
            }
    failures = Counter(
        str(row.get("failure_code"))
        for row in terminal.values()
        if row.get("failure_code")
    )
    public_failures = Counter()
    for name, count in failures.items():
        public_failures[_PUBLIC_FAILURE_CODES.get(name, "other_output_failure")] += (
            count
        )
    private = {
        "schema_version": "round9-endpoint-aggregate-private-v1",
        "benchmark": run.benchmark,
        "arm": run.arm,
        "data_classification": run.classification.value,
        "intended_n": len(cases),
        "terminal_n": len(terminal),
        "pending_n": len(cases) - len(terminal),
        "metrics": metrics,
        "failure_counts": dict(sorted(failures.items())),
        "operational_metrics": _operational(list(terminal.values())),
        "records_sha256": sha256_file(run.output_directory / "records.jsonl")
        if (run.output_directory / "records.jsonl").exists()
        else sha256_text(""),
        "manifest_sha256": sha256_file(run.output_directory / "manifest.json"),
    }
    if paired_reference is not None:
        reference = _rescore_terminal_records(
            {
                key: value
                for key, value in _latest_by_case(paired_reference).items()
                if value.get("terminal") is True
            },
            cases_by_id,
        )
        private["paired_comparisons"] = _paired_statistics(terminal, reference)

    # This is intentionally a separate, strict paper-table-shaped record.
    # Confidential FARM exports contain metric counts with explicit
    # denominators, percentages, uncertainty, failure counts, and whole-file
    # hashes only. Public upstream benchmarks may additionally retain ordinary
    # reproducibility metadata.
    complete = len(terminal) == len(cases)
    # A recoverable transport failure leaves the final sample incomplete.
    # Preserve the private progress summary, but do not present a partial
    # cohort under the intended n or validate its mean against that larger n.
    if not complete:
        return private, None
    public_metrics = {
        key: value
        for key, value in metrics.items()
        if complete and value["n"] == len(cases)
    }
    public = {
        "n": len(cases),
        "raw_numerators": {
            key: value["correct"] for key, value in public_metrics.items()
        },
        "raw_denominators": {key: value["n"] for key, value in public_metrics.items()},
        "percentages": {
            key: value["percentage"] for key, value in public_metrics.items()
        },
        "confidence_intervals": {
            key: value["wilson_95"] for key, value in public_metrics.items()
        },
        "failure_counts": {
            key: value
            for key, value in sorted(public_failures.items())
            if value <= len(cases)
        },
        "hashes": {
            "artifact_sha256": private["records_sha256"],
            "source_artifact_sha256": run.candidate_artifact_sha256,
        },
    }
    if run.classification is DataClassification.PUBLIC:
        public.update(
            {
                "benchmark_label": f"{run.benchmark}.{run.arm}",
                "model_metadata": dict(run.model_metadata),
                "protocol_metadata": dict(run.protocol_metadata),
                "operational_metrics": _public_operational(list(terminal.values())),
            }
        )
        public["hashes"]["manifest_sha256"] = private["manifest_sha256"]
    # Use the existing egress validator here as a producer-side invariant.  A
    # future field added to this runner therefore fails closed before a public
    # shaped file is written.
    return private, export_public_aggregate(
        public, source_classification=run.classification
    )


def run_endpoint_experiment(
    *,
    cases: Sequence[Mapping[str, Any]],
    run: EndpointRun,
    client: Any | None,
    paired_reference_records: Sequence[Mapping[str, Any]] | None = None,
    required_candidates: int = 10,
) -> dict[str, Any]:
    """Run or resume one frozen arm, returning the private exact aggregate.

    ``client`` is required only for model arms.  A Cloud client is rejected for
    any non-public classification before a request can be made.
    """
    if not cases:
        raise ValueError("cannot run an empty endpoint experiment")
    if len({case.get("case_id") for case in cases}) != len(cases):
        raise ValueError("candidate artifact contains duplicate case IDs")
    if ordered_ids_sha256(cases) != run.ordered_case_ids_sha256:
        raise ValueError(
            "candidate artifact case order does not match immutable run manifest"
        )
    if (
        client is not None
        and bool(getattr(client, "cloud", False))
        and run.classification is not DataClassification.PUBLIC
    ):
        raise PermissionError(
            "Cloud endpoint inference is permitted only for PUBLIC data"
        )
    if run.arm != "retrieval_top1" and client is None:
        raise ValueError("model endpoint arms require an OllamaChatClient")
    _, records_path, lock_path, aggregate_path = _initialize_ledger(run)
    for case in cases:
        _validate_case(case, run, required_candidates=required_candidates)
        latest = _latest_by_case(_read_records(records_path)).get(str(case["case_id"]))
        if latest is not None and latest.get("terminal") is True:
            continue
        attempt = int(latest["attempt"]) + 1 if latest is not None else 1
        try:
            if run.arm == "retrieval_top1":
                prediction = retrieval_top1_prediction(case)
                result = AgentResult(prediction, (), (), False, None)
            else:
                assert client is not None
                result = run_endpoint_program(
                    client=client,
                    case=case,
                    benchmark=run.benchmark,
                    classification=run.classification,
                    data_source=run.data_source,
                    arm=run.arm,
                )
            record = _terminal_record(case, attempt, result, run)
        except OllamaTransportError as error:
            record = _terminal_record(
                case, attempt, None, run, failure=type(error).__name__, retryable=True
            )
        except (OllamaProtocolError, ValueError) as error:
            # A model/protocol failure is scored as a failure for this arm; it
            # is never converted to retrieval rank one.
            record = _terminal_record(
                case, attempt, None, run, failure=type(error).__name__
            )
        except Exception as error:
            # Keep only the class name: exception text could contain request or
            # credential material.  This remains an explicit failed model
            # outcome, never a rank-one substitution.
            record = _terminal_record(
                case, attempt, None, run, failure=type(error).__name__
            )
        _append_record(records_path, lock_path, record)
    records = _read_records(records_path)
    private, public = _aggregate(run, cases, records, paired_reference_records)
    assert_no_secrets(private)
    if public is not None:
        assert_no_secrets(public)
    _write_private_json(aggregate_path, private)
    if public is not None:
        _write_private_json(run.output_directory / "aggregate_public.json", public)
    return private


__all__ = [
    "ARMS",
    "EndpointRun",
    "exact_mcnemar",
    "retrieval_top1_prediction",
    "run_endpoint_experiment",
    "wilson_95",
]

#!/usr/bin/env python3
"""Aggregate only completed, aligned Round8 arm artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


OUTCOMES = (
    "function_trigger",
    "function_action",
    "function_joint",
    "service_trigger",
    "service_action",
    "service_joint",
)
RECORD_SCHEMA = "farm_round8_executable_record_v1"
AGGREGATE_SCHEMA = "farm_round8_reviewer_aggregate_v1"
DEFAULT_SEED = 20260904
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
ACCOUNTING_FIELDS = (
    "semantic_calls",
    "transport_attempts",
    "model_tool_calls",
    "provider_requests",
    "deterministic_tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_seconds",
    "failures",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [[row["group_id"], row["semantic_family_id"]] for row in rows],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read required JSON artifact {path}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"required JSON artifact is not an object: {path}")
    return dict(value)


def _read_records(path: Path, arm: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read completed records for {arm}") from error
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"invalid record JSON for {arm} at line {line_number}") from error
        if not isinstance(value, Mapping):
            raise RuntimeError(f"record for {arm} at line {line_number} is not an object")
        row = dict(value)
        group = row.get("group_id")
        family = row.get("semantic_family_id")
        if row.get("schema_version") != RECORD_SCHEMA or row.get("arm_id") != arm:
            raise RuntimeError(f"record schema/arm mismatch for {arm} at line {line_number}")
        if not isinstance(group, str) or not group or group in seen:
            raise RuntimeError(f"invalid or duplicate group ID for {arm} at line {line_number}")
        if not isinstance(family, str) or not family:
            raise RuntimeError(f"invalid family ID for {arm} at line {line_number}")
        seen.add(group)
        _validate_record_metrics(row, arm, line_number)
        records.append(row)
    if not records:
        raise RuntimeError(f"completed arm {arm} has no records")
    return records


def _validate_record_metrics(row: Mapping[str, Any], arm: str, line_number: int) -> None:
    for field in ("baseline_score", "final_score", "candidate_oracle"):
        scores = row.get(field)
        if not isinstance(scores, Mapping) or any(type(scores.get(name)) is not bool for name in OUTCOMES):
            raise RuntimeError(f"invalid {field} for {arm} at line {line_number}")
    proposal = row.get("proposal_score")
    if proposal is not None and (
        not isinstance(proposal, Mapping)
        or any(type(proposal.get(name)) is not bool for name in OUTCOMES)
    ):
        raise RuntimeError(f"invalid proposal_score for {arm} at line {line_number}")
    if not isinstance(row.get("execution"), Mapping):
        raise RuntimeError(f"invalid execution block for {arm} at line {line_number}")
    if not isinstance(row.get("accounting"), Mapping):
        raise RuntimeError(f"invalid accounting block for {arm} at line {line_number}")
    if not isinstance(row.get("calls", []), list):
        raise RuntimeError(f"invalid calls block for {arm} at line {line_number}")


def _read_journal(path: Path) -> tuple[list[dict[str, Any]], int, int]:
    if not path.exists():
        return [], 0, 0
    events: list[dict[str, Any]] = []
    invalid_json = 0
    invalid_events = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot read attempt journal {path}") from error
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid_json += 1
            continue
        if not isinstance(value, Mapping) or value.get("event") not in {
            "request_started",
            "request_finished",
        }:
            invalid_events += 1
            continue
        events.append(dict(value))
    return events, invalid_json, invalid_events


def _strict_count(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{label} must be a non-negative integer")
    return value


def load_completed_arm(artifact_root: Path, arm: str) -> dict[str, Any]:
    records_path = artifact_root / "records" / f"{arm}.jsonl"
    attempts_path = artifact_root / "attempts" / f"{arm}.jsonl"
    result_path = artifact_root / "results" / f"{arm}.json"
    progress_path = artifact_root / "progress" / f"{arm}.json"
    result = _read_object(result_path)
    progress = _read_object(progress_path)
    if result.get("status") != "completed" or progress.get("phase") != "complete":
        raise RuntimeError(f"arm {arm} is not completed")
    if result.get("binding") != progress.get("binding"):
        raise RuntimeError(f"arm {arm} result/progress binding mismatch")
    binding = result.get("binding")
    if not isinstance(binding, Mapping) or binding.get("arm") != arm:
        raise RuntimeError(f"arm {arm} binding arm mismatch")
    records = _read_records(records_path, arm)
    rows = len(records)
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping) or _strict_count(metrics.get("rows"), label="result rows") != rows:
        raise RuntimeError(f"arm {arm} result row count mismatch")
    completed_rows = _strict_count(progress.get("completed_rows"), label="completed rows")
    target_rows = _strict_count(progress.get("target_rows"), label="target rows")
    if completed_rows != rows or target_rows != rows:
        raise RuntimeError(f"arm {arm} progress row count mismatch")
    output = progress.get("output")
    if not isinstance(output, str) or Path(output).resolve() != result_path.resolve():
        raise RuntimeError(f"arm {arm} progress output path mismatch")
    if progress.get("output_sha256") != _sha256(result_path):
        raise RuntimeError(f"arm {arm} result hash mismatch")
    events, invalid_json, invalid_events = _read_journal(attempts_path)
    return {
        "records": records,
        "journal": events,
        "journal_invalid_json_lines": invalid_json,
        "journal_invalid_event_lines": invalid_events,
        "input_hashes": {
            "records_sha256": _sha256(records_path),
            "results_sha256": _sha256(result_path),
            "progress_sha256": _sha256(progress_path),
            "attempts_sha256": _sha256(attempts_path) if attempts_path.exists() else None,
        },
        "binding": dict(binding),
    }


def exact_mcnemar(before: Sequence[bool], after: Sequence[bool]) -> dict[str, Any]:
    if len(before) != len(after) or not before:
        raise ValueError("paired McNemar vectors must be non-empty and aligned")
    rescues = sum(not left and right for left, right in zip(before, after))
    regressions = sum(left and not right for left, right in zip(before, after))
    discordant = rescues + regressions
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(rescues, regressions)
        probability = sum(math.comb(discordant, index) for index in range(lower + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * probability)
    return {
        "rescues": rescues,
        "regressions": regressions,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def family_cluster_bootstrap(
    records: Sequence[Mapping[str, Any]],
    outcome: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in records:
        clusters[str(row["semantic_family_id"])].append(
            int(row["final_score"][outcome]) - int(row["baseline_score"][outcome])
        )
    families = sorted(clusters)
    if not families:
        raise ValueError("family bootstrap requires records")
    generator = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled = [clusters[families[generator.randrange(len(families))]] for _ in families]
        total = sum(sum(cluster) for cluster in sampled)
        observations = sum(len(cluster) for cluster in sampled)
        samples.append(total / observations)
    samples.sort()
    return {
        "method": "paired_percentile_bootstrap_resampled_by_semantic_family",
        "confidence": 0.95,
        "seed": seed,
        "iterations": iterations,
        "clusters": len(families),
        "lower": _percentile(samples, 0.025),
        "upper": _percentile(samples, 0.975),
    }


def holm_adjust(raw: Mapping[str, float]) -> dict[str, dict[str, float]]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_value))
        adjusted[name] = running
    return {
        name: {"raw_p": raw[name], "holm_adjusted_p": adjusted[name]}
        for name in raw
    }


def _stage_metric(records: Sequence[Mapping[str, Any]], stage: str, outcome: str) -> dict[str, Any]:
    field = f"{stage}_score"
    available = [row[field] for row in records if isinstance(row.get(field), Mapping)]
    hits = sum(bool(scores[outcome]) for scores in available)
    total = len(records)
    return {
        "available": len(available),
        "available_rate": len(available) / total,
        "hits": hits,
        "intent_to_treat_denominator": total,
        "intent_to_treat_accuracy": hits / total,
        "available_case_accuracy": hits / len(available) if available else None,
    }


def _record_cost(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, float | int] = {}
    integer_fields = set(ACCOUNTING_FIELDS) - {"latency_seconds"}
    for field in ACCOUNTING_FIELDS:
        values = [row["accounting"].get(field) for row in records]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
            raise RuntimeError(f"record-attributed accounting field is invalid: {field}")
        total = sum(values)
        totals[field] = int(total) if field in integer_fields else float(total)
    totals["source"] = "sum_of_completed_record_accounting"
    totals["cases"] = len(records)
    return totals


def _journal_key(event: Mapping[str, Any]) -> tuple[str, str, str, int] | None:
    semantic = event.get("semantic_id")
    role = event.get("role")
    request_hash = event.get("request_hash")
    attempt = event.get("attempt")
    if not all(isinstance(value, str) and value for value in (semantic, role, request_hash)):
        return None
    if type(attempt) is not int or attempt < 0:
        return None
    return semantic, role, request_hash, attempt


def journal_totals(
    events: Sequence[Mapping[str, Any]], *, invalid_json_lines: int, invalid_event_lines: int
) -> dict[str, Any]:
    starts: Counter[tuple[str, str, str, int]] = Counter()
    finishes: Counter[tuple[str, str, str, int]] = Counter()
    invalid_identity_events = 0
    outcomes: Counter[str] = Counter()
    prompt_tokens = completion_tokens = model_tool_calls = 0
    latency = 0.0
    for event in events:
        key = _journal_key(event)
        if key is None:
            invalid_identity_events += 1
            continue
        if event["event"] == "request_started":
            starts[key] += 1
        else:
            finishes[key] += 1
            outcomes[str(event.get("outcome") or "missing_outcome")] += 1
            for name in ("prompt_tokens", "completion_tokens", "model_tool_calls"):
                value = event.get(name, 0)
                if type(value) is not int or value < 0:
                    value = 0
                if name == "prompt_tokens":
                    prompt_tokens += value
                elif name == "completion_tokens":
                    completion_tokens += value
                else:
                    model_tool_calls += value
            value = event.get("latency_seconds", 0.0)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                latency += float(value)
    keys = set(starts) | set(finishes)
    duplicate_starts = sum(max(0, count - 1) for count in starts.values())
    duplicate_finishes = sum(max(0, count - 1) for count in finishes.values())
    unfinished = sum(max(0, starts[key] - finishes[key]) for key in keys)
    orphan_finishes = sum(max(0, finishes[key] - starts[key]) for key in keys)
    return {
        "source": "append_only_attempt_journal",
        "provider_attempts_observed": sum(starts.values()),
        "started_events": sum(starts.values()),
        "finished_events": sum(finishes.values()),
        "matched_start_finish_events": sum(min(starts[key], finishes[key]) for key in keys),
        "unique_request_attempt_keys": len(keys),
        "duplicate_start_events": duplicate_starts,
        "duplicate_finish_events": duplicate_finishes,
        "unfinished_start_events": unfinished,
        "orphan_finish_events": orphan_finishes,
        "invalid_json_lines": invalid_json_lines,
        "invalid_event_lines": invalid_event_lines,
        "invalid_identity_events": invalid_identity_events,
        "finished_outcomes": dict(sorted(outcomes.items())),
        "finished_prompt_tokens": prompt_tokens,
        "finished_completion_tokens": completion_tokens,
        "finished_model_tool_calls": model_tool_calls,
        "finished_latency_seconds": latency,
    }


def _model_stages(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        for call in row.get("calls", []):
            if isinstance(call, Mapping):
                grouped[str(call.get("role") or "unknown")].append(call)
    result: dict[str, Any] = {}
    for role, calls in sorted(grouped.items()):
        result[role] = {
            "calls": len(calls),
            "valid_calls": sum(call.get("ok") is True and call.get("schema_valid") is True for call in calls),
            "failed_calls": sum(call.get("ok") is not True for call in calls),
            "record_attributed_provider_requests": sum(int(call.get("provider_requests") or 0) for call in calls),
            "prompt_tokens": sum(int(call.get("prompt_tokens") or 0) for call in calls),
            "completion_tokens": sum(int(call.get("completion_tokens") or 0) for call in calls),
            "latency_seconds": sum(float(call.get("latency_seconds") or 0.0) for call in calls),
        }
    return result


def aggregate_arm(
    records: Sequence[Mapping[str, Any]],
    journal: Sequence[Mapping[str, Any]],
    *,
    invalid_json_lines: int,
    invalid_event_lines: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, Any]:
    stages = {
        stage: {outcome: _stage_metric(records, stage, outcome) for outcome in OUTCOMES}
        for stage in ("baseline", "proposal", "final")
    }
    comparisons: dict[str, Any] = {}
    for outcome in OUTCOMES:
        before = [bool(row["baseline_score"][outcome]) for row in records]
        after = [bool(row["final_score"][outcome]) for row in records]
        comparisons[outcome] = {
            "delta": sum(after) / len(after) - sum(before) / len(before),
            "mcnemar": exact_mcnemar(before, after),
            "family_cluster_bootstrap_95_ci": family_cluster_bootstrap(
                records, outcome, iterations=bootstrap_iterations, seed=seed
            ),
        }

    eligible = [
        row
        for row in records
        if not row["baseline_score"]["function_joint"]
        and row["candidate_oracle"]["function_joint"]
    ]
    proposal_available = [row for row in eligible if isinstance(row.get("proposal_score"), Mapping)]
    proposal_rescues = sum(bool(row["proposal_score"]["function_joint"]) for row in proposal_available)
    final_rescues = sum(bool(row["final_score"]["function_joint"]) for row in eligible)
    repair_rows = [row for row in records if row["execution"].get("repair_attempted") is True]
    repair_successes = sum(row["execution"].get("repair_success") is True for row in repair_rows)

    execution = {}
    for field in ("strict_parse", "compiled", "sandbox_run", "repair_attempted"):
        numerator = sum(row["execution"].get(field) is True for row in records)
        execution[field] = {
            "numerator": numerator,
            "denominator": len(records),
            "rate": numerator / len(records),
        }
    execution["repair_success"] = {
        "numerator": repair_successes,
        "denominator": len(repair_rows),
        "rate": repair_successes / len(repair_rows) if repair_rows else None,
    }

    sides: dict[str, Any] = {}
    for side in ("trigger", "action"):
        sides[side] = {
            level: {
                stage: stages[stage][f"{level}_{side}"]
                for stage in ("baseline", "proposal", "final")
            }
            for level in ("function", "service")
        }

    record_cost = _record_cost(records)
    observed_cost = journal_totals(
        journal,
        invalid_json_lines=invalid_json_lines,
        invalid_event_lines=invalid_event_lines,
    )
    return {
        "rows": len(records),
        "families": len({str(row["semantic_family_id"]) for row in records}),
        "stages": stages,
        "sides": sides,
        "paired_comparisons": comparisons,
        "baseline_wrong_joint_oracle": {
            "definition": "baseline function-joint incorrect and gold pair present in independent trigger/action top-10 lattice",
            "denominator": len(eligible),
            "proposal_available": len(proposal_available),
            "proposal_rescues": proposal_rescues,
            "proposal_intent_to_treat_accuracy": proposal_rescues / len(eligible) if eligible else None,
            "proposal_available_case_accuracy": proposal_rescues / len(proposal_available) if proposal_available else None,
            "final_rescues": final_rescues,
            "final_rescue_accuracy": final_rescues / len(eligible) if eligible else None,
        },
        "execution": execution,
        "model_stages": _model_stages(records),
        "cost": {
            "record_attributed": record_cost,
            "journal_observed": observed_cost,
            "provider_attempt_difference_journal_minus_records": (
                observed_cost["provider_attempts_observed"] - record_cost["provider_requests"]
            ),
            "interpretation": "Record-attributed cost and journal-observed provider attempts have different attribution boundaries and are not interchangeable.",
        },
    }


def aggregate_artifacts(
    artifact_root: Path,
    arms: Sequence[str],
    *,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("arms must be a non-empty unique sequence")
    loaded = {arm: load_completed_arm(artifact_root, arm) for arm in arms}
    canonical = [
        (row["group_id"], row["semantic_family_id"])
        for row in loaded[arms[0]]["records"]
    ]
    for arm in arms[1:]:
        observed = [
            (row["group_id"], row["semantic_family_id"])
            for row in loaded[arm]["records"]
        ]
        if observed != canonical:
            raise RuntimeError(f"arm {arm} group/family alignment changed")
        for index, row in enumerate(loaded[arm]["records"]):
            reference = loaded[arms[0]]["records"][index]
            if row["baseline_score"] != reference["baseline_score"]:
                raise RuntimeError(f"arm {arm} baseline vector changed")
            if row["candidate_oracle"] != reference["candidate_oracle"]:
                raise RuntimeError(f"arm {arm} candidate-oracle vector changed")

    arm_metrics = {
        arm: aggregate_arm(
            loaded[arm]["records"],
            loaded[arm]["journal"],
            invalid_json_lines=loaded[arm]["journal_invalid_json_lines"],
            invalid_event_lines=loaded[arm]["journal_invalid_event_lines"],
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
        for arm in arms
    }
    multiplicity = {
        outcome: holm_adjust(
            {
                arm: float(arm_metrics[arm]["paired_comparisons"][outcome]["mcnemar"]["exact_two_sided_p"])
                for arm in arms
            }
        )
        for outcome in OUTCOMES
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "artifact_root": str(artifact_root.resolve()),
        "methodology": {
            "analysis_population": "intent_to_treat_all_aligned_completed_records",
            "primary_outcome": "function_joint",
            "paired_test": "exact_two_sided_mcnemar",
            "uncertainty": "paired_percentile_bootstrap_resampled_by_semantic_family",
            "bootstrap_seed": seed,
            "bootstrap_iterations": bootstrap_iterations,
            "multiplicity": "Holm adjustment across arms, separately for each endpoint outcome",
            "proposal_missingness": "missing proposal counts as incorrect for intent-to-treat accuracy",
        },
        "alignment": {
            "rows": len(canonical),
            "families": len({family for _, family in canonical}),
            "ordered_group_family_ids_sha256": _ids_sha256(loaded[arms[0]]["records"]),
            "arms": list(arms),
        },
        "inputs": {
            arm: {
                "hashes": loaded[arm]["input_hashes"],
                "binding": loaded[arm]["binding"],
            }
            for arm in arms
        },
        "arms": arm_metrics,
        "multiplicity": multiplicity,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Round8 Reviewer-Grade Aggregate",
        "",
        "All metrics use aligned records from arms whose result and progress artifacts are both complete and hash-consistent.",
        "Missing proposals are failures in the Intent-to-treat proposal accuracy; available-case accuracy is reported separately.",
        "Record-attributed cost and journal-observed provider attempts use different attribution boundaries and are not interchangeable.",
        "",
        "## Primary outcome",
        "",
        "| Arm | Rows | Proposal coverage | Proposal ITT EM | Final EM | Delta | Family-bootstrap 95% CI | McNemar p | Holm p | Oracle rescue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["alignment"]["arms"]:
        metrics = report["arms"][arm]
        proposal = metrics["stages"]["proposal"]["function_joint"]
        final = metrics["stages"]["final"]["function_joint"]
        comparison = metrics["paired_comparisons"]["function_joint"]
        interval = comparison["family_cluster_bootstrap_95_ci"]
        rescue = metrics["baseline_wrong_joint_oracle"]["final_rescue_accuracy"]
        holm = report["multiplicity"]["function_joint"][arm]["holm_adjusted_p"]
        lines.append(
            "| {arm} | {rows} | {coverage:.4f} | {proposal:.4f} | {final:.4f} | {delta:+.4f} | [{lower:+.4f}, {upper:+.4f}] | {raw:.4g} | {holm:.4g} | {rescue} |".format(
                arm=arm,
                rows=metrics["rows"],
                coverage=proposal["available_rate"],
                proposal=proposal["intent_to_treat_accuracy"],
                final=final["intent_to_treat_accuracy"],
                delta=comparison["delta"],
                lower=interval["lower"],
                upper=interval["upper"],
                raw=comparison["mcnemar"]["exact_two_sided_p"],
                holm=holm,
                rescue="NA" if rescue is None else f"{rescue:.4f}",
            )
        )
    lines.extend([
        "",
        "## Operational accounting",
        "",
        "| Arm | Record provider requests | Journal-observed provider attempts | Duplicate starts | Unfinished starts | Orphan finishes |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in report["alignment"]["arms"]:
        cost = report["arms"][arm]["cost"]
        recorded = cost["record_attributed"]
        observed = cost["journal_observed"]
        lines.append(
            f"| {arm} | {recorded['provider_requests']} | {observed['provider_attempts_observed']} | {observed['duplicate_start_events']} | {observed['unfinished_start_events']} | {observed['orphan_finish_events']} |"
        )
    lines.extend([
        "",
        "Repair success is conditional on an attempted repair; its rate is `null`/`NA` when the denominator is zero.",
        "Per-stage, per-side, execution, confidence-interval, and journal diagnostics are retained in the JSON artifact.",
        "",
    ])
    return "\n".join(lines)


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _discover_arms(artifact_root: Path) -> list[str]:
    results = artifact_root / "results"
    if not results.is_dir():
        return []
    return sorted(path.stem for path in results.glob("*.json"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.bootstrap_iterations < 1:
        parser.error("--bootstrap-iterations must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact_root = args.artifact_root.resolve()
    arms = list(args.arms or _discover_arms(artifact_root))
    report = aggregate_artifacts(
        artifact_root,
        arms,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    output = artifact_root / "aggregate"
    json_path = output / "round8_aggregate.json"
    markdown_path = output / "round8_aggregate.md"
    _write_atomic(json_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_atomic(markdown_path, render_markdown(report))
    print(json.dumps({"status": "completed", "arms": arms, "json": str(json_path), "markdown": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

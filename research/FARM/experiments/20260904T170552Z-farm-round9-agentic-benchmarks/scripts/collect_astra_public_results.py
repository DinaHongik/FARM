#!/usr/bin/env python3
"""Collect only complete public runs; no network, model calls, or source edits.

Reports are immutable and aggregate-only. A pending arm never contributes an
interim accuracy. Raw ledgers stay in their original access-controlled folders.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.artifact_io import read_json, read_jsonl, sha256_file
from farm_r9.paired_statistics import paired_delta_bootstrap
from farm_r9.yao_agent import wilson_interval, exact_mcnemar

N = 150
ARMS = {
    "yao_text_one_shot": "yao_v2/deepseek/same_model_one_shot",
    "yao_text_adaptive": "yao_v2/deepseek/bounded_clarification_agent",
    "yao_native_one_shot": "yao_native_astra_v3/native_one_shot",
    "yao_native_adaptive": "yao_native_astra_v3/native_adaptive",
    "yao_native_ask_all": "yao_native_astra_v3/native_ask_all",
    "catalog_fixed_top10": "catalog_search_astra_v1/native_fixed_top10",
    "catalog_search_agent": "catalog_search_astra_v1/native_search_agent",
}
PAIRS = {
    "yao_text": ("yao_text_adaptive", "yao_text_one_shot"),
    "yao_native_vs_one_shot": ("yao_native_adaptive", "yao_native_one_shot"),
    "yao_native_vs_ask_all": ("yao_native_adaptive", "yao_native_ask_all"),
    "catalog_search": ("catalog_search_agent", "catalog_fixed_top10"),
}


def save_once(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError("refusing to change a previously collected report")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def summarize(rows: list[dict], aggregate: dict, expected: dict, *, kind: str) -> dict:
    """Validate every intended case before emitting any accuracy or cost."""
    ids = [r.get("case_id") for r in rows]
    if (len(rows) != N or len(expected) != N or len(set(ids)) != N
            or set(ids) != set(expected) or aggregate.get("n") != N):
        raise ValueError("complete matched 150-case sample required")
    metrics = set(aggregate["raw_numerators"])
    if not metrics or kind not in ("yao", "catalog"):
        raise ValueError("invalid public result type")
    for row in rows:
        if row.get("terminal") is not True or set(row.get("scores", {})) != metrics:
            raise ValueError("nonterminal row or inconsistent metric schema")
        if any(type(v) is not bool for v in row["scores"].values()):
            raise ValueError("scores must be strict booleans")
        if kind == "yao" and row.get("stratum") != expected[row["case_id"]].get("stratum"):
            raise ValueError("frozen Yao stratum mismatch")
        if kind == "catalog":
            if any(type(row.get(k)) is not bool for k in ("initial_joint_coverage", "observed_joint_coverage")):
                raise ValueError("invalid coverage flag")
            if ((row["initial_joint_coverage"] or row["scores"]["function_joint"])
                    and not row["observed_joint_coverage"]):
                raise ValueError("observed candidate coverage invariant violated")
    counts = {k: sum(r["scores"][k] for r in rows) for k in sorted(metrics)}
    for k, value in counts.items():
        if (type(aggregate["raw_numerators"][k]) is not int
                or aggregate["raw_numerators"][k] != value
                or not math.isclose(aggregate["percentages"][k], 100 * value / N, abs_tol=1e-6)):
            raise ValueError("saved aggregate disagrees with terminal ledger")
    usage_keys = ("semantic_calls", "prompt_tokens", "completion_tokens", "physical_attempts",
                  "cache_hits", "provider_latency_ms", "queue_wait_ms")
    usage = {k: sum(r["usage"].get(k, 0) for r in rows) for k in usage_keys}
    report = {
        "n": N, "classification": "public_benchmark", "raw_numerators": counts,
        "raw_denominators": {k: N for k in counts},
        "percentages": {k: 100 * v / N for k, v in counts.items()},
        "wilson_95_percent": {k: wilson_interval(v, N) for k, v in counts.items()},
        "failure_counts": dict(Counter(r["failure_code"] for r in rows if r.get("failure_code"))),
        "usage_totals": usage, "usage_means_per_case": {k: v / N for k, v in usage.items()},
        "cost_boundary": "terminal-ledger usage; consult all request journals for exhausted transport attempts",
        "scope": "simulated_endpoint_clarification" if kind == "yao" else "public_catalog_endpoint_search",
        "live_applet_execution": False,
    }
    if kind == "yao":
        report["questions_total"] = sum(r["question_count"] for r in rows)
        report["questions_mean_per_case"] = report["questions_total"] / N
        if all("interaction_turns" in r for r in rows):
            report["interaction_turns_total"] = sum(r["interaction_turns"] for r in rows)
        report["strata"] = {s: {"n": sum(r["stratum"] == s for r in rows),
            "joint_correct": sum(r["stratum"] == s and r["scores"]["joint"] for r in rows)}
            for s in sorted({r["stratum"] for r in rows})}
    else:
        report["search_calls_total"] = sum(r["search_calls"] for r in rows)
        report["coverage_counts"] = {
            "initial_joint": sum(r["initial_joint_coverage"] for r in rows),
            "observed_joint": sum(r["observed_joint_coverage"] for r in rows),
            "successful_outside_initial_top10": sum(r["scores"]["function_joint"] and not
                r["initial_joint_coverage"] for r in rows),
        }
        report["field_metric_boundary"] = "selected metadata field names; not generated values or semantic bindings"
    return report


def compare_rows(current: list[dict], reference: list[dict], *, kind: str) -> dict:
    a, b = ({r["case_id"]: r for r in rows} for rows in (current, reference))
    if len(current) != N or len(reference) != N or len(a) != N or set(a) != set(b):
        raise ValueError("paired cohorts differ")
    metric = "joint" if kind == "yao" else "function_joint"
    ordered = sorted(a)
    x, y = ([mapping[i]["scores"][metric] for i in ordered] for mapping in (a, b))
    rescues = sum(v and not w for v, w in zip(x, y))
    regressions = sum(w and not v for v, w in zip(x, y))
    extra = {}
    if kind == "yao":
        strata = [a[i]["stratum"] for i in ordered]
        if strata != [b[i]["stratum"] for i in ordered]:
            raise ValueError("paired strata differ")
        extra = {"strata": strata, "expected_strata": dict(Counter(strata))}
    result = {"n": N, "metric": metric, "rescues": rescues, "regressions": regressions,
        "both_correct": sum(v and w for v, w in zip(x, y)),
        "both_wrong": sum(not v and not w for v, w in zip(x, y)),
        "paired_delta": paired_delta_bootstrap(x, y, expected_n=N, **extra),
        "mcnemar_exact_two_sided_p": exact_mcnemar(rescues, regressions),
        "interpretation": "exploratory; unequal information and call budgets; not architectural superiority",
        "multiplicity": "unadjusted exploratory comparison; do not present as confirmatory significance"}
    if kind == "catalog":
        if any(a[i]["initial_joint_coverage"] != b[i]["initial_joint_coverage"] for i in ordered):
            raise ValueError("catalog arms did not start from identical coverage")
        result["initially_uncovered"] = {
            "n": sum(not a[i]["initial_joint_coverage"] for i in ordered),
            "agent_correct": sum(a[i]["scores"][metric] and not a[i]["initial_joint_coverage"] for i in ordered),
            "reference_correct": sum(b[i]["scores"][metric] and not b[i]["initial_joint_coverage"] for i in ordered),
        }
    return result


def load_complete(base: Path, expected: dict, *, kind: str):
    ledger, aggregate_path = base / "records.jsonl", base / "aggregate.json"
    if not ledger.is_file() or not aggregate_path.is_file():
        return None
    before = sha256_file(ledger), sha256_file(aggregate_path)
    rows, aggregate = read_jsonl(ledger), read_json(aggregate_path)
    report = summarize(rows, aggregate, expected, kind=kind)
    if before != (sha256_file(ledger), sha256_file(aggregate_path)):
        raise ValueError("completed artifacts changed during collection")
    report["hashes"] = {"records_sha256": before[0], "aggregate_sha256": before[1]}
    return rows, report


def collect(root: Path, output: Path) -> dict:
    samples = {
        "yao": ("prepared/interactive_ifttt/cases.jsonl", "5543d5e331033147a960392c474f5bb2b1560e2d9a7ca1942e8de8d06aa345ff"),
        "catalog": ("derived/recipegen_noisy/candidates.jsonl", "34ba86f638024c4c9e834fa6020386006ec91f916dc039c92dcd95a867ec3009"),
    }
    expected = {}
    for kind, (relative, digest) in samples.items():
        path = root / relative
        if sha256_file(path) != digest:
            raise ValueError("frozen public source binding mismatch")
        expected[kind] = {r["case_id"]: r for r in read_jsonl(path)}
    complete, status = {}, {}
    for name, relative in ARMS.items():
        kind = "yao" if name.startswith("yao_") else "catalog"
        found = load_complete(root / "results" / relative, expected[kind], kind=kind)
        status[name] = "complete" if found else "pending"
        if found:
            rows, report = found
            complete[name] = rows
            save_once(output / "arms" / (name + ".json"), report)
    for name, (current, reference) in PAIRS.items():
        if current in complete and reference in complete:
            kind = "yao" if name.startswith("yao_") else "catalog"
            report = compare_rows(complete[current], complete[reference], kind=kind)
            report["arms"] = {"current": current, "reference": reference}
            report["hashes"] = {arm: sha256_file(output / "arms" / (arm + ".json"))
                                for arm in (current, reference)}
            save_once(output / "pairs" / (name + ".json"), report)
    # Official BFCL failures, not a custom parser, remain the source of truth.
    from farm_r9.bfcl_comparison import compare_bfcl_arms
    base = root / "results/bfcl-v4-missing-v2"
    current = root / "results/bfcl-v4-missing-v3-official-horizon"
    single, native = "farm-r9-bfcl-dsv4-single-v2", "farm-r9-bfcl-dsv4-agent-v3h21"
    basename = "BFCL_v4_multi_turn_miss_param"
    score = current / "scores-isolated-v1" / native / "multi_turn" / (basename + "_score.json")
    summary = current / "summaries" / (native + ".json")
    status["bfcl_official_horizon"] = "pending"
    if score.is_file() and summary.is_file():
        report = compare_bfcl_arms(
            single_step_result=base / "raw" / single / "multi_turn" / (basename + "_result.json"),
            single_step_score=base / "scores" / single / "multi_turn" / (basename + "_score.json"),
            native_agent_result=current / "raw" / native / "multi_turn" / (basename + "_result.json"),
            native_agent_score=score,
            single_step_binding=base / "raw" / single / "RUN_BINDING.json",
            native_agent_binding=current / "raw" / native / "RUN_BINDING.json",
            native_agent_call_budget=21)
        save_once(output / "pairs/bfcl_official_horizon.json", report)
        status["bfcl_official_horizon"] = "complete"
    if all(s == "complete" for s in status.values()):
        save_once(output / "COMPLETE.json", {"status": status,
            "report_hashes": {str(p.relative_to(output)): sha256_file(p)
                for directory in (output / "arms", output / "pairs") for p in sorted(directory.glob("*.json"))},
            "scope": "public benchmark aggregates only; no live FARM execution claim"})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)
    with (args.output / ".collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for _ in range(240 if args.watch else 1):
            event = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                event["runs"] = collect(args.round9_root.resolve(), args.output.resolve())
            except (OSError, ValueError, KeyError, TypeError) as error:
                # No content-bearing exception message is printed or released.
                event["collection_error_type"] = type(error).__name__
            with (args.output / "collector_progress.jsonl").open("a") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(json.dumps(event, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            print(json.dumps(event, sort_keys=True), flush=True)
            if (args.output / "COMPLETE.json").exists():
                return 0
            if args.watch:
                time.sleep(300)
        return 0 if "runs" in event else 2


if __name__ == "__main__":
    raise SystemExit(main())

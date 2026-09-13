#!/usr/bin/env python3
"""Replay completed agent traces and fail on the observed escalation defects.

This is deliberately a result-contract check rather than a model call.  It is
fast, deterministic, and safe to run against frozen development artifacts.
Future agent runs can use the same command as an acceptance gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SINGLE_ARM = "routed_plain"
ESCALATION_ARM = "routed_plain_to_schema"


def load_result(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "completed":
        raise ValueError(f"result is not complete: {path}")
    for arm in ("retrieval_top1", SINGLE_ARM, ESCALATION_ARM):
        if arm not in value.get("metrics", {}):
            raise ValueError(f"missing arm {arm}: {path}")
    return value


def audit_result(value: dict, *, confidence_threshold: float, max_fallback_rate: float) -> dict:
    rows = value["rows_detail"]
    single = value["metrics"][SINGLE_ARM]
    escalation = value["metrics"][ESCALATION_ARM]
    second_call_rows = []
    high_confidence_unrequested = []
    recoveries = 0
    regressions = 0
    rank6_10_exact = 0

    for row in rows:
        trace = row["arms"][ESCALATION_ARM]
        iterations = trace.get("iterations", [])
        if len(iterations) < 2:
            continue
        second_call_rows.append(row)
        first = iterations[0]["selection"]
        if (
            float(first.get("confidence", 0.0)) >= confidence_threshold
            and not bool(first.get("request_more", False))
        ):
            high_confidence_unrequested.append(row)
        flags = trace.get("iteration_exact_pair", [])
        if len(flags) >= 2:
            recoveries += int(not flags[0] and flags[1])
            regressions += int(flags[0] and not flags[1])
        if row.get("rank_bucket") == "rank6_10" and trace.get("exact_pair"):
            rank6_10_exact += 1

    logical_calls = int(escalation["total_calls"])
    fallback_count = int(escalation["protocol_fallbacks"])
    fallback_rate = fallback_count / logical_calls if logical_calls else 0.0
    single_accuracy = float(single["micro_exact_pair"])
    escalation_accuracy = float(escalation["micro_exact_pair"])

    defects = []
    if escalation_accuracy < single_accuracy:
        defects.append(
            "schema escalation reduced exact-pair accuracy "
            f"({escalation_accuracy:.6f} < {single_accuracy:.6f})"
        )
    if regressions > recoveries:
        defects.append(
            f"second iteration caused more regressions than recoveries ({regressions} > {recoveries})"
        )
    if high_confidence_unrequested:
        defects.append(
            "second iteration ran despite a high-confidence first answer that did not request more "
            f"({len(high_confidence_unrequested)} cases)"
        )
    if fallback_rate > max_fallback_rate:
        defects.append(
            f"protocol fallback rate exceeded budget ({fallback_rate:.3%} > {max_fallback_rate:.3%})"
        )

    return {
        "model": value["model"],
        "cases": len(rows),
        "single_accuracy": single_accuracy,
        "escalation_accuracy": escalation_accuracy,
        "accuracy_delta": escalation_accuracy - single_accuracy,
        "second_call_cases": len(second_call_rows),
        "second_call_recoveries": recoveries,
        "second_call_regressions": regressions,
        "high_confidence_unrequested_second_calls": len(high_confidence_unrequested),
        "protocol_fallbacks": fallback_count,
        "protocol_fallback_rate": fallback_rate,
        "rank6_10_exact": rank6_10_exact,
        "defects": defects,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--max-fallback-rate", type=float, default=0.05)
    args = parser.parse_args()

    reports = [
        audit_result(
            load_result(path),
            confidence_threshold=args.confidence_threshold,
            max_fallback_rate=args.max_fallback_rate,
        )
        for path in args.results
    ]
    print(json.dumps({"reports": reports}, indent=2, sort_keys=True))
    return int(any(report["defects"] for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge aligned round-four arms and produce the registered reviewer table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_metrics import evaluate_records, holm_adjust
from run_round4 import load_jsonl, write_json_atomic


MAIN_ARMS = (
    "gpt_top5_plain",
    "gpt_top10_plain",
    "gpt_service_to_function",
    "gemma_top5_plain_replication",
)
PRIMARY_ARM = "gpt_top5_plain"
REVERSED_ARM = "gpt_top10_reversed_audit"
BASELINE_ARM = "retrieval_top1"


def _arm_value(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "final_pair",
            "policy",
            "selected_service_pair",
            "calls",
            "accounting",
            "protocol_valid",
            "fallback_used",
            "retained_baseline",
        )
        if key in record
    }


def _base_value(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prediction": record["baseline_pair"],
        "protocol_valid": True,
        "fallback_used": False,
        "routed": False,
        "accounting": {
            "logical_calls": 0,
            "api_attempts": 0,
            "tool_calls": 0,
            "catalog_reads": 0,
        },
    }


def _gold_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "group_id",
            "query",
            "valid_pairs",
            "gold_service_pairs",
            "trigger_candidates",
            "action_candidates",
            "baseline_pair",
        )
    }


def merge_aligned(records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if not records_by_arm:
        raise ValueError("no arms supplied")
    first_name = next(iter(records_by_arm))
    first = list(records_by_arm[first_name])
    identifiers = [record["group_id"] for record in first]
    merged: list[dict[str, Any]] = []
    for index, base in enumerate(first):
        arms = {BASELINE_ARM: _base_value(base)}
        expected_gold = json.dumps(_gold_payload(base), sort_keys=True, ensure_ascii=False)
        for arm_name, arm_records in records_by_arm.items():
            if len(arm_records) != len(first) or arm_records[index]["group_id"] != identifiers[index]:
                raise ValueError(f"arm alignment changed: {arm_name}")
            observed_gold = json.dumps(
                _gold_payload(arm_records[index]), sort_keys=True, ensure_ascii=False
            )
            if observed_gold != expected_gold:
                raise ValueError(f"gold/candidate payload changed across arms: {arm_name}")
            arms[arm_name] = _arm_value(arm_records[index])
        merged.append(_gold_payload(base) | {"arms": arms})
    return merged


def _joint_rank_bucket(record: Mapping[str, Any]) -> str:
    trigger_rank = {
        item["url"]: int(item.get("retrieval_rank", index))
        for index, item in enumerate(record["trigger_candidates"], start=1)
    }
    action_rank = {
        item["url"]: int(item.get("retrieval_rank", index))
        for index, item in enumerate(record["action_candidates"], start=1)
    }
    ranks = [
        max(trigger_rank[pair["trigger_url"]], action_rank[pair["action_url"]])
        for pair in record["valid_pairs"]
        if pair["trigger_url"] in trigger_rank and pair["action_url"] in action_rank
    ]
    if not ranks:
        return "outside10"
    rank = min(ranks)
    if rank == 1:
        return "rank1"
    if rank <= 5:
        return "rank2_5"
    if rank <= 10:
        return "rank6_10"
    return "outside10"


def _exact(record: Mapping[str, Any], pair: Mapping[str, Any]) -> bool:
    selected = (pair.get("trigger_url"), pair.get("action_url"))
    valid = {(item["trigger_url"], item["action_url"]) for item in record["valid_pairs"]}
    return selected in valid


def bucket_transitions(records: Sequence[Mapping[str, Any]], arm_name: str) -> dict[str, Any]:
    rows: dict[str, list[tuple[bool, bool]]] = {
        "rank1": [], "rank2_5": [], "rank6_10": [], "outside10": [],
    }
    for record in records:
        bucket = _joint_rank_bucket(record)
        baseline = record["arms"][BASELINE_ARM]["prediction"]
        final = record["arms"][arm_name]["final_pair"]
        rows[bucket].append((_exact(record, baseline), _exact(record, final)))
    return {
        bucket: {
            "cases": len(values),
            "baseline_hits": sum(left for left, _ in values),
            "agent_hits": sum(right for _, right in values),
            "recoveries": sum(not left and right for left, right in values),
            "regressions": sum(left and not right for left, right in values),
        }
        for bucket, values in rows.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="new output path; defaults to results/combined_confirmation.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = (args.output or (run_root / "results" / "combined_confirmation.json")).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}")

    records_by_arm = {
        arm: load_jsonl(run_root / "records" / f"{arm}.jsonl") for arm in MAIN_ARMS
    }
    if any(len(rows) != 300 for rows in records_by_arm.values()):
        raise RuntimeError("all main confirmation arms must contain exactly 300 rows")
    merged = merge_aligned(records_by_arm)
    metrics = evaluate_records(
        merged,
        baseline_arm=BASELINE_ARM,
        arm_names=[BASELINE_ARM, *MAIN_ARMS],
        bootstrap_iterations=10000,
        bootstrap_seed=42,
        min_service_support=10,
    )

    primary_comparison = metrics["comparisons"][f"{PRIMARY_ARM}_vs_{BASELINE_ARM}"]
    primary = primary_comparison["outcomes"]["function_joint"]
    exploratory_raw: dict[str, float] = {}
    for arm in MAIN_ARMS:
        outcomes = metrics["comparisons"][f"{arm}_vs_{BASELINE_ARM}"]["outcomes"]
        for outcome, result in outcomes.items():
            if arm == PRIMARY_ARM and outcome == "function_joint":
                continue
            exploratory_raw[f"{arm}:{outcome}"] = float(result["raw_p"])
    exploratory_adjusted = holm_adjust(exploratory_raw)

    ranked_records = records_by_arm["gpt_top10_plain"][:100]
    reversed_records = load_jsonl(run_root / "records" / f"{REVERSED_ARM}.jsonl")
    if len(reversed_records) != 100:
        raise RuntimeError("reversed-order audit must contain exactly 100 rows")
    order_records = merge_aligned({
        "gpt_top10_ranked": ranked_records,
        "gpt_top10_reversed": reversed_records,
    })
    # Use ranked GPT top-10 as the paired order-control baseline.
    for record in order_records:
        record["arms"].pop(BASELINE_ARM)
    order_metrics = evaluate_records(
        order_records,
        baseline_arm="gpt_top10_ranked",
        arm_names=["gpt_top10_ranked", "gpt_top10_reversed"],
        bootstrap_iterations=10000,
        bootstrap_seed=42,
        min_service_support=5,
    )
    agreement = sum(
        left["final_pair"] == right["final_pair"]
        for left, right in zip(ranked_records, reversed_records)
    )

    transitions = {arm: bucket_transitions(merged, arm) for arm in MAIN_ARMS}
    protocol_gates = {}
    for arm in (*MAIN_ARMS, REVERSED_ARM):
        rows = records_by_arm.get(arm, reversed_records)
        valid = sum(bool(row["protocol_valid"]) for row in rows)
        protocol_gates[arm] = {
            "valid_cases": valid,
            "denominator_cases": len(rows),
            "rate": valid / len(rows),
            "minimum": 0.99,
            "passed": valid / len(rows) >= 0.99,
        }

    result = {
        "status": "completed",
        "schema_version": "farm_round4_combined_v2",
        "registered_primary": {
            "comparison": f"{PRIMARY_ARM}_vs_{BASELINE_ARM}",
            "outcome": "function_joint",
            "confirmatory_unadjusted": True,
            "baseline_rate": primary["baseline_rate"],
            "arm_rate": primary["arm_rate"],
            "delta": primary["delta"],
            "mcnemar": primary["mcnemar"],
            "paired_bootstrap": primary["paired_bootstrap"],
        },
        "exploratory_multiplicity": {
            "method": "Holm step-down",
            "family_definition": (
                "all 24 main-arm x exact-outcome comparisons except the single "
                "registered GPT top-5 function-joint primary"
            ),
            "family_size": len(exploratory_raw),
            "raw_p": exploratory_raw,
            "adjusted_p": exploratory_adjusted,
        },
        "protocol_acceptance_gates": protocol_gates,
        "rank_bucket_transitions": transitions,
        "order_audit": {
            "cases": 100,
            "exact_pair_agreement_cases": agreement,
            "exact_pair_agreement_rate": agreement / 100,
            "metrics": order_metrics,
        },
        "metrics": metrics,
    }
    write_json_atomic(output, result)
    print(json.dumps({
        "status": "completed",
        "output": str(output),
        "primary_baseline": primary["baseline_rate"],
        "primary_agent": primary["arm_rate"],
        "primary_delta": primary["delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

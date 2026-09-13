#!/usr/bin/env python3
"""Replay a frozen round-3 router over completed round-4 resolver traces.

This is a deterministic, no-provider-call counterfactual.  A routed case uses
the already-recorded resolver prediction and accounting; an unrouted case keeps
retrieval top-1 with zero agent cost.  Because this composition was selected
after round-4 results were available, the output is explicitly exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent_metrics import (
    evaluate_records,
    exact_mcnemar,
    score_prediction,
)


FEATURE_NAMES = (
    "trigger_top1",
    "action_top1",
    "trigger_gap12",
    "action_gap12",
    "trigger_gap15",
    "action_gap15",
    "trigger_entropy5",
    "action_entropy5",
    "trigger_services5",
    "action_services5",
    "trigger_services10",
    "action_services10",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, value: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def entropy(values: Sequence[float]) -> float:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum((weight / total) * math.log(max(weight / total, 1e-12)) for weight in weights)


def feature_row(row: Mapping[str, Any]) -> list[float]:
    sides: dict[str, dict[str, float]] = {}
    for side in ("trigger", "action"):
        candidates = row[f"{side}_candidates"]
        if len(candidates) < 10:
            raise ValueError(f"{side} needs at least ten frozen candidates")
        values = [float(item["retrieval_score"]) for item in candidates]
        sides[side] = {
            "top1": values[0],
            "gap12": values[0] - values[1],
            "gap15": values[0] - values[4],
            "entropy5": entropy(values[:5]),
            "services5": float(len({item["channel"] for item in candidates[:5]})),
            "services10": float(len({item["channel"] for item in candidates[:10]})),
        }
    return [
        sides["trigger"]["top1"],
        sides["action"]["top1"],
        sides["trigger"]["gap12"],
        sides["action"]["gap12"],
        sides["trigger"]["gap15"],
        sides["action"]["gap15"],
        sides["trigger"]["entropy5"],
        sides["action"]["entropy5"],
        sides["trigger"]["services5"],
        sides["action"]["services5"],
        sides["trigger"]["services10"],
        sides["action"]["services10"],
    ]


def probability(features: Sequence[float], model: Mapping[str, Any]) -> float:
    if not all(len(model[key]) == len(features) for key in ("mean", "scale", "coef")):
        raise ValueError("router model width does not match feature vector")
    standardized = [
        (value - float(mean)) / float(scale) if float(scale) else 0.0
        for value, mean, scale in zip(features, model["mean"], model["scale"])
    ]
    logit = float(model["intercept"]) + sum(
        value * float(weight) for value, weight in zip(standardized, model["coef"])
    )
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


def routing_score(row: Mapping[str, Any], router: Mapping[str, Any]) -> float:
    features = feature_row(row)
    return probability(features, router["models"]["wrong"]) * probability(
        features, router["models"]["covered"]
    )


def rank_bucket(row: Mapping[str, Any]) -> str:
    trigger_rank = {
        item["url"]: int(item.get("retrieval_rank", index))
        for index, item in enumerate(row["trigger_candidates"], start=1)
    }
    action_rank = {
        item["url"]: int(item.get("retrieval_rank", index))
        for index, item in enumerate(row["action_candidates"], start=1)
    }
    joint_rank = min(
        (
            max(trigger_rank[pair["trigger_url"]], action_rank[pair["action_url"]])
            for pair in row["valid_pairs"]
            if pair["trigger_url"] in trigger_rank and pair["action_url"] in action_rank
        ),
        default=None,
    )
    if joint_rank == 1:
        return "rank1"
    if joint_rank is not None and joint_rank <= 5:
        return "rank2_5"
    if joint_rank is not None and joint_rank <= 10:
        return "rank6_10"
    return "outside10"


def zero_accounting() -> dict[str, Any]:
    return {
        "logical_calls": 0,
        "api_attempts": 0,
        "tool_calls": 0,
        "catalog_reads": 0,
        "complete": True,
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        },
    }


def derived_record(row: Mapping[str, Any], *, routed: bool) -> dict[str, Any]:
    keep = (
        "group_id",
        "query",
        "trigger_candidates",
        "action_candidates",
        "valid_pairs",
        "gold_service_pairs",
    )
    result = {key: row[key] for key in keep if key in row}
    result.update(
        {
            "baseline_pair": row["baseline_pair"],
            "final_pair": row["final_pair"] if routed else row["baseline_pair"],
            "routed": routed,
            "protocol_valid": True,
            "fallback_used": False,
            "retained_baseline": not routed,
            "policy": {
                "mode": "function_topk",
                "top_k": 5,
                "view": "plain",
                "router": "frozen_round3_recoverability",
            },
            "accounting": row["accounting"] if routed else zero_accounting(),
            "calls": row.get("calls", []) if routed else [],
        }
    )
    return result


def extract_router(source: Mapping[str, Any]) -> Mapping[str, Any]:
    value = source.get("router", source)
    if not isinstance(value, Mapping):
        raise ValueError("router source must be a router artifact or contain .router")
    required = {"feature_names", "models", "threshold", "dataset_id", "calibration_split"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"router artifact is missing {sorted(missing)}")
    if tuple(value["feature_names"]) != FEATURE_NAMES:
        raise ValueError("router feature contract differs from the frozen implementation")
    if value["calibration_split"] != "reranker_train":
        raise ValueError("router was not fit solely on reranker_train")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--arm-result", type=Path, required=True)
    parser.add_argument("--router-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.records)
    if not rows:
        raise ValueError("resolver record file is empty")
    if len({row["group_id"] for row in rows}) != len(rows):
        raise ValueError("resolver record file has duplicate group IDs")

    source = read_json(args.router_source)
    if not isinstance(source, Mapping):
        raise ValueError("router source root must be an object")
    router = extract_router(source)
    arm_result = read_json(args.arm_result)
    binding = arm_result.get("binding", {})
    if binding.get("dataset_id") != router["dataset_id"]:
        raise ValueError("round-4 records and frozen router have different dataset IDs")
    if binding.get("arm") != "gemma_top5_plain_replication":
        raise ValueError("this analysis requires the frozen Gemma top-5 arm")
    prior_arms = source.get("arms", [])
    prior_metrics = source.get("metrics", {})
    if (
        "routed_plain" not in prior_arms
        or not isinstance(prior_metrics, Mapping)
        or "routed_plain" not in prior_metrics
        or source.get("model") != binding.get("model")
        or source.get("model_digest") != binding.get("model_digest")
    ):
        raise ValueError(
            "router source does not prove the same router/Gemma plain-top5 policy predates round four"
        )

    threshold = float(router["threshold"])
    decisions: list[dict[str, Any]] = []
    derived: list[dict[str, Any]] = []
    buckets: Counter[str] = Counter()
    for row in rows:
        score = routing_score(row, router)
        routed = score >= threshold
        bucket = rank_bucket(row)
        if routed:
            buckets[bucket] += 1
        baseline_correct = score_prediction(row, row["baseline_pair"])["function_joint"]
        resolver_correct = score_prediction(row, row["final_pair"])["function_joint"]
        decisions.append(
            {
                "group_id": row["group_id"],
                "routing_score": score,
                "routed": routed,
                "rank_bucket": bucket,
                "baseline_function_joint": baseline_correct,
                "resolver_function_joint": resolver_correct,
                "transition": (
                    "recovery"
                    if routed and not baseline_correct and resolver_correct
                    else "regression"
                    if routed and baseline_correct and not resolver_correct
                    else "unchanged"
                ),
            }
        )
        derived.append(derived_record(row, routed=routed))

    metrics = evaluate_records(
        derived,
        baseline_arm="baseline",
        arm_names=("baseline", "agent"),
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    routed_count = sum(decision["routed"] for decision in decisions)
    transitions = Counter(decision["transition"] for decision in decisions)
    prior_baseline = prior_metrics["retrieval_top1"]["exact_hit_vector"]
    prior_selective = prior_metrics["routed_plain"]["exact_hit_vector"]
    if len(prior_baseline) != 300 or len(prior_selective) != 300:
        raise ValueError("pre-round4 routed_plain evidence is not the declared first 300 cases")
    prior_test = exact_mcnemar(prior_baseline, prior_selective)
    output = {
        "schema_version": "farm_round4_selective_router_analysis_v2",
        "status": "exploratory_unregistered_confirmation_replay",
        "scientific_interpretation": (
            "The same frozen-router plus one-call Gemma plain-top5 policy predates round four "
            "as the round-3 routed_plain arm and the current 300 cases are disjoint. However, "
            "the policy was omitted from the registered round-4 matrix and reconstructed from "
            "stateless always-on calls instead of being prospectively routed. Treat this as "
            "replicated exploratory development evidence, not as the registered round-4 result "
            "or a locked-test claim."
        ),
        "counterfactual_policy": (
            "Invoke the recorded stateless Gemma top-5 resolver iff the frozen round-3 "
            "recoverability score meets its frozen threshold; otherwise retain retrieval top-1."
        ),
        "binding": {
            "dataset_id": router["dataset_id"],
            "confirmation_rows": len(rows),
            "records_sha256": sha256_file(args.records),
            "arm_result_sha256": sha256_file(args.arm_result),
            "router_source_sha256": sha256_file(args.router_source),
            "router_canonical_sha256": canonical_sha256(router),
            "router_calibration_split": router["calibration_split"],
            "router_threshold": threshold,
            "router_budget_fraction": router.get("budget_fraction"),
            "resolver_arm": binding["arm"],
            "resolver_model": binding.get("model"),
            "resolver_model_digest": binding.get("model_digest"),
            "policy_predates_round4": True,
            "pre_round4_arm": "routed_plain",
            "pre_round4_selected_group_ids_sha256": source.get("selected_group_ids_sha256"),
        },
        "routing": {
            "routed_cases": routed_count,
            "routed_fraction": routed_count / len(rows),
            "unrouted_cases": len(rows) - routed_count,
            "routed_rank_buckets": {name: buckets[name] for name in ("rank1", "rank2_5", "rank6_10", "outside10")},
            "transitions": dict(sorted(transitions.items())),
        },
        "pre_round4_disjoint_slice_evidence": {
            "status": "preexisting_exploratory_secondary_arm",
            "cases": len(prior_baseline),
            "baseline_hits": sum(prior_baseline),
            "selective_hits": sum(prior_selective),
            "baseline_rate": sum(prior_baseline) / len(prior_baseline),
            "selective_rate": sum(prior_selective) / len(prior_selective),
            "delta": (sum(prior_selective) - sum(prior_baseline)) / len(prior_baseline),
            "routed_cases": prior_metrics["routed_plain"]["routed_cases"],
            "mcnemar": prior_test,
        },
        "two_slice_consistency": {
            "status": "directionally_consistent_not_pooled",
            "caveat": (
                "The two slices used different resolver harnesses/prompts, and the round-4 replay "
                "was not registered as a confirmatory comparison. No pooled effect or p-value is "
                "reported."
            ),
            "pre_round4_delta": (sum(prior_selective) - sum(prior_baseline)) / len(prior_baseline),
            "round4_replay_delta": metrics["comparisons"]["agent_vs_baseline"]["outcomes"]["function_joint"]["delta"],
        },
        "metrics": metrics,
        "route_decisions": decisions,
    }
    write_json_atomic(args.output, output, overwrite=args.overwrite)
    joint = metrics["comparisons"]["agent_vs_baseline"]["outcomes"]["function_joint"]
    print(
        json.dumps(
            {
                "status": output["status"],
                "routed_cases": routed_count,
                "baseline_rate": joint["baseline_rate"],
                "selective_rate": joint["arm_rate"],
                "delta": joint["delta"],
                "recoveries": joint["mcnemar"]["recoveries"],
                "regressions": joint["mcnemar"]["regressions"],
                "raw_p": joint["raw_p"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

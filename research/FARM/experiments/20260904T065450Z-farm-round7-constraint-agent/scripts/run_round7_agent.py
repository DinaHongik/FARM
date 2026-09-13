#!/usr/bin/env python3
"""Run one resumable Round 7 agent-discovery arm on the frozen 231-case screen."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from constraint_agent import Policy, resolve
from ollama_constraint_adapter import OllamaConstraintChooser


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
SCREEN_HASH = "087c4f5d87c357952a3a94c02abe359710d06be6927859b1644da092cbf87f14"
CONFIRMATION_HASH = "368a5004afffde548444c3837a01efb00b7185dc30b4025739e29300c8fa2995"


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_round5_runtime() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "20260902T042236Z-farm-round5-selective-verifier"
        / "scripts"
        / "run_round5_agent.py"
    )
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    return _load_module("farm_round5_runtime_for_round7", path)


def _load_router_runtime() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "20260904T043613Z-farm-round6-safe-agent"
        / "scripts"
        / "frozen_router.py"
    )
    return _load_module("farm_round6_router_for_round7", path)


BASE = _load_round5_runtime()
ROUTER = _load_router_runtime()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return BASE.sha256_file(path)


def verify_pinned_data_manifest(data_root: Path, expected_sha256: str) -> Path:
    """Bind corpus hydration to the preregistered Dataset-v2 manifest."""

    manifest_path = data_root.resolve() / "manifest.json"
    if sha256_file(manifest_path) != expected_sha256:
        raise ValueError("matrix-pinned source changed: data_manifest")
    return manifest_path


def ids_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        # Preserve the preregistered Round 6 serialization exactly.  Changing
        # insignificant JSON whitespace would change the split identity hash.
        json.dumps([row["group_id"] for row in rows]).encode()
    ).hexdigest()


def hash_order(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(str(row["group_id"]).encode()).hexdigest(),
            str(row["group_id"]),
        ),
    )


def select_discovery_rows(
    split_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select only the previously consumed Round 6 discovery screen.

    The reserved window is used only to exclude its family IDs and verify its
    preregistered ID hash. Its queries, endpoint labels, and valid pairs are not
    copied into the returned rows.
    """

    if len(split_rows) != 1145 or len(candidate_rows) != 1145:
        raise ValueError("Round 7 expects exactly 1,145 development rows")
    ordered = hash_order(split_rows)
    consumed_families = {str(row["semantic_family_id"]) for row in ordered[:600]}
    confirmation = [
        row for row in ordered[600:900]
        if str(row["semantic_family_id"]) not in consumed_families
    ]
    if len(confirmation) != 292 or ids_hash(confirmation) != CONFIRMATION_HASH:
        raise ValueError("reserved agent-policy confirmation identity changed")
    excluded_families = consumed_families | {
        str(row["semantic_family_id"]) for row in confirmation
    }
    screen_source = [
        row for row in ordered[900:1145]
        if str(row["semantic_family_id"]) not in excluded_families
    ]
    if len(screen_source) != 231 or ids_hash(screen_source) != SCREEN_HASH:
        raise ValueError("Round 7 discovery-screen identity changed")
    candidate_map = {str(row["group_id"]): dict(row) for row in candidate_rows}
    if len(candidate_map) != len(candidate_rows):
        raise ValueError("candidate artifact contains duplicate group IDs")
    selected: list[dict[str, Any]] = []
    for source in screen_source:
        row = candidate_map.get(str(source["group_id"]))
        if row is None:
            raise ValueError("discovery row missing from candidate artifact")
        row["semantic_family_id"] = str(source["semantic_family_id"])
        selected.append(row)
    return selected, {
        "source_rows": 1145,
        "raw_window": [900, 1145],
        "selected_rows": len(selected),
        "selected_families": len({row["semantic_family_id"] for row in selected}),
        "selected_group_ids_sha256": ids_hash(selected),
        "reserved_confirmation_rows": 292,
        "reserved_confirmation_group_ids_sha256": ids_hash(confirmation),
        "reserved_payload_sent_to_model": False,
    }


def _pair_with_services(
    pair: Mapping[str, Any],
    corpora: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, str]:
    trigger = pair.get("trigger_url")
    action = pair.get("action_url")
    if not isinstance(trigger, str) or trigger not in corpora["trigger"]:
        raise ValueError("unknown trigger URL in evaluation pair")
    if not isinstance(action, str) or action not in corpora["action"]:
        raise ValueError("unknown action URL in evaluation pair")
    return {
        "trigger_url": trigger,
        "action_url": action,
        "trigger_service": str(corpora["trigger"][trigger]["channel"]),
        "action_service": str(corpora["action"][action]["channel"]),
    }


def _score_pair(prediction: Mapping[str, str], gold: Sequence[Mapping[str, str]]) -> dict[str, bool]:
    function_pairs = {(item["trigger_url"], item["action_url"]) for item in gold}
    service_pairs = {(item["trigger_service"], item["action_service"]) for item in gold}
    function = (prediction["trigger_url"], prediction["action_url"])
    service = (prediction["trigger_service"], prediction["action_service"])
    return {
        "function_trigger": function[0] in {pair[0] for pair in function_pairs},
        "function_action": function[1] in {pair[1] for pair in function_pairs},
        "function_joint": function in function_pairs,
        "service_trigger": service[0] in {pair[0] for pair in service_pairs},
        "service_action": service[1] in {pair[1] for pair in service_pairs},
        "service_joint": service in service_pairs,
    }


def _rank_bucket(row: Mapping[str, Any]) -> tuple[str, int | None]:
    valid = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
    trigger_ranks = {
        item["url"]: index for index, item in enumerate(row["trigger_candidates"][:10], start=1)
    }
    action_ranks = {
        item["url"]: index for index, item in enumerate(row["action_candidates"][:10], start=1)
    }
    rank = min(
        (
            max(trigger_ranks[trigger], action_ranks[action])
            for trigger, action in valid
            if trigger in trigger_ranks and action in action_ranks
        ),
        default=None,
    )
    if rank == 1:
        return "rank1", rank
    if rank is not None and rank <= 5:
        return "rank2_5", rank
    if rank is not None and rank <= 10:
        return "rank6_10", rank
    return "outside10", rank


def unrouted_trace(case: Mapping[str, Any], routing_score: float) -> dict[str, Any]:
    pair = {
        "trigger_url": case["trigger_candidates"][0]["url"],
        "action_url": case["action_candidates"][0]["url"],
    }
    return {
        "schema_version": "farm_round7_unrouted_trace_v1",
        "case_id": case["group_id"],
        "policy": {"safe_fallback": "retrieval_top1_pair"},
        "baseline_pair": pair,
        "proposal_pair": None,
        "final_pair": pair,
        "retained_baseline": True,
        "stable_decision": "NOT_ROUTED",
        "fallback_reason": None,
        "repair_attempted": False,
        "validations": [],
        "compiled": False,
        "compilation_reason": "agent_not_routed",
        "calls": [],
        "tool_trace": [],
        "accounting": {
            "logical_model_calls": 0, "api_attempts": 0, "model_tool_calls": 0,
            "deterministic_tool_calls": 0,
            "tool_counts": {"search_catalog": 0, "read_endpoint_schema": 0, "validate_pair": 0},
            "catalog_reads": 0, "complete": True,
            "usage": {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "latency_seconds": 0.0,
            },
        },
        "routing_score": routing_score,
    }


def evaluation_record(
    row: Mapping[str, Any],
    trace: Mapping[str, Any],
    corpora: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    arm: str,
    routed: bool,
    routing_score: float,
) -> dict[str, Any]:
    gold = [_pair_with_services(item, corpora) for item in row["valid_pairs"]]
    baseline = _pair_with_services(trace["baseline_pair"], corpora)
    final = _pair_with_services(trace["final_pair"], corpora)
    proposal_raw = trace.get("proposal_pair")
    proposal = (
        _pair_with_services(proposal_raw, corpora)
        if isinstance(proposal_raw, Mapping)
        else None
    )
    triggers = list(row["trigger_candidates"][:10])
    actions = list(row["action_candidates"][:10])
    trigger_ids = {item["url"] for item in triggers}
    action_ids = {item["url"] for item in actions}
    function_pairs = {(item["trigger_url"], item["action_url"]) for item in gold}
    service_pairs = {(item["trigger_service"], item["action_service"]) for item in gold}
    trigger_services = {str(corpora["trigger"][item["url"]]["channel"]) for item in triggers}
    action_services = {str(corpora["action"][item["url"]]["channel"]) for item in actions}
    function_trigger_oracle = any(pair[0] in trigger_ids for pair in function_pairs)
    function_action_oracle = any(pair[1] in action_ids for pair in function_pairs)
    function_joint_oracle = any(t in trigger_ids and a in action_ids for t, a in function_pairs)
    service_trigger_oracle = any(pair[0] in trigger_services for pair in service_pairs)
    service_action_oracle = any(pair[1] in action_services for pair in service_pairs)
    service_joint_oracle = any(t in trigger_services and a in action_services for t, a in service_pairs)
    bucket, joint_rank = _rank_bucket(row)
    calls = list(trace.get("calls") or [])
    accounting = dict(trace.get("accounting") or {})
    protocol_valid = (
        all(
            isinstance(call, Mapping)
            and call.get("reported_ok") is True
            and bool(call.get("accounting_complete"))
            for call in calls
        )
        if routed
        else True
    )
    return {
        "schema_version": "farm_round7_constraint_record_v1",
        "group_id": row["group_id"],
        "semantic_family_id": row["semantic_family_id"],
        "query": row["query"],
        "arm_id": arm,
        "routed": routed,
        "routing_score": routing_score,
        "valid_pairs": gold,
        "baseline_pair": baseline,
        "proposal_pair": proposal,
        "final_pair": final,
        "baseline_score": _score_pair(baseline, gold),
        "proposal_score": _score_pair(proposal, gold) if proposal is not None else None,
        "final_score": _score_pair(final, gold),
        "candidate_oracle": {
            "function_trigger": function_trigger_oracle,
            "function_action": function_action_oracle,
            "function_joint": function_joint_oracle,
            "service_trigger": service_trigger_oracle,
            "service_action": service_action_oracle,
            "service_joint": service_joint_oracle,
        },
        "rank_bucket": bucket,
        "gold_joint_rank": joint_rank,
        "trigger_candidates": [
            dict(item) | {"service": str(corpora["trigger"][item["url"]]["channel"])}
            for item in triggers
        ],
        "action_candidates": [
            dict(item) | {"service": str(corpora["action"][item["url"]]["channel"])}
            for item in actions
        ],
        "stable_decision": trace.get("stable_decision"),
        "fallback_reason": trace.get("fallback_reason"),
        "repair_attempted": bool(trace.get("repair_attempted")),
        "validations": list(trace.get("validations") or []),
        "compiled": bool(trace.get("compiled")),
        "compilation_reason": trace.get("compilation_reason"),
        "side_edit": {
            "trigger": final["trigger_url"] != baseline["trigger_url"],
            "action": final["action_url"] != baseline["action_url"],
        },
        "calls": calls,
        "tool_trace": list(trace.get("tool_trace") or []),
        "accounting": accounting,
        "protocol_valid": protocol_valid,
        "fallback_used": trace.get("fallback_reason") is not None,
        "retained_baseline": final == baseline,
    }


def exact_mcnemar(before: Sequence[int], after: Sequence[int]) -> dict[str, Any]:
    recovered = sum(x == 0 and y == 1 for x, y in zip(before, after))
    regressed = sum(x == 1 and y == 0 for x, y in zip(before, after))
    discordant = recovered + regressed
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(recovered, regressed) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "recoveries": recovered,
        "regressions": regressed,
        "discordant": discordant,
        "net": recovered - regressed,
        "exact_two_sided_p": p_value,
    }


def family_bootstrap(
    records: Sequence[Mapping[str, Any]], *, iterations: int = 10000, seed: int = 20260904
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for row in records:
        delta = int(row["final_score"]["function_joint"]) - int(row["baseline_score"]["function_joint"])
        clusters[str(row["semantic_family_id"])].append(delta)
    family_ids = sorted(clusters)
    generator = random.Random(seed)
    values = []
    for _ in range(iterations):
        sampled = [family_ids[generator.randrange(len(family_ids))] for _ in family_ids]
        deltas = [value for family in sampled for value in clusters[family]]
        values.append(sum(deltas) / len(deltas))
    values.sort()
    observed = sum(sum(values_) for values_ in clusters.values()) / len(records)
    return {
        "delta": observed,
        "ci95": [values[int(0.025 * (iterations - 1))], values[int(0.975 * (iterations - 1))]],
        "iterations": iterations,
        "families": len(family_ids),
    }


def conditional_oracle_metrics(
    records: Sequence[Mapping[str, Any]], outcomes: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate end-to-end oracle coverage from routed agent selection quality."""

    system: dict[str, Any] = {}
    routed_agent: dict[str, Any] = {}
    for outcome in outcomes:
        eligible = [row for row in records if row["candidate_oracle"][outcome]]
        system_hits = sum(bool(row["final_score"][outcome]) for row in eligible)
        system[outcome] = {
            "eligible": len(eligible),
            "final_hits": system_hits,
            "final_accuracy": system_hits / len(eligible) if eligible else None,
        }

        routed = [row for row in eligible if row["routed"]]
        proposals = [
            row
            for row in routed
            if isinstance(row.get("proposal_score"), Mapping)
        ]
        proposal_hits = sum(
            bool(row["proposal_score"][outcome]) for row in proposals
        )
        final_hits = sum(bool(row["final_score"][outcome]) for row in routed)
        routed_agent[outcome] = {
            "eligible_routed": len(routed),
            "protocol_valid": sum(bool(row["protocol_valid"]) for row in routed),
            "proposal_available": len(proposals),
            "proposal_hits": proposal_hits,
            "proposal_accuracy_when_available": (
                proposal_hits / len(proposals) if proposals else None
            ),
            "final_hits": final_hits,
            "final_accuracy": final_hits / len(routed) if routed else None,
        }
    return system, routed_agent


def accepted_override_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report comparable discordant precision plus outcomes over every change."""

    changed = [row for row in records if not row["retained_baseline"]]
    recoveries = sum(
        not row["baseline_score"]["function_joint"]
        and row["final_score"]["function_joint"]
        for row in records
    )
    regressions = sum(
        row["baseline_score"]["function_joint"]
        and not row["final_score"]["function_joint"]
        for row in records
    )
    discordant = recoveries + regressions
    changed_final_correct = sum(
        bool(row["final_score"]["function_joint"]) for row in changed
    )
    return {
        "changed_cases": len(changed),
        "accepted_override_precision": (
            recoveries / discordant if discordant else None
        ),
        "accepted_override_precision_denominator": "recoveries_plus_regressions",
        "changed_final_correct": changed_final_correct,
        "changed_final_correct_rate": (
            changed_final_correct / len(changed) if changed else None
        ),
        "neutral_changed_cases": len(changed) - discordant,
    }


def protocol_by_baseline_service(
    records: Sequence[Mapping[str, Any]], side: str
) -> dict[str, Any]:
    """Attribute whole-screen protocol cost to the retrieval baseline service."""

    if side not in {"trigger", "action"}:
        raise ValueError("protocol service side must be trigger or action")
    service_key = f"{side}_service"
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        service = row["baseline_pair"][service_key]
        if not isinstance(service, str) or not service:
            raise ValueError(f"baseline pair lacks {service_key}")
        grouped[service].append(row)
    output: dict[str, Any] = {}
    for service in sorted(grouped):
        rows = grouped[service]
        calls = sum(int(row["accounting"]["logical_model_calls"]) for row in rows)
        tokens = sum(float(row["accounting"]["usage"]["total_tokens"]) for row in rows)
        routed = sum(bool(row["routed"]) for row in rows)
        output[service] = {
            "cases": len(rows),
            "routed_cases": routed,
            "logical_model_calls": calls,
            "logical_model_calls_per_case": calls / len(rows),
            "logical_model_calls_per_routed_case": calls / routed if routed else None,
            "total_tokens": tokens,
            "tokens_per_case": tokens / len(rows),
            "tokens_per_routed_case": tokens / routed if routed else None,
        }
    return output


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty record set")
    outcomes = (
        "function_trigger", "function_action", "function_joint",
        "service_trigger", "service_action", "service_joint",
    )
    metrics: dict[str, Any] = {"rows": len(records)}
    for outcome in outcomes:
        before = [int(row["baseline_score"][outcome]) for row in records]
        after = [int(row["final_score"][outcome]) for row in records]
        metrics[outcome] = {
            "baseline_hits": sum(before),
            "baseline_rate": sum(before) / len(records),
            "agent_hits": sum(after),
            "agent_rate": sum(after) / len(records),
            "absolute_delta": (sum(after) - sum(before)) / len(records),
            "comparison": exact_mcnemar(before, after),
        }
    primary = metrics["function_joint"]["comparison"]
    baseline_correct = metrics["function_joint"]["baseline_hits"]
    metrics["primary_comparison"] = {
        **primary,
        "family_cluster_paired_bootstrap": family_bootstrap(records),
        **accepted_override_metrics(records),
        "baseline_correct_retention": (
            1.0 - primary["regressions"] / baseline_correct if baseline_correct else None
        ),
    }
    metrics["candidate_oracle"] = {
        outcome: {
            "hits": sum(bool(row["candidate_oracle"][outcome]) for row in records),
            "rate": sum(bool(row["candidate_oracle"][outcome]) for row in records) / len(records),
        }
        for outcome in outcomes
    }
    (
        metrics["system_conditional_on_oracle"],
        metrics["agent_routed_conditional_on_oracle"],
    ) = conditional_oracle_metrics(records, outcomes)
    metrics["by_rank_bucket"] = {}
    for bucket in ("rank1", "rank2_5", "rank6_10", "outside10"):
        rows = [row for row in records if row["rank_bucket"] == bucket]
        metrics["by_rank_bucket"][bucket] = {
            "rows": len(rows),
            "baseline_joint_hits": sum(bool(row["baseline_score"]["function_joint"]) for row in rows),
            "agent_joint_hits": sum(bool(row["final_score"]["function_joint"]) for row in rows),
            "logical_model_calls": sum(int(row["accounting"]["logical_model_calls"]) for row in rows),
        }
    accounting_totals = {
        "logical_model_calls": sum(int(row["accounting"]["logical_model_calls"]) for row in records),
        "api_attempts": sum(int(row["accounting"]["api_attempts"]) for row in records),
        "model_tool_calls": sum(int(row["accounting"]["model_tool_calls"]) for row in records),
        "deterministic_tool_calls": sum(int(row["accounting"]["deterministic_tool_calls"]) for row in records),
        "catalog_reads": sum(int(row["accounting"]["catalog_reads"]) for row in records),
    }
    usage = {
        key: sum(float(row["accounting"]["usage"][key]) for row in records)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")
    }
    routed = sum(bool(row["routed"]) for row in records)
    metrics["protocol"] = {
        **accounting_totals,
        **usage,
        "routed_cases": routed,
        "valid_rows": sum(bool(row["protocol_valid"]) for row in records),
        "fallback_rows": sum(bool(row["fallback_used"]) for row in records),
        "logical_calls_per_case": accounting_totals["logical_model_calls"] / len(records),
        "logical_calls_per_routed_case": (
            accounting_totals["logical_model_calls"] / routed if routed else None
        ),
        "tokens_per_case": usage["total_tokens"] / len(records),
    }
    metrics["protocol_by_baseline_service"] = {
        side: protocol_by_baseline_service(records, side)
        for side in ("trigger", "action")
    }
    metrics["decisions"] = dict(Counter(str(row["stable_decision"]) for row in records))
    metrics["side_edits"] = {
        "trigger_only": sum(row["side_edit"]["trigger"] and not row["side_edit"]["action"] for row in records),
        "action_only": sum(row["side_edit"]["action"] and not row["side_edit"]["trigger"] for row in records),
        "both": sum(row["side_edit"]["trigger"] and row["side_edit"]["action"] for row in records),
        "neither": sum(not row["side_edit"]["trigger"] and not row["side_edit"]["action"] for row in records),
    }
    metrics["validation"] = {
        "repair_attempted": sum(bool(row["repair_attempted"]) for row in records),
        "compiled": 0,
        "compiled_claim_allowed": False,
        "reason": "Dataset-v2 has no reference field bindings",
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    matrix_path = run_root / "EXPERIMENT_MATRIX.json"
    matrix = read_json(matrix_path)
    if matrix.get("run_id") != run_root.name or matrix.get("dataset_id") != DATASET_ID:
        raise ValueError("run root is not bound to the Round 7 experiment matrix")
    arm = next((row for row in matrix["agent_discovery_arms"] if row["id"] == args.arm), None)
    if not isinstance(arm, Mapping):
        raise ValueError("unknown Round 7 agent arm")
    if int(arm.get("scope", 0)) != 231:
        raise ValueError("agent arm scope must be the frozen 231-case discovery screen")

    sources = matrix["sources"]
    pinned = {
        "dev_split": "dev_split_sha256",
        "candidate_dev": "candidate_dev_sha256",
        "candidate_manifest": "candidate_manifest_sha256",
        "router": "router_sha256",
    }
    paths = {name: Path(sources[name]).resolve() for name in pinned}
    for name, hash_name in pinned.items():
        if sha256_file(paths[name]) != sources[hash_name]:
            raise ValueError(f"matrix-pinned source changed: {name}")
    candidate_manifest = read_json(paths["candidate_manifest"])
    if (
        candidate_manifest.get("dataset_id") != DATASET_ID
        or candidate_manifest.get("split") != "dev"
        or candidate_manifest.get("output_sha256") != sources["candidate_dev_sha256"]
    ):
        raise ValueError("candidate manifest binding changed")

    data_root = Path(sources["data_root"]).resolve()
    verify_pinned_data_manifest(data_root, str(sources["data_manifest_sha256"]))
    split_rows = read_json(paths["dev_split"])
    candidate_rows = read_json(paths["candidate_dev"])
    selected, selection = select_discovery_rows(split_rows, candidate_rows)
    all_rows, corpora, _ = BASE._load_inputs(
        paths["candidate_dev"], paths["candidate_manifest"], data_root
    )
    if len(all_rows) != 1145:
        raise ValueError("candidate hydration source count changed")
    router = ROUTER.validate_router(
        read_json(paths["router"]), candidate_sha256=sources["router_candidate_sha256"]
    )
    threshold = float(matrix["routing"]["threshold"])
    if abs(float(router["threshold"]) - threshold) > 1e-15:
        raise ValueError("frozen router threshold changed")

    output_root = run_root / "smoke" if args.smoke else run_root
    if args.smoke:
        routed_rows = [row for row in selected if ROUTER.routing_score(row, router) >= threshold]
        if not routed_rows:
            raise RuntimeError("no routed discovery row exists for smoke")
        selected = routed_rows[:1]

    environment = BASE._load_environment(args.env_file.resolve())
    model = matrix["model"]
    key = os.environ.get(model["key_env"]) or environment.get(model["key_env"])
    host = os.environ.get(model["host_env"]) or environment.get(model["host_env"])
    if not key or not host:
        raise RuntimeError("Ollama host/key environment slot is unconfigured")
    BASE.verify_model_digest(host, key, model["name"], model["digest"])

    policy = Policy(
        candidate_depth=10,
        evidence_view=str(arm["evidence_view"]),
        verification=str(arm["verification"]),
        max_repairs=int(arm["max_repairs"]),
        presentation_seed=int(arm["presentation_seed"]),
    )
    records_path = output_root / "records" / f"{args.arm}.jsonl"
    attempts_path = output_root / "attempts" / f"{args.arm}.jsonl"
    progress_path = output_root / "manifests" / f"{args.arm}.progress.json"
    result_path = output_root / "results" / f"{args.arm}.json"
    if result_path.exists():
        print(json.dumps({"status": "already_complete", "result": str(result_path)}))
        return
    if records_path.exists() and not args.resume:
        raise RuntimeError("work record exists; pass --resume")

    binding = {
        "run_id": run_root.name,
        "mode": "smoke" if args.smoke else "family_purged_agent_discovery",
        "arm": args.arm,
        "dataset_id": DATASET_ID,
        "split": "dev",
        "selection": selection | {
            "executed_rows": len(selected),
            "executed_group_ids_sha256": ids_hash(selected),
        },
        "matrix_sha256": sha256_file(matrix_path),
        "candidate_sha256": sources["candidate_dev_sha256"],
        "router_sha256": sources["router_sha256"],
        "runtime_sha256": sha256_file(Path(__file__).resolve()),
        "core_sha256": sha256_file(run_root / "scripts/constraint_agent.py"),
        "adapter_sha256": sha256_file(run_root / "scripts/ollama_constraint_adapter.py"),
        "model": model["name"],
        "model_digest": model["digest"],
        "policy": dict(arm),
        "protocol": {
            "fresh_context_per_call": True,
            "temperature": 0,
            "seed": model["seed"],
            "reasoning_effort": model["reasoning_effort"],
            "max_attempts_per_logical_call": 2,
            "write_ahead_content_free_attempt_log": True,
            "gold_at_inference_boundary": False,
            "fallback": "retrieval_top1_pair",
            "program_compilation_claim": False,
        },
    }
    completed = BASE.load_jsonl(records_path) if args.resume else []
    if args.resume and progress_path.exists() and read_json(progress_path).get("binding") != binding:
        raise RuntimeError("resume binding changed")
    done = {row["group_id"] for row in completed}
    if not done <= {row["group_id"] for row in selected}:
        raise RuntimeError("resume rows escaped the selected screen")
    BASE.write_json_atomic(progress_path, {
        "phase": "running", "binding": binding,
        "completed_rows": len(completed), "target_rows": len(selected),
    }, secret=key)

    chooser = OllamaConstraintChooser(
        base_url=host,
        api_key=key,
        model=model["name"],
        journal_path=attempts_path,
        timeout=240,
        max_tokens=512,
        reasoning_effort=model["reasoning_effort"],
        retry_delay=1.0,
        seed=int(model["seed"]),
    )
    try:
        for row in selected:
            if row["group_id"] in done:
                continue
            case = BASE.inference_case(row, corpora)
            score = float(ROUTER.routing_score(row, router))
            routed = score >= threshold
            trace = resolve(case, chooser, policy) if routed else unrouted_trace(case, score)
            record = evaluation_record(
                row, trace, corpora, arm=args.arm, routed=routed, routing_score=score
            )
            BASE.append_jsonl(records_path, record, secret=key)
            completed.append(record)
            done.add(row["group_id"])
            BASE.write_json_atomic(progress_path, {
                "phase": "running", "binding": binding,
                "completed_rows": len(completed), "target_rows": len(selected),
                "last_group_id": row["group_id"],
            }, secret=key)
            print(json.dumps({
                "arm": args.arm, "completed": len(completed), "target": len(selected),
                "routed": routed, "decision": trace["stable_decision"],
                "protocol_valid": record["protocol_valid"],
            }), flush=True)
    finally:
        chooser.close()

    order = {row["group_id"]: index for index, row in enumerate(selected)}
    completed.sort(key=lambda row: order[row["group_id"]])
    if len(completed) != len(selected) or ids_hash(completed) != ids_hash(selected):
        raise RuntimeError("completed record alignment changed")
    result = {
        "status": "completed",
        "binding": binding,
        "metrics": summarize_records(completed),
    }
    BASE.write_json_atomic(result_path, result, secret=key)
    BASE.write_json_atomic(progress_path, {
        "phase": "complete", "binding": binding,
        "completed_rows": len(completed), "target_rows": len(selected),
        "output": str(result_path), "output_sha256": sha256_file(result_path),
    }, secret=key)
    print(json.dumps({"status": "completed", "arm": args.arm, "rows": len(completed)}))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run one resumable, routed Round6 agent arm on the frozen discovery screen."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from frozen_router import routing_score, validate_router
from ollama_round6_adapter import OllamaRound6Chooser
from safe_pair_policy import SafePairPolicy, resolve, unrouted_trace


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
SCREEN_HASH = "087c4f5d87c357952a3a94c02abe359710d06be6927859b1644da092cbf87f14"
CONFIRMATION_HASH = "368a5004afffde548444c3837a01efb00b7185dc30b4025739e29300c8fa2995"


def _load_round5_runtime() -> Any:
    directory = (
        Path(__file__).resolve().parents[2]
        / "20260902T042236Z-farm-round5-selective-verifier"
        / "scripts"
    )
    sys.path.insert(0, str(directory))
    path = directory / "run_round5_agent.py"
    spec = importlib.util.spec_from_file_location("farm_round5_agent_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Round5 agent runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_round5_runtime()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return BASE.sha256_file(path)


def ids_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(json.dumps([row["group_id"] for row in rows]).encode()).hexdigest()


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
    if len(split_rows) != 1145 or len(candidate_rows) != 1145:
        raise ValueError("Round6 expects exactly 1,145 development rows")
    ordered = hash_order(split_rows)
    consumed_families = {row["semantic_family_id"] for row in ordered[:600]}
    confirmation = [
        row for row in ordered[600:900]
        if row["semantic_family_id"] not in consumed_families
    ]
    if len(confirmation) != 292 or ids_hash(confirmation) != CONFIRMATION_HASH:
        raise ValueError("reserved policy-confirmation identity changed")
    reserved_families = consumed_families | {row["semantic_family_id"] for row in confirmation}
    screen = [
        row for row in ordered[900:1145]
        if row["semantic_family_id"] not in reserved_families
    ]
    if len(screen) != 231 or ids_hash(screen) != SCREEN_HASH:
        raise ValueError("agent-discovery identity changed")
    candidate_map = {row["group_id"]: dict(row) for row in candidate_rows}
    if len(candidate_map) != len(candidate_rows):
        raise ValueError("candidate artifact has duplicate group IDs")
    selected = []
    for source in screen:
        candidate = candidate_map.get(source["group_id"])
        if candidate is None:
            raise ValueError("screen row missing from candidate artifact")
        candidate["semantic_family_id"] = source["semantic_family_id"]
        selected.append(candidate)
    return selected, {
        "source_rows": len(split_rows), "raw_window": [900, 1145],
        "excluded_family_sources": ["dev[0:600]", "family-purged dev[600:900]"],
        "selected_rows": len(selected), "selected_families": len({row["semantic_family_id"] for row in selected}),
        "selected_group_ids_sha256": ids_hash(selected),
        "reserved_confirmation_rows": len(confirmation),
        "reserved_confirmation_group_ids_sha256": ids_hash(confirmation),
    }


def exact_mcnemar(recoveries: int, regressions: int) -> float:
    discordant = recoveries + regressions
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(recoveries, regressions) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def bootstrap_delta(records: Sequence[Mapping[str, Any]], *, iterations: int = 10000, seed: int = 20260904) -> list[float]:
    differences = [
        int(row["final_score"]["function_joint"]) - int(row["baseline_score"]["function_joint"])
        for row in records
    ]
    generator = random.Random(seed)
    values = sorted(
        sum(differences[generator.randrange(len(differences))] for _ in differences) / len(differences)
        for _ in range(iterations)
    )
    return [values[int(0.025 * (iterations - 1))], values[int(0.975 * (iterations - 1))]]


def extra_metrics(records: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any]) -> dict[str, Any]:
    recovered = sum(
        not row["baseline_score"]["function_joint"] and row["final_score"]["function_joint"]
        for row in records
    )
    regressed = sum(
        row["baseline_score"]["function_joint"] and not row["final_score"]["function_joint"]
        for row in records
    )
    baseline_correct = sum(bool(row["baseline_score"]["function_joint"]) for row in records)
    accepted = recovered + regressed
    protocol_fallbacks = sum(bool(row.get("protocol_fallback")) for row in records)
    semantic_fallbacks = sum(bool(row.get("semantic_fallback")) for row in records)
    calls = int(metrics["protocol"]["logical_calls"])
    tokens = int(metrics["protocol"]["total_tokens"])
    return {
        "exact_mcnemar_two_sided_p": exact_mcnemar(recovered, regressed),
        "family_cluster_paired_bootstrap_95_ci_delta": bootstrap_delta(records),
        "recoveries": recovered, "regressions": regressed,
        "accepted_override_precision": recovered / accepted if accepted else None,
        "baseline_correct_retention": 1.0 - regressed / baseline_correct if baseline_correct else None,
        "routed_cases": sum(bool(row.get("routed")) for row in records),
        "protocol_fallback_cases": protocol_fallbacks,
        "semantic_fallback_cases": semantic_fallbacks,
        "logical_calls_per_case": calls / len(records),
        "tokens_per_case": tokens / len(records),
    }


def arm_ranking(
    spec: Mapping[str, Any], group_id: str,
    plain: Mapping[str, Sequence[str]], schema: Mapping[str, Sequence[str]],
) -> list[dict[str, str]]:
    if spec["ranking"] == "plain":
        if group_id not in plain:
            raise ValueError("plain ranking lacks screen group")
        return [BASE._pair_object(item) for item in plain[group_id]]
    if spec["ranking"] == "rrf_plain_schema":
        if group_id not in plain or group_id not in schema:
            raise ValueError("fused ranking lacks screen group")
        return BASE.rrf_pair_ranking(plain[group_id], schema[group_id])
    raise ValueError("unsupported arm ranking")


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
    if matrix.get("dataset_id") != DATASET_ID or matrix.get("run_id") != run_root.name:
        raise ValueError("experiment matrix binding changed")
    arm = next((item for item in matrix["agent_arms"] if item["id"] == args.arm), None)
    if not isinstance(arm, Mapping):
        raise ValueError("unknown agent arm")
    sources = matrix["sources"]
    pinned = {
        "dev_split": "dev_split_sha256", "candidate_dev": "candidate_dev_sha256",
        "candidate_manifest": "candidate_manifest_sha256", "pair_plain": "pair_plain_sha256",
        "pair_schema": "pair_schema_sha256", "router": "router_sha256",
    }
    paths = {key: Path(sources[key]).resolve() for key in pinned}
    for key, hash_key in pinned.items():
        if sha256_file(paths[key]) != sources[hash_key]:
            raise ValueError(f"matrix-pinned source hash changed: {hash_key}")
    candidate_manifest = read_json(paths["candidate_manifest"])
    if candidate_manifest.get("dataset_id") != DATASET_ID or candidate_manifest.get("split") != "dev":
        raise ValueError("candidate manifest is not Dataset-v2 dev")
    if candidate_manifest.get("output_sha256") != sources["candidate_dev_sha256"]:
        raise ValueError("candidate manifest output binding changed")
    split_rows = read_json(paths["dev_split"])
    candidate_rows = read_json(paths["candidate_dev"])
    selected, selection = select_discovery_rows(split_rows, candidate_rows)
    if arm.get("scope") != len(selected):
        raise ValueError("arm scope differs from frozen screen")
    output_root = run_root / "smoke" if args.smoke else run_root
    if args.smoke:
        # Pick the first routed case so the smoke proves a real provider call.
        pass
    all_rows, corpus_maps, _ = BASE._load_inputs(
        paths["candidate_dev"], paths["candidate_manifest"], Path(sources["data_root"])
    )
    if len(all_rows) != 1145:
        raise ValueError("hydration source row count changed")
    router = validate_router(read_json(paths["router"]), candidate_sha256=sources["router_candidate_sha256"])
    if abs(float(router["threshold"]) - float(matrix["router"]["threshold"])) > 1e-15:
        raise ValueError("router threshold changed")
    plain = BASE.load_pair_rankings(paths["pair_plain"])
    schema = BASE.load_pair_rankings(paths["pair_schema"])
    environment = BASE._load_environment(args.env_file.resolve())
    model_spec = matrix["model"]
    key = os.environ.get(model_spec["key_env"]) or environment.get(model_spec["key_env"])
    host = os.environ.get(model_spec["host_env"]) or environment.get(model_spec["host_env"])
    if not key or not host:
        raise RuntimeError("Ollama credential/host slot is unconfigured")
    BASE.verify_model_digest(host, key, model_spec["name"], model_spec["digest"])
    if args.smoke:
        routed = [row for row in selected if routing_score(row, router) >= float(router["threshold"])]
        if not routed:
            raise RuntimeError("screen has no routed smoke case")
        selected = routed[:1]
    policy = SafePairPolicy(
        card_count=int(arm["card_count"]), proposer_view=str(arm["proposer_view"]),
        verifier_view=str(arm["verifier_view"]), verification=str(arm["verification"]),
    )
    records_path = output_root / "records" / f"{args.arm}.jsonl"
    attempts_path = output_root / "attempts" / f"{args.arm}.jsonl"
    progress_path = output_root / "manifests" / f"{args.arm}.progress.json"
    result_path = output_root / "results" / f"{args.arm}.json"
    if result_path.exists():
        print(json.dumps({"status": "already_complete", "result": str(result_path)}))
        return
    if records_path.exists() and not args.resume:
        raise RuntimeError("work file exists; use --resume")
    binding = {
        "run_id": run_root.name, "mode": "smoke" if args.smoke else "family_purged_agent_discovery",
        "arm": args.arm, "dataset_id": DATASET_ID, "split": "dev",
        "selection": selection | {"executed_rows": len(selected), "executed_group_ids_sha256": ids_hash(selected)},
        "matrix_sha256": sha256_file(matrix_path),
        "candidate_sha256": sources["candidate_dev_sha256"],
        "router_sha256": sources["router_sha256"],
        "pair_plain_sha256": sources["pair_plain_sha256"],
        "pair_schema_sha256": sources["pair_schema_sha256"],
        "policy_sha256": sha256_file(run_root / "scripts/safe_pair_policy.py"),
        "adapter_sha256": sha256_file(run_root / "scripts/ollama_round6_adapter.py"),
        "router_runtime_sha256": sha256_file(run_root / "scripts/frozen_router.py"),
        "runtime_sha256": sha256_file(Path(__file__).resolve()),
        "round5_runtime_sha256": sha256_file(Path(BASE.__file__).resolve()),
        "model": model_spec["name"], "model_digest": model_spec["digest"],
        "policy": dict(arm),
        "protocol": {
            "fresh_context_per_call": True, "temperature": 0, "seed": 42,
            "reasoning_effort": "none", "max_attempts_per_call": 2,
            "write_ahead_attempt_log": True, "gold_in_inference": False,
            "fallback": "retrieval_top1",
        },
    }
    completed = BASE.load_jsonl(records_path) if args.resume else []
    if args.resume and progress_path.exists() and read_json(progress_path).get("binding") != binding:
        raise RuntimeError("resume binding changed")
    done = {row["group_id"] for row in completed}
    if not done <= {row["group_id"] for row in selected}:
        raise RuntimeError("records escaped the selected screen")
    BASE.write_json_atomic(progress_path, {
        "phase": "running", "binding": binding, "completed_rows": len(completed), "target_rows": len(selected),
    }, secret=key)
    chooser = OllamaRound6Chooser(
        base_url=host, api_key=key, model=model_spec["name"], journal_path=attempts_path,
        timeout=240, max_tokens=512, reasoning_effort="none", retry_delay=1.0, seed=42,
    )
    try:
        for row in selected:
            if row["group_id"] in done:
                continue
            case = BASE.inference_case(row, corpus_maps)
            score = routing_score(row, router)
            ranking = arm_ranking(arm, row["group_id"], plain, schema)
            BASE.validate_ranking_against_case(ranking, case)
            trace = (
                resolve(case, ranking, policy, chooser, routing_score=score)
                if score >= float(router["threshold"])
                else unrouted_trace(case, routing_score=score)
            )
            record = BASE.evaluation_record(
                row, trace, corpus_maps, arm=args.arm,
                pair_ranking=ranking if trace["routed"] else None,
            )
            record["semantic_family_id"] = row["semantic_family_id"]
            record["routed"] = trace["routed"]
            record["routing_score"] = score
            if not trace["routed"]:
                record["protocol_valid"] = True
                record["fallback_used"] = False
            record["protocol_fallback"] = trace["stable_decision"] in {"PROPOSER_FAILURE", "VERIFIER_FAILURE"}
            record["semantic_fallback"] = trace["stable_decision"] in {
                "PROPOSER_ABSTAIN", "VERIFIER_ABSTAIN", "VERIFIER_DISAGREE"
            }
            BASE.append_jsonl(records_path, record, secret=key)
            completed.append(record)
            done.add(row["group_id"])
            BASE.write_json_atomic(progress_path, {
                "phase": "running", "binding": binding, "completed_rows": len(completed),
                "target_rows": len(selected), "last_group_id": row["group_id"],
            }, secret=key)
            print(json.dumps({
                "arm": args.arm, "completed": len(completed), "target": len(selected),
                "routed": trace["routed"], "decision": trace["stable_decision"],
                "protocol_valid": record["protocol_valid"],
            }), flush=True)
    finally:
        chooser.close()
    order = {row["group_id"]: index for index, row in enumerate(selected)}
    completed.sort(key=lambda row: order[row["group_id"]])
    if len(completed) != len(selected) or ids_hash(completed) != ids_hash(selected):
        raise RuntimeError("completed record alignment changed")
    metrics = BASE.summarize_records(completed)
    metrics["round6"] = extra_metrics(completed, metrics)
    result = {"status": "completed", "binding": binding, "metrics": metrics}
    BASE.write_json_atomic(result_path, result, secret=key)
    BASE.write_json_atomic(progress_path, {
        "phase": "complete", "binding": binding, "completed_rows": len(completed),
        "target_rows": len(selected), "output": str(result_path),
        "output_sha256": sha256_file(result_path),
    }, secret=key)
    print(json.dumps({"status": "completed", "arm": args.arm, "rows": len(completed)}))


if __name__ == "__main__":
    main()

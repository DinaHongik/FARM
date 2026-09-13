#!/usr/bin/env python3
"""Train one leakage-safe baseline-vs-challenger override comparator.

The classifier is deliberately asymmetric: USE_CHALLENGER is positive only
when it repairs an incorrect retrieval-top1 pair. Every tie, wrong-to-wrong
change, or potential regression is KEEP_TOP1. Training and model selection are
restricted to a group-hash split inside Dataset-v2 reranker_train.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from urllib.parse import urlparse
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
VIEWS = {"names", "plain", "schema", "hybrid"}
TASKS = {"function", "service"}
POLICIES = {"retrieval_score", "endpoint_hard"}


def _load_round5_base() -> Any:
    path = (
        Path(__file__).resolve().parents[2]
        / "20260902T042236Z-farm-round5-selective-verifier"
        / "scripts"
        / "train_round5.py"
    )
    if not path.is_file():
        raise RuntimeError(f"pinned Round5 training dependency is missing: {path}")
    spec = importlib.util.spec_from_file_location("farm_round5_train_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned Round5 training dependency")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_round5_base()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return BASE.sha256_file(path)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_id", "experiment_id", "gpu", "dataset_id", "base_model",
        "base_model_revision", "train_candidates", "train_manifest",
        "train_candidates_sha256", "task", "evidence_view", "candidate_policy",
        "candidate_depth", "holdout_modulus", "holdout_fold",
        "max_positive_per_row", "max_negative_per_row", "class_weights",
        "override_threshold", "epochs", "batch_size",
        "gradient_accumulation_steps", "learning_rate", "warmup_ratio",
        "max_length", "prediction_batch_size", "seed", "fp16",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration lacks fields: {sorted(missing)}")
    value = dict(config)
    if value["dataset_id"] != DATASET_ID:
        raise ValueError("Dataset-v2 identity mismatch")
    if value["task"] not in TASKS or value["evidence_view"] not in VIEWS:
        raise ValueError("unsupported task or evidence view")
    if value["candidate_policy"] not in POLICIES:
        raise ValueError("unsupported candidate policy")
    if int(value["candidate_depth"]) != 10:
        raise ValueError("Round6 override lattice is frozen at top10")
    if bool(value["fp16"]):
        raise ValueError("V100 control uses FP32")
    if not isinstance(value["gpu"], int) or value["gpu"] < 0:
        raise ValueError("gpu must be a physical non-negative index")
    modulus, fold = int(value["holdout_modulus"]), int(value["holdout_fold"])
    if modulus < 2 or not 0 <= fold < modulus:
        raise ValueError("invalid group-hash holdout")
    if len(value["class_weights"]) != 2 or any(float(x) <= 0 for x in value["class_weights"]):
        raise ValueError("two positive class weights are required")
    if not 0.5 < float(value["override_threshold"]) < 1.0:
        raise ValueError("override threshold must be strictly between .5 and 1")
    for key in ("train_candidates", "train_manifest"):
        name = Path(str(value[key])).name.casefold()
        if "reranker_train" not in name or any(word in name for word in ("dev", "test", "validation")):
            raise ValueError(f"{key} must be the frozen reranker_train artifact")
    return value


def valid_pairs(row: Mapping[str, Any], task: str) -> set[tuple[str, str]]:
    function = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
    if task == "function":
        return function
    # Some valid functions are outside the top-ten candidate lattice.  The
    # frozen union candidates retain their service identity and are label-only;
    # they are never included in inference inputs.
    trigger_rows = list(row["trigger_candidates"]) + list(row.get("trigger_union_candidates") or [])
    action_rows = list(row["action_candidates"]) + list(row.get("action_union_candidates") or [])
    trigger_service = {item["url"]: str(item["channel"]) for item in trigger_rows}
    action_service = {item["url"]: str(item["channel"]) for item in action_rows}
    # The RRF artifact intentionally truncates some union tails. Dataset-v2
    # endpoint URLs have the canonical form ifttt.com/<service>/<side>/<fn>;
    # recover only that service slug, never function labels or relevance.
    def service_from_url(url: str) -> str:
        parsed = urlparse(url)
        pieces = [piece for piece in parsed.path.split("/") if piece]
        if parsed.hostname not in {"ifttt.com", "www.ifttt.com"} or len(pieces) < 3:
            raise ValueError("cannot derive canonical service slug from endpoint URL")
        return pieces[0]

    for trigger, action in function:
        trigger_service.setdefault(trigger, service_from_url(trigger))
        action_service.setdefault(action, service_from_url(action))
    return {(trigger_service[t], action_service[a]) for t, a in function}


def pair_key(pair: tuple[Mapping[str, Any], Mapping[str, Any]], task: str) -> tuple[str, str]:
    if task == "function":
        return str(pair[0]["url"]), str(pair[1]["url"])
    return str(pair[0]["channel"]), str(pair[1]["channel"])


def _rank(item: Mapping[str, Any], fallback: int) -> int:
    return int(item.get("retrieval_rank", fallback))


def challenger_pairs(row: Mapping[str, Any], config: Mapping[str, Any]) -> list[tuple[dict, dict]]:
    depth = int(config["candidate_depth"])
    triggers = list(row["trigger_candidates"][:depth])
    actions = list(row["action_candidates"][:depth])
    baseline = (str(triggers[0]["url"]), str(actions[0]["url"]))
    lattice = [
        (trigger, action)
        for trigger in triggers
        for action in actions
        if (str(trigger["url"]), str(action["url"])) != baseline
    ]

    def score_key(pair: tuple[dict, dict]) -> tuple[Any, ...]:
        trigger, action = pair
        tr = _rank(trigger, triggers.index(trigger) + 1)
        ar = _rank(action, actions.index(action) + 1)
        return (
            -(float(trigger["retrieval_score"]) + float(action["retrieval_score"])),
            max(tr, ar), tr + ar, tr, ar, str(trigger["url"]), str(action["url"]),
        )

    ranked = sorted(lattice, key=score_key)
    if config["candidate_policy"] == "retrieval_score":
        return ranked
    prioritized = []
    prioritized.extend((trigger, actions[0]) for trigger in triggers[1:])
    prioritized.extend((triggers[0], action) for action in actions[1:])
    prioritized.extend(
        pair for pair in ranked
        if pair[0]["channel"] == triggers[0]["channel"]
        or pair[1]["channel"] == actions[0]["channel"]
    )
    prioritized.extend(ranked)
    output, seen = [], set()
    for pair in prioritized:
        identity = (pair[0]["url"], pair[1]["url"])
        if identity not in seen and identity != baseline:
            output.append(pair)
            seen.add(identity)
    if len(output) != len(lattice):
        raise AssertionError("hard candidate ordering lost lattice pairs")
    return output


def candidate_text(item: Mapping[str, Any], view: str, side: str) -> str:
    service = str(item.get("service_name") or item.get("channel_display") or item["channel"])
    function = str(item.get("function_name") or item.get("name") or item["url"])
    if view == "names":
        evidence = f"service={service}"
    elif view == "plain":
        evidence = str(item["text_plain"])
    elif view == "schema":
        evidence = str(item["text_schema"])
    elif view == "hybrid":
        evidence = f"{item['text_plain']}\nSCHEMA: {item['text_schema']}"
    else:
        raise ValueError("unknown evidence view")
    return f"{side.upper()} service={service}; function={function}; evidence={evidence}"[:1800]


def comparison_document(
    baseline: tuple[Mapping[str, Any], Mapping[str, Any]],
    challenger: tuple[Mapping[str, Any], Mapping[str, Any]],
    view: str,
) -> str:
    return (
        "RETRIEVAL TOP1 (keep by default):\n"
        + candidate_text(baseline[0], view, "trigger")
        + "\n"
        + candidate_text(baseline[1], view, "action")
        + "\n\nCHALLENGER (use only if strictly better):\n"
        + candidate_text(challenger[0], view, "trigger")
        + "\n"
        + candidate_text(challenger[1], view, "action")
    )


def training_examples(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    columns: dict[str, list[Any]] = {"query": [], "context": [], "label": []}
    per_row: list[dict[str, int]] = []
    task, view = str(config["task"]), str(config["evidence_view"])
    max_positive = int(config["max_positive_per_row"])
    max_negative = int(config["max_negative_per_row"])
    for row in rows:
        baseline = (row["trigger_candidates"][0], row["action_candidates"][0])
        valid = valid_pairs(row, task)
        baseline_ok = pair_key(baseline, task) in valid
        positives, negatives = [], []
        for challenger in challenger_pairs(row, config):
            challenger_ok = pair_key(challenger, task) in valid
            # Positive means a strict recovery. All other comparisons train KEEP.
            (positives if challenger_ok and not baseline_ok else negatives).append(challenger)
        chosen_positive = positives[:max_positive]
        chosen_negative = negatives[:max_negative]
        for label, candidates in ((1, chosen_positive), (0, chosen_negative)):
            for challenger in candidates:
                columns["query"].append(
                    "Decide whether the challenger strictly improves the complete trigger-action pair for: "
                    + str(row["query"])
                )
                columns["context"].append(comparison_document(baseline, challenger, view))
                columns["label"].append(label)
        per_row.append({"positive": len(chosen_positive), "negative": len(chosen_negative)})
    counts = Counter(columns["label"])
    if not counts[0] or not counts[1]:
        raise ValueError("override training requires both KEEP and USE examples")
    return columns, {
        "source_rows": len(rows),
        "examples": len(columns["label"]),
        "label_counts": {"KEEP_TOP1": counts[0], "USE_CHALLENGER": counts[1]},
        "rows_with_recoverable_challenger": sum(item["positive"] > 0 for item in per_row),
        "target": "strict baseline recovery; ties and wrong-to-wrong changes are KEEP",
        "gold_injection_at_inference": False,
    }


def _softmax_use(logits: Any) -> list[float]:
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1 and values.size == 2:
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"expected N x 2 override logits, got {values.shape}")
    values -= values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    return [float(value) for value in probabilities[:, 1]]


def _score(pair: tuple[Mapping[str, Any], Mapping[str, Any]], row: Mapping[str, Any]) -> dict[str, bool]:
    function_valid = valid_pairs(row, "function")
    service_valid = valid_pairs(row, "service")
    function_pair = pair_key(pair, "function")
    service_pair = pair_key(pair, "service")
    return {
        "function_trigger": function_pair[0] in {item[0] for item in function_valid},
        "function_action": function_pair[1] in {item[1] for item in function_valid},
        "function_joint": function_pair in function_valid,
        "service_trigger": service_pair[0] in {item[0] for item in service_valid},
        "service_action": service_pair[1] in {item[1] for item in service_valid},
        "service_joint": service_pair in service_valid,
    }


def evaluate(model: Any, rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    outcomes = (
        "function_trigger", "function_action", "function_joint",
        "service_trigger", "service_action", "service_joint",
    )
    threshold = float(config["override_threshold"])
    details: list[dict[str, Any]] = []
    for row in rows:
        baseline = (row["trigger_candidates"][0], row["action_candidates"][0])
        challengers = challenger_pairs(row, config)
        logits = model.predict(
            [
                (
                    "Decide whether the challenger strictly improves the complete trigger-action pair for: "
                    + str(row["query"]),
                    comparison_document(baseline, challenger, str(config["evidence_view"])),
                )
                for challenger in challengers
            ],
            batch_size=int(config["prediction_batch_size"]),
            show_progress_bar=False,
        )
        probabilities = _softmax_use(logits)
        if not probabilities or not all(math.isfinite(value) for value in probabilities):
            raise RuntimeError("invalid override probabilities")
        best_index = max(range(len(challengers)), key=lambda i: (probabilities[i], -i))
        challenger = challengers[best_index]
        override = probabilities[best_index] >= threshold
        final = challenger if override else baseline
        baseline_score, final_score = _score(baseline, row), _score(final, row)
        task_key = f"{config['task']}_joint"
        oracle = any(pair_key(pair, str(config["task"])) in valid_pairs(row, str(config["task"])) for pair in challengers)
        details.append({
            "group_id": row["group_id"],
            "baseline_pair": {"trigger_url": baseline[0]["url"], "action_url": baseline[1]["url"]},
            "challenger_pair": {"trigger_url": challenger[0]["url"], "action_url": challenger[1]["url"]},
            "final_pair": {"trigger_url": final[0]["url"], "action_url": final[1]["url"]},
            "use_probability": probabilities[best_index],
            "override": override,
            "target_oracle_in_top10_lattice": oracle,
            "baseline_score": baseline_score,
            "final_score": final_score,
            "target_recovery": (not baseline_score[task_key]) and final_score[task_key],
            "target_regression": baseline_score[task_key] and (not final_score[task_key]),
        })
    metrics: dict[str, Any] = {"rows": len(details), "task": config["task"]}
    for outcome in outcomes:
        before = [int(item["baseline_score"][outcome]) for item in details]
        after = [int(item["final_score"][outcome]) for item in details]
        metrics[outcome] = {
            "baseline_hits": sum(before), "baseline_rate": sum(before) / len(before),
            "model_hits": sum(after), "model_rate": sum(after) / len(after),
            "comparison": BASE._binary_comparison(before, after),
        }
    target = f"{config['task']}_joint"
    target_before = [int(item["baseline_score"][target]) for item in details]
    target_after = [int(item["final_score"][target]) for item in details]
    comparison = BASE._binary_comparison(target_before, target_after)
    overrides = sum(item["override"] for item in details)
    baseline_correct = sum(target_before)
    metrics["safe_override"] = {
        **comparison,
        "threshold": threshold,
        "overrides": overrides,
        "coverage": overrides / len(details),
        "override_precision": comparison["recoveries"] / overrides if overrides else None,
        "baseline_correct_retention": 1.0 - comparison["regressions"] / baseline_correct if baseline_correct else None,
        "lattice_oracle_rows": sum(item["target_oracle_in_top10_lattice"] for item in details),
    }
    return metrics, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = validate_config(read_json(args.config.resolve()))
    run_root = args.run_root.resolve()
    if run_root.name != config["run_id"]:
        raise ValueError("run root and config run_id differ")
    rows, manifest = BASE.load_training_artifact(config)
    training_rows = [
        row for row in rows
        if not BASE.stable_holdout(row["group_id"], int(config["holdout_modulus"]), int(config["holdout_fold"]))
    ]
    holdout_rows = [row for row in rows if row not in training_rows]
    if not training_rows or not holdout_rows:
        raise ValueError("empty training or holdout partition")
    if args.smoke:
        training_rows, holdout_rows = training_rows[:32], holdout_rows[:8]
        config = dict(config)
        config["epochs"] = 0.05
    columns, assembly = training_examples(training_rows, config)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "assembly": assembly}, indent=2, sort_keys=True))
        return
    BASE.assert_single_v100(int(config["gpu"]))
    BASE._seed_everything(int(config["seed"]))
    output_root = run_root / ("smoke/checkpoints" if args.smoke else "checkpoints") / config["experiment_id"]
    result_path = run_root / ("smoke/results" if args.smoke else "results") / f"{config['experiment_id']}_train_holdout.json"
    progress_path = run_root / ("smoke/manifests" if args.smoke else "manifests") / f"{config['experiment_id']}.progress.json"
    dependency_path = Path(BASE.__file__).resolve()
    binding = {
        "run_id": config["run_id"], "experiment_id": config["experiment_id"],
        "dataset_id": DATASET_ID, "split": "reranker_train",
        "candidate_sha256": manifest["output_sha256"],
        "config_sha256": sha256_file(args.config.resolve()),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "round5_dependency_sha256": sha256_file(dependency_path),
        "holdout": {"modulus": config["holdout_modulus"], "fold": config["holdout_fold"]},
        "train_rows": len(training_rows), "holdout_rows": len(holdout_rows),
        "task": config["task"], "evidence_view": config["evidence_view"],
        "candidate_policy": config["candidate_policy"], "smoke": args.smoke,
    }
    binding_hash = hashlib.sha256(stable_json(binding).encode()).hexdigest()
    BASE.write_json_atomic(progress_path, {"phase": "training", "binding": binding, "assembly": assembly})
    model, seconds, reused = BASE.train_cross_encoder(
        config=config, columns=columns, output_root=output_root,
        binding_hash=binding_hash, num_labels=2,
        class_weights=[float(value) for value in config["class_weights"]],
    )
    metrics, details = evaluate(model, holdout_rows, config)
    result = {
        "status": "completed", "scientific_status": "reranker_train internal holdout only",
        "binding": binding, "assembly": assembly, "training_seconds": seconds,
        "checkpoint_reused": reused, "metrics": metrics, "rows_detail": details,
    }
    BASE.write_json_atomic(result_path, result)
    BASE.write_json_atomic(progress_path, {
        "phase": "complete", "binding": binding, "output": str(result_path),
        "output_sha256": sha256_file(result_path),
    })
    print(json.dumps({"status": "completed", "experiment": config["experiment_id"], "result": str(result_path)}))


if __name__ == "__main__":
    main()

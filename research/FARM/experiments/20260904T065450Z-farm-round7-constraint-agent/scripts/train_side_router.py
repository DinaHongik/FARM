#!/usr/bin/env python3
"""Train one fold of the leakage-safe four-class FARM side router.

The model sees only the request and the public plain/schema evidence attached to
the current retrieval top-1 trigger and action. Gold pairs are used solely to
construct training labels and post-inference held-out metrics. Development and
test artifacts are rejected at the configuration boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
EXPECTED_ROWS = 1145
CANDIDATE_SHA256 = "0e672749736ccecc79a14f917cd590f046c876764a969d2a1b203159f99ffbc3"
MANIFEST_SHA256 = "8236d7f373979539d9f070765de116dc52327e85ddc238aca4e0a4dcf5923d2e"
TRAIN_SPLIT_SHA256 = "8108ea1c757fb2a29c5c25b49d69a9c9b87839d2d2a7e83e002f557a5acf932a"
CLASS_NAMES = ("KEEP_PAIR", "CHANGE_TRIGGER", "CHANGE_ACTION", "CHANGE_BOTH")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
FORBIDDEN_SPLIT_TOKENS = {"dev", "development", "test", "tests", "validation"}
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
DETERMINISM_CONTRACT_VERSION = "side-router-determinism-v1"


def configure_pre_torch_environment(environment: dict[str, str]) -> dict[str, Any]:
    """Pin process settings that must exist before CUDA/PyTorch initializes.

    This helper deliberately has no ML dependencies so its fail-closed behavior
    can be unit tested on machines without CUDA, PyTorch, or Transformers.
    """
    existing = environment.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the pinned reproducibility "
            f"contract: expected {CUBLAS_WORKSPACE_CONFIG!r}, got {existing!r}"
        )
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return {
        "contract_version": DETERMINISM_CONTRACT_VERSION,
        "CUBLAS_WORKSPACE_CONFIG": environment["CUBLAS_WORKSPACE_CONFIG"],
    }


# This executes before any optional ML dependency is imported below. Merely
# setting CUBLAS_WORKSPACE_CONFIG after PyTorch has initialized CUDA is too late,
# so training later rejects a process in which torch was already imported.
_TORCH_WAS_ABSENT_DURING_ENVIRONMENT_CONFIGURATION = "torch" not in sys.modules
_PRE_TORCH_ENVIRONMENT = configure_pre_torch_environment(os.environ)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    with staging.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    with staging.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(stable_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def _data_path_has_forbidden_split(path_value: str) -> bool:
    """Reject split tokens while allowing the workspace container `experiments`."""
    for component in Path(path_value).parts:
        if component.casefold() == "experiments":
            continue
        tokens = {token for token in re.split(r"[^a-z0-9]+", component.casefold()) if token}
        if tokens & FORBIDDEN_SPLIT_TOKENS:
            return True
    return False


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "run_id", "experiment_id", "gpu", "dataset_id", "base_model",
        "base_model_revision", "train_candidates", "train_candidates_sha256",
        "train_manifest", "train_manifest_sha256", "train_split",
        "train_split_sha256", "expected_rows", "fold_modulus", "holdout_fold",
        "family_key", "class_names", "num_labels", "class_balance",
        "evidence_views", "chars_per_view", "route_threshold",
        "calibration_bins", "routing_thresholds", "epochs", "batch_size",
        "gradient_accumulation_steps", "learning_rate", "warmup_ratio",
        "max_length", "prediction_batch_size", "seed", "fp16",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration lacks fields: {sorted(missing)}")
    value = dict(config)
    if value["dataset_id"] != DATASET_ID:
        raise ValueError("Dataset-v2 identity mismatch")
    if int(value["expected_rows"]) != EXPECTED_ROWS:
        raise ValueError("side router requires all 1,145 reranker_train rows")
    if value["train_candidates_sha256"] != CANDIDATE_SHA256:
        raise ValueError("candidate artifact is not the pinned Round3 function RRF artifact")
    if value["train_manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("candidate manifest hash is not pinned")
    if value["train_split_sha256"] != TRAIN_SPLIT_SHA256:
        raise ValueError("reranker_train source split hash is not pinned")
    if int(value["fold_modulus"]) != 4:
        raise ValueError("side router requires exactly four family folds")
    fold = int(value["holdout_fold"])
    if not 0 <= fold < 4:
        raise ValueError("holdout_fold must be in [0,3]")
    if not isinstance(value["gpu"], int) or value["gpu"] != fold:
        raise ValueError("each fold must bind to the same-numbered physical GPU")
    if value["experiment_id"] != f"side_router_fold{fold}":
        raise ValueError("experiment_id must identify its declared fold")
    if value["family_key"] != "semantic_family_id":
        raise ValueError("folds must be keyed by semantic_family_id")
    if tuple(value["class_names"]) != CLASS_NAMES or int(value["num_labels"]) != 4:
        raise ValueError("the declared four-class order changed")
    if value["class_balance"] != "inverse_frequency_cross_entropy":
        raise ValueError("unsupported deterministic class balancing")
    if list(value["evidence_views"]) != ["plain", "schema"]:
        raise ValueError("inference must contain both public plain and schema evidence")
    if bool(value["fp16"]):
        raise ValueError("V100 side-router protocol is pinned to FP32")
    if int(value["chars_per_view"]) <= 0 or int(value["max_length"]) <= 0:
        raise ValueError("text limits must be positive")
    if int(value["calibration_bins"]) < 2:
        raise ValueError("at least two calibration bins are required")
    thresholds = [float(item) for item in value["routing_thresholds"]]
    if thresholds != sorted(set(thresholds)) or any(not 0.0 < item < 1.0 for item in thresholds):
        raise ValueError("routing thresholds must be unique, sorted, and inside (0,1)")
    if float(value["route_threshold"]) not in thresholds:
        raise ValueError("declared route threshold must be in routing_thresholds")
    for key in ("epochs", "batch_size", "gradient_accumulation_steps", "learning_rate"):
        if float(value[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("train_candidates", "train_manifest", "train_split"):
        path_value = str(value[key])
        if _data_path_has_forbidden_split(path_value):
            raise ValueError(f"development/test path is prohibited: {key}={path_value}")
        if "reranker_train" not in Path(path_value).name.casefold():
            raise ValueError(f"{key} must identify reranker_train")
    return value


def _unique_by_group_id(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        group_id = row.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"{source} row lacks group_id")
        if group_id in output:
            raise ValueError(f"duplicate {source} group_id: {group_id}")
        output[group_id] = row
    return output


def join_semantic_families(
    candidate_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(candidate_rows) != expected_rows or len(split_rows) != expected_rows:
        raise ValueError(f"family join requires exactly {expected_rows} rows on each side")
    candidates = _unique_by_group_id(candidate_rows, "candidate")
    split = _unique_by_group_id(split_rows, "source split")
    if set(candidates) != set(split):
        missing = sorted(set(candidates) - set(split))[:3]
        extra = sorted(set(split) - set(candidates))[:3]
        raise ValueError(f"candidate/split group IDs differ; missing={missing}, extra={extra}")
    joined: list[dict[str, Any]] = []
    fallback_count = 0
    for source_row in candidate_rows:
        group_id = str(source_row["group_id"])
        family = split[group_id].get("semantic_family_id")
        family_source = "semantic_family_id"
        if not isinstance(family, str) or not family.strip():
            # The fallback is explicit and audited; an absent split row is never
            # allowed to fall through to this branch.
            family = group_id
            family_source = "group_id_fallback"
            fallback_count += 1
        row = dict(source_row)
        row["_semantic_family_id"] = family.strip()
        row["_family_source"] = family_source
        joined.append(row)
    return joined, {
        "candidate_rows": len(candidate_rows),
        "source_split_rows": len(split_rows),
        "joined_rows": len(joined),
        "unique_group_ids": len(candidates),
        "unique_semantic_families": len({row["_semantic_family_id"] for row in joined}),
        "group_id_fallback_count": fallback_count,
    }


def family_fold(family_id: str, modulus: int = 4) -> int:
    if modulus != 4:
        raise ValueError("side router is pinned to four folds")
    value = int(hashlib.sha256(family_id.encode("utf-8")).hexdigest()[:16], 16)
    return value % modulus


def partition_family_fold(
    rows: Sequence[Mapping[str, Any]], holdout_fold: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    if not 0 <= holdout_fold < 4:
        raise ValueError("invalid holdout fold")
    training = [row for row in rows if family_fold(str(row["_semantic_family_id"])) != holdout_fold]
    heldout = [row for row in rows if family_fold(str(row["_semantic_family_id"])) == holdout_fold]
    train_families = {str(row["_semantic_family_id"]) for row in training}
    heldout_families = {str(row["_semantic_family_id"]) for row in heldout}
    overlap = train_families & heldout_families
    if not training or not heldout or overlap:
        raise ValueError(f"invalid family partition; overlap={sorted(overlap)[:3]}")
    assignments = [
        {"group_id": row["group_id"], "family_id": row["_semantic_family_id"],
         "fold": family_fold(str(row["_semantic_family_id"]))}
        for row in rows
    ]
    return training, heldout, {
        "holdout_fold": holdout_fold,
        "training_rows": len(training),
        "heldout_rows": len(heldout),
        "training_families": len(train_families),
        "heldout_families": len(heldout_families),
        "family_overlap": 0,
        "assignment_sha256": hashlib.sha256(stable_json(assignments).encode("utf-8")).hexdigest(),
    }


def _top1(row: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    candidates = row.get(f"{side}_candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) < 10:
        raise ValueError(f"{side} requires a frozen top-10 candidate list")
    item = candidates[0]
    if not isinstance(item, Mapping) or int(item.get("retrieval_rank", -1)) != 1:
        raise ValueError(f"{side} first candidate is not retrieval rank one")
    if not isinstance(item.get("url"), str) or not item["url"]:
        raise ValueError(f"{side} top1 lacks URL identity")
    return item


def side_label(row: Mapping[str, Any]) -> tuple[int, dict[str, bool]]:
    trigger = str(_top1(row, "trigger")["url"])
    action = str(_top1(row, "action")["url"])
    raw_valid = row.get("valid_pairs")
    if not isinstance(raw_valid, Sequence) or isinstance(raw_valid, (str, bytes)) or not raw_valid:
        raise ValueError("training label requires at least one observed valid pair")
    valid = {(str(item["trigger_url"]), str(item["action_url"])) for item in raw_valid}
    pair_correct = (trigger, action) in valid
    trigger_correct = any(candidate_trigger == trigger for candidate_trigger, _ in valid)
    action_correct = any(candidate_action == action for _, candidate_action in valid)
    individually_correct_invalid = trigger_correct and action_correct and not pair_correct
    if pair_correct:
        label = "KEEP_PAIR"
    elif individually_correct_invalid:
        # Independent endpoint correctness is insufficient when the observed
        # pairing is invalid. This explicitly maps to the conservative class.
        label = "CHANGE_BOTH"
    elif trigger_correct:
        label = "CHANGE_ACTION"
    elif action_correct:
        label = "CHANGE_TRIGGER"
    else:
        label = "CHANGE_BOTH"
    return CLASS_TO_INDEX[label], {
        "pair_correct": pair_correct,
        "trigger_correct": trigger_correct,
        "action_correct": action_correct,
        "individually_correct_but_invalid_pair": individually_correct_invalid,
    }


def _public_endpoint_text(item: Mapping[str, Any], side: str, chars_per_view: int) -> str:
    plain, schema = item.get("text_plain"), item.get("text_schema")
    if not isinstance(plain, str) or not plain.strip():
        raise ValueError(f"{side} top1 lacks public plain evidence")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError(f"{side} top1 lacks public schema evidence")
    service = item.get("service_name") or item.get("channel_display") or item.get("channel")
    function = item.get("function_name") or item.get("name")
    if not isinstance(service, str) or not service.strip() or not isinstance(function, str) or not function.strip():
        raise ValueError(f"{side} top1 lacks public service/function names")
    return (
        f"CURRENT {side.upper()} TOP1\n"
        f"service={service.strip()}\nfunction={function.strip()}\n"
        f"PLAIN: {plain.strip()[:chars_per_view]}\n"
        f"SCHEMA: {schema.strip()[:chars_per_view]}"
    )


def inference_text(row: Mapping[str, Any], chars_per_view: int) -> tuple[str, str]:
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("row lacks query")
    instruction = (
        "Classify which side of the current trigger-action top1 pair needs correction: "
        "KEEP_PAIR, CHANGE_TRIGGER, CHANGE_ACTION, or CHANGE_BOTH. Request: "
        + query.strip()
    )
    context = (
        _public_endpoint_text(_top1(row, "trigger"), "trigger", chars_per_view)
        + "\n\n"
        + _public_endpoint_text(_top1(row, "action"), "action", chars_per_view)
    )
    return instruction, context


def balanced_class_weights(counts: Mapping[int, int]) -> list[float]:
    values = [int(counts.get(index, 0)) for index in range(len(CLASS_NAMES))]
    if any(value <= 0 for value in values):
        raise ValueError(f"every training fold needs all four labels, got {values}")
    total = sum(values)
    return [total / (len(values) * value) for value in values]


def assemble_training_examples(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    columns: dict[str, list[Any]] = {"query": [], "context": [], "label": []}
    rare_count = 0
    input_hashes: list[str] = []
    for row in rows:
        query, context = inference_text(row, int(config["chars_per_view"]))
        label, label_state = side_label(row)
        columns["query"].append(query)
        columns["context"].append(context)
        columns["label"].append(label)
        rare_count += int(label_state["individually_correct_but_invalid_pair"])
        input_hashes.append(hashlib.sha256(stable_json([query, context]).encode("utf-8")).hexdigest())
    counts = Counter(columns["label"])
    weights = balanced_class_weights(counts)
    return columns, {
        "rows": len(rows),
        "class_order": list(CLASS_NAMES),
        "class_counts": {CLASS_NAMES[index]: counts[index] for index in range(len(CLASS_NAMES))},
        "class_weights": {CLASS_NAMES[index]: weights[index] for index in range(len(CLASS_NAMES))},
        "class_balance": "deterministic inverse-frequency CrossEntropyLoss",
        "individually_correct_but_invalid_pair_count": rare_count,
        "inference_inputs_sha256": hashlib.sha256(stable_json(input_hashes).encode("utf-8")).hexdigest(),
        "inference_fields": ["query", "trigger_top1.public_plain", "trigger_top1.public_schema",
                             "action_top1.public_plain", "action_top1.public_schema"],
        "gold_in_inference": False,
    }


def _softmax_rows(logits: Any) -> list[list[float]]:
    values = logits.tolist() if hasattr(logits, "tolist") else logits
    if not isinstance(values, list) or (values and not isinstance(values[0], list)):
        if isinstance(values, list) and len(values) == len(CLASS_NAMES):
            values = [values]
        else:
            raise ValueError("expected N x 4 logits")
    output: list[list[float]] = []
    for row in values:
        if len(row) != len(CLASS_NAMES):
            raise ValueError("expected exactly four logits per row")
        numeric = [float(value) for value in row]
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("non-finite side-router logit")
        maximum = max(numeric)
        exponentials = [math.exp(value - maximum) for value in numeric]
        total = sum(exponentials)
        output.append([value / total for value in exponentials])
    return output


def _binary_metrics(gold: Sequence[bool], predicted: Sequence[bool]) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise ValueError("binary metric inputs must be non-empty and aligned")
    tp = sum(left and right for left, right in zip(gold, predicted))
    fp = sum((not left) and right for left, right in zip(gold, predicted))
    fn = sum(left and (not right) for left, right in zip(gold, predicted))
    tn = len(gold) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(gold),
        "predicted_positive_rate": (tp + fp) / len(gold),
        "gold_positive_rate": (tp + fn) / len(gold),
    }


def _ece(probabilities: Sequence[float], correct: Sequence[bool], bins: int) -> dict[str, Any]:
    if len(probabilities) != len(correct) or not probabilities:
        raise ValueError("calibration inputs must be non-empty and aligned")
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [position for position, value in enumerate(probabilities)
                    if value >= low and (value < high or (index == bins - 1 and value <= high))]
        confidence = sum(probabilities[position] for position in selected) / len(selected) if selected else None
        accuracy = sum(correct[position] for position in selected) / len(selected) if selected else None
        if selected:
            ece += len(selected) / len(probabilities) * abs(float(confidence) - float(accuracy))
        rows.append({"low": low, "high": high, "count": len(selected),
                     "mean_confidence": confidence, "accuracy": accuracy})
    return {"expected_calibration_error": ece, "bins": rows}


def four_class_metrics(
    gold: Sequence[int], predicted: Sequence[int], probabilities: Sequence[Sequence[float]], bins: int
) -> dict[str, Any]:
    if len(gold) != len(predicted) or len(gold) != len(probabilities) or not gold:
        raise ValueError("four-class metric inputs must be non-empty and aligned")
    size = len(CLASS_NAMES)
    confusion = [[0 for _ in range(size)] for _ in range(size)]
    for expected, actual in zip(gold, predicted):
        confusion[expected][actual] += 1
    per_class: dict[str, Any] = {}
    f1_values, recalls = [], []
    for index, name in enumerate(CLASS_NAMES):
        tp = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(confusion[row][index] for row in range(size))
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        recalls.append(recall)
        per_class[name] = {"support": support, "predicted": predicted_count,
                           "precision": precision, "recall": recall, "f1": f1}
    # Operational thresholding can deliberately select a non-argmax class, so
    # calibrate the probability assigned to the emitted class rather than the
    # unrelated maximum probability.
    confidences = [float(row[label]) for row, label in zip(probabilities, predicted)]
    correct = [left == right for left, right in zip(gold, predicted)]
    nll = -sum(math.log(max(float(row[label]), 1e-12)) for row, label in zip(probabilities, gold)) / len(gold)
    brier = sum(
        sum((float(row[index]) - float(index == label)) ** 2 for index in range(size))
        for row, label in zip(probabilities, gold)
    ) / len(gold)
    return {
        "class_order": list(CLASS_NAMES),
        "accuracy": sum(correct) / len(gold),
        "balanced_accuracy": sum(recalls) / size,
        "macro_f1": sum(f1_values) / size,
        "confusion_rows_gold_columns_predicted": confusion,
        "per_class": per_class,
        "calibration": {
            "negative_log_likelihood": nll,
            "multiclass_brier_score": brier,
            "maximum_probability": _ece(confidences, correct, bins),
        },
    }


def routing_metrics(
    gold: Sequence[int], probabilities: Sequence[Sequence[float]], threshold: float, bins: int
) -> dict[str, Any]:
    route_probability = [1.0 - float(row[CLASS_TO_INDEX["KEEP_PAIR"]]) for row in probabilities]
    gold_route = [label != CLASS_TO_INDEX["KEEP_PAIR"] for label in gold]
    predicted_route = [value >= threshold for value in route_probability]
    binary = _binary_metrics(gold_route, predicted_route)
    binary["threshold"] = threshold
    binary["calibration"] = {
        "binary_brier_score": sum((value - float(label)) ** 2 for value, label in zip(route_probability, gold_route)) / len(gold),
        "route_probability": _ece(route_probability, gold_route, bins),
    }
    return binary


def evaluate_predictions(
    rows: Sequence[Mapping[str, Any]], probabilities: Sequence[Sequence[float]], config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("heldout rows and probabilities must align")
    gold: list[int] = []
    argmax_predictions: list[int] = []
    operational_predictions: list[int] = []
    details: list[dict[str, Any]] = []
    threshold = float(config["route_threshold"])
    for row, probability_row in zip(rows, probabilities):
        if len(probability_row) != len(CLASS_NAMES) or abs(sum(probability_row) - 1.0) > 1e-6:
            raise ValueError("invalid probability simplex")
        label, state = side_label(row)
        argmax_class = max(range(len(CLASS_NAMES)), key=lambda index: (probability_row[index], -index))
        route_probability = 1.0 - float(probability_row[CLASS_TO_INDEX["KEEP_PAIR"]])
        route = route_probability >= threshold
        operational = (
            max(range(1, len(CLASS_NAMES)), key=lambda index: (probability_row[index], -index))
            if route else CLASS_TO_INDEX["KEEP_PAIR"]
        )
        query, context = inference_text(row, int(config["chars_per_view"]))
        gold.append(label)
        argmax_predictions.append(argmax_class)
        operational_predictions.append(operational)
        details.append({
            "group_id": row["group_id"],
            "semantic_family_id": row["_semantic_family_id"],
            "family_source": row["_family_source"],
            "heldout_fold": int(config["holdout_fold"]),
            "gold_class": CLASS_NAMES[label],
            "gold_class_source": "post_inference_evaluation_only",
            "individually_correct_but_invalid_pair": state["individually_correct_but_invalid_pair"],
            "argmax_class": CLASS_NAMES[argmax_class],
            "operational_class": CLASS_NAMES[operational],
            "route_probability": route_probability,
            "routed_at_declared_threshold": route,
            "probabilities": {CLASS_NAMES[index]: float(probability_row[index]) for index in range(len(CLASS_NAMES))},
            "inference_input_sha256": hashlib.sha256(stable_json([query, context]).encode("utf-8")).hexdigest(),
        })
    bins = int(config["calibration_bins"])
    operational_metrics = four_class_metrics(gold, operational_predictions, probabilities, bins)
    argmax_metrics = four_class_metrics(gold, argmax_predictions, probabilities, bins)
    route = routing_metrics(gold, probabilities, threshold, bins)
    route["threshold_sweep"] = {
        str(value): {key: metric for key, metric in routing_metrics(gold, probabilities, value, bins).items()
                     if key not in {"calibration"}}
        for value in config["routing_thresholds"]
    }
    trigger_gold = [label in {CLASS_TO_INDEX["CHANGE_TRIGGER"], CLASS_TO_INDEX["CHANGE_BOTH"]} for label in gold]
    action_gold = [label in {CLASS_TO_INDEX["CHANGE_ACTION"], CLASS_TO_INDEX["CHANGE_BOTH"]} for label in gold]
    trigger_pred = [label in {CLASS_TO_INDEX["CHANGE_TRIGGER"], CLASS_TO_INDEX["CHANGE_BOTH"]}
                    for label in operational_predictions]
    action_pred = [label in {CLASS_TO_INDEX["CHANGE_ACTION"], CLASS_TO_INDEX["CHANGE_BOTH"]}
                   for label in operational_predictions]
    rare = [index for index, row in enumerate(details) if row["individually_correct_but_invalid_pair"]]
    return {
        "rows": len(rows),
        "operational_four_class": operational_metrics,
        "raw_argmax_four_class": argmax_metrics,
        "routing": route,
        "side_routing": {
            "trigger": _binary_metrics(trigger_gold, trigger_pred),
            "action": _binary_metrics(action_gold, action_pred),
        },
        "rare_individually_correct_invalid_pair": {
            "count": len(rare),
            "mapped_gold_class": "CHANGE_BOTH",
            "predicted_change_both": sum(operational_predictions[index] == CLASS_TO_INDEX["CHANGE_BOTH"] for index in rare),
        },
    }, details


def _sdp_flag(
    cuda_backend: Any,
    *,
    setter_name: str,
    getter_name: str,
    requested: bool,
) -> dict[str, Any]:
    """Set and verify one scaled-dot-product attention backend flag."""
    setter = getattr(cuda_backend, setter_name, None)
    getter = getattr(cuda_backend, getter_name, None)
    if callable(setter):
        setter(requested)
    actual = bool(getter()) if callable(getter) else None
    if callable(setter) and callable(getter) and actual is not requested:
        raise RuntimeError(
            f"failed to set deterministic SDP contract: {getter_name}={actual}, "
            f"expected {requested}"
        )
    return {
        "setter_available": callable(setter),
        "getter_available": callable(getter),
        "requested_enabled": requested,
        "actual_enabled": actual,
    }


def _torch_determinism_snapshot(torch_module: Any) -> dict[str, Any]:
    cuda_backend = getattr(torch_module.backends, "cuda", None)

    def read_flag(name: str) -> bool | None:
        getter = getattr(cuda_backend, name, None) if cuda_backend is not None else None
        return bool(getter()) if callable(getter) else None

    warn_only_getter = getattr(
        torch_module, "is_deterministic_algorithms_warn_only_enabled", None
    )
    return {
        "contract_version": DETERMINISM_CONTRACT_VERSION,
        "configured_before_torch_import": _TORCH_WAS_ABSENT_DURING_ENVIRONMENT_CONFIGURATION,
        "environment": dict(_PRE_TORCH_ENVIRONMENT),
        "torch_version": str(getattr(torch_module, "__version__", "unknown")),
        "cuda_runtime_version": str(getattr(getattr(torch_module, "version", None), "cuda", None)),
        "strict_deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_algorithms_warn_only": (
            bool(warn_only_getter()) if callable(warn_only_getter) else None
        ),
        "cudnn": {
            "benchmark": bool(torch_module.backends.cudnn.benchmark),
            "deterministic": bool(torch_module.backends.cudnn.deterministic),
        },
        "scaled_dot_product_attention": {
            "flash_enabled": read_flag("flash_sdp_enabled"),
            "memory_efficient_enabled": read_flag("mem_efficient_sdp_enabled"),
            "cudnn_enabled": read_flag("cudnn_sdp_enabled"),
            "math_enabled": read_flag("math_sdp_enabled"),
            "required_policy": "math_only_when_backend_controls_are_available",
        },
        "transformer_training": {
            # The trainer's generic full_determinism helper is intentionally
            # disabled because some Transformers versions replace the pinned
            # :4096:8 workspace with :16:8. This module applies the stricter
            # contract explicitly before constructing the model/trainer.
            "determinism_provider": DETERMINISM_CONTRACT_VERSION,
            "transformers_full_determinism_flag": False,
            "dataloader_num_workers": 0,
            "fp16": False,
            "bf16": False,
        },
    }


def configure_torch_determinism(torch_module: Any, seed: int) -> dict[str, Any]:
    """Apply and audit strict single-GPU deterministic execution settings."""
    if not _TORCH_WAS_ABSENT_DURING_ENVIRONMENT_CONFIGURATION:
        raise RuntimeError(
            "torch was imported before CUBLAS_WORKSPACE_CONFIG was pinned; "
            "start side-router training in a fresh process"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG changed after module initialization")

    torch_module.manual_seed(seed)
    manual_seed_all = getattr(torch_module.cuda, "manual_seed_all", None)
    if callable(manual_seed_all):
        manual_seed_all(seed)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True

    cuda_backend = getattr(torch_module.backends, "cuda", None)
    sdp_settings: dict[str, dict[str, Any]] = {}
    if cuda_backend is not None:
        sdp_settings = {
            "flash": _sdp_flag(
                cuda_backend, setter_name="enable_flash_sdp",
                getter_name="flash_sdp_enabled", requested=False,
            ),
            "memory_efficient": _sdp_flag(
                cuda_backend, setter_name="enable_mem_efficient_sdp",
                getter_name="mem_efficient_sdp_enabled", requested=False,
            ),
            "cudnn": _sdp_flag(
                cuda_backend, setter_name="enable_cudnn_sdp",
                getter_name="cudnn_sdp_enabled", requested=False,
            ),
            "math": _sdp_flag(
                cuda_backend, setter_name="enable_math_sdp",
                getter_name="math_sdp_enabled", requested=True,
            ),
        }

    # warn_only must remain False: an unsupported nondeterministic operation is
    # an experiment failure, not a warning that silently weakens reproducibility.
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    snapshot = _torch_determinism_snapshot(torch_module)
    snapshot["seed"] = int(seed)
    snapshot["sdp_configuration_calls"] = sdp_settings
    if not snapshot["strict_deterministic_algorithms"]:
        raise RuntimeError("strict deterministic algorithms were not enabled")
    if snapshot["deterministic_algorithms_warn_only"] is True:
        raise RuntimeError("deterministic algorithm enforcement is still warn-only")
    attention = snapshot["scaled_dot_product_attention"]
    forbidden = ("flash_enabled", "memory_efficient_enabled", "cudnn_enabled")
    if any(attention[name] is True for name in forbidden):
        raise RuntimeError(f"nondeterministic CUDA SDP backend remains enabled: {attention}")
    if attention["math_enabled"] is False:
        raise RuntimeError("math SDP backend is disabled")
    return snapshot


def _seed_everything(seed: int) -> dict[str, Any]:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    return configure_torch_determinism(torch, seed)


def assert_single_v100(expected_physical_gpu: int) -> None:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible != [str(expected_physical_gpu)]:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={expected_physical_gpu}, got {visible or '<unset>'}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if torch.cuda.get_device_capability(0) != (7, 0):
        major, minor = torch.cuda.get_device_capability(0)
        raise RuntimeError(f"side router requires V100 sm_70, got sm_{major}{minor}")


def _train_cross_encoder(
    config: Mapping[str, Any], columns: Mapping[str, list[Any]], class_weights: Sequence[float],
    checkpoint_root: Path, binding_sha256: str,
) -> tuple[Any, float]:
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import CrossEntropyLoss

    if checkpoint_root.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {checkpoint_root}")
    trainer_root = checkpoint_root / "trainer"
    trainer_root.mkdir(parents=True, exist_ok=False)
    model = CrossEncoder(
        str(config["base_model"]), num_labels=4, max_length=int(config["max_length"]), device="cuda:0",
        model_kwargs={"ignore_mismatched_sizes": True},
    )
    arguments = CrossEncoderTrainingArguments(
        output_dir=str(trainer_root),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        fp16=False, bf16=False,
        full_determinism=False,
        seed=int(config["seed"]), data_seed=int(config["seed"]),
        save_strategy="epoch", save_total_limit=1, logging_steps=25,
        dataloader_num_workers=0, report_to=[],
    )
    weights = torch.tensor(list(class_weights), dtype=torch.float32, device=model.device)
    trainer = CrossEncoderTrainer(
        model=model, args=arguments, train_dataset=Dataset.from_dict(dict(columns)),
        loss=CrossEntropyLoss(model, weight=weights),
    )
    started = time.monotonic()
    trainer.train()
    training_seconds = time.monotonic() - started
    final_root = checkpoint_root / "final"
    staging = checkpoint_root / f".final.staging-{os.getpid()}"
    if staging.exists() or final_root.exists():
        raise FileExistsError("refusing to replace an existing final checkpoint")
    model.save_pretrained(str(staging))
    _write_json_atomic(staging / "side_router_checkpoint.json", {
        "binding_sha256": binding_sha256,
        "num_labels": 4,
        "class_order": list(CLASS_NAMES),
        "training_seconds": training_seconds,
    })
    os.replace(staging, final_root)
    return model, training_seconds


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    files = {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*")) if item.is_file()
    }
    return {
        "path": str(path),
        "files": files,
        "tree_sha256": hashlib.sha256(stable_json(files).encode("utf-8")).hexdigest(),
    }


def _stratified_smoke(rows: Sequence[Mapping[str, Any]], per_class: int) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = []
    counts: Counter[int] = Counter()
    for row in rows:
        label, _ = side_label(row)
        if counts[label] < per_class:
            output.append(row)
            counts[label] += 1
    if len(counts) != len(CLASS_NAMES):
        raise ValueError("smoke partition lacks one or more classes")
    return output


def load_training_sources(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = {
        "candidates": Path(str(config["train_candidates"])),
        "manifest": Path(str(config["train_manifest"])),
        "train_split": Path(str(config["train_split"])),
    }
    expected_hashes = {
        "candidates": str(config["train_candidates_sha256"]),
        "manifest": str(config["train_manifest_sha256"]),
        "train_split": str(config["train_split_sha256"]),
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"pinned source hash mismatch: actual={actual_hashes}")
    manifest = read_json(paths["manifest"])
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("split") != "reranker_train":
        raise ValueError("candidate manifest is not Dataset-v2 reranker_train")
    if manifest.get("output_sha256") != CANDIDATE_SHA256 or int(manifest.get("rows", -1)) != EXPECTED_ROWS:
        raise ValueError("candidate manifest binding changed")
    candidates, split = read_json(paths["candidates"]), read_json(paths["train_split"])
    if not isinstance(candidates, list) or not isinstance(split, list):
        raise ValueError("training sources must be JSON arrays")
    joined, join_stats = join_semantic_families(candidates, split, expected_rows=EXPECTED_ROWS)
    return joined, {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": actual_hashes,
        "family_join": join_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    config = validate_config(read_json(config_path))
    if run_root.name != config["run_id"]:
        raise ValueError("run root and config run_id differ")
    rows, source = load_training_sources(config)
    training_rows, heldout_rows, partition = partition_family_fold(rows, int(config["holdout_fold"]))
    if args.smoke:
        training_rows = _stratified_smoke(training_rows, 8)
        heldout_rows = _stratified_smoke(heldout_rows, 2)
        config = dict(config)
        config["epochs"] = 0.05
    columns, assembly = assemble_training_examples(training_rows, config)
    heldout_counts = Counter(side_label(row)[0] for row in heldout_rows)
    heldout_rare = sum(side_label(row)[1]["individually_correct_but_invalid_pair"] for row in heldout_rows)
    dry_summary = {
        "status": "dry_run_passed", "experiment_id": config["experiment_id"],
        "source": source, "partition": partition, "training_assembly": assembly,
        "executed_training_rows": len(training_rows), "executed_heldout_rows": len(heldout_rows),
        "heldout_class_counts": {CLASS_NAMES[index]: heldout_counts[index] for index in range(4)},
        "heldout_individually_correct_but_invalid_pair_count": heldout_rare,
    }
    if args.dry_run:
        print(json.dumps(dry_summary, indent=2, sort_keys=True))
        return

    suffix = "smoke" if args.smoke else "full"
    checkpoint_root = run_root / ("smoke/checkpoints" if args.smoke else "checkpoints") / str(config["experiment_id"])
    result_path = run_root / ("smoke/results" if args.smoke else "results") / f"{config['experiment_id']}.json"
    predictions_path = run_root / ("smoke/predictions" if args.smoke else "predictions") / f"{config['experiment_id']}_heldout.jsonl"
    progress_path = run_root / ("smoke/manifests" if args.smoke else "manifests") / f"{config['experiment_id']}.progress.json"
    for path in (checkpoint_root, result_path, predictions_path, progress_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing {suffix} artifact: {path}")

    assert_single_v100(int(config["gpu"]))
    determinism = _seed_everything(int(config["seed"]))
    binding = {
        "run_id": config["run_id"], "experiment_id": config["experiment_id"],
        "scientific_scope": "reranker_train internal family-heldout fold only",
        "dataset_id": DATASET_ID, "config": config,
        "config_sha256": sha256_file(config_path),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "source": source, "partition": partition,
        "gold_in_inference": False, "smoke": args.smoke,
        "determinism": determinism,
    }
    binding_sha256 = hashlib.sha256(stable_json(binding).encode("utf-8")).hexdigest()
    _write_json_atomic(progress_path, {
        "phase": "training", "binding": binding, "binding_sha256": binding_sha256,
        "training_assembly": assembly,
    })
    class_weights = [assembly["class_weights"][name] for name in CLASS_NAMES]
    model, training_seconds = _train_cross_encoder(
        config, columns, class_weights, checkpoint_root, binding_sha256,
    )
    pairs = [inference_text(row, int(config["chars_per_view"])) for row in heldout_rows]
    logits = model.predict(pairs, batch_size=int(config["prediction_batch_size"]), show_progress_bar=False)
    probabilities = _softmax_rows(logits)
    metrics, predictions = evaluate_predictions(heldout_rows, probabilities, config)
    _write_jsonl_atomic(predictions_path, predictions)
    checkpoint = _checkpoint_manifest(checkpoint_root / "final")
    result = {
        "status": "completed",
        "scientific_status": "reranker_train internal family-heldout fold only; no dev/test claim",
        "binding": binding, "binding_sha256": binding_sha256,
        "determinism": determinism,
        "training_assembly": assembly,
        "heldout_label_assembly": {
            "class_counts": {CLASS_NAMES[index]: heldout_counts[index] for index in range(4)},
            "individually_correct_but_invalid_pair_count": heldout_rare,
        },
        "training_seconds": training_seconds,
        "metrics": metrics,
        "predictions": {
            "path": str(predictions_path), "rows": len(predictions),
            "sha256": sha256_file(predictions_path),
        },
        "checkpoint": checkpoint,
    }
    _write_json_atomic(result_path, result)
    _write_json_atomic(progress_path, {
        "phase": "complete", "binding": binding, "binding_sha256": binding_sha256,
        "output": str(result_path), "output_sha256": sha256_file(result_path),
        "predictions": str(predictions_path), "predictions_sha256": sha256_file(predictions_path),
        "checkpoint_tree_sha256": checkpoint["tree_sha256"],
    }, replace=True)
    print(json.dumps({"status": "completed", "experiment": config["experiment_id"],
                      "result": str(result_path)}, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CLASS_NAMES", "CUBLAS_WORKSPACE_CONFIG", "DETERMINISM_CONTRACT_VERSION",
    "assemble_training_examples", "balanced_class_weights",
    "configure_pre_torch_environment", "configure_torch_determinism",
    "evaluate_predictions", "family_fold", "four_class_metrics", "inference_text",
    "join_semantic_families", "partition_family_fold", "routing_metrics", "side_label",
    "validate_config",
]

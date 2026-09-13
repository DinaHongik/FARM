#!/usr/bin/env python3
"""Train leakage-safe Round-5 ranking components on one declared V100.

The trainer deliberately accepts only the frozen Dataset-v2 ``reranker_train``
lattice.  Model selection metrics are computed on a deterministic group-hash
holdout inside that split; development and locked-test paths are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
TRAINER_KINDS = {"pair", "endpoint", "depth"}
PAIR_POLICIES = {"balanced_guard", "action_guard"}
BUCKET_NAMES = ("rank1", "rank2_5", "rank6_10", "outside10")


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


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def stable_holdout(group_id: str, modulus: int, fold: int) -> bool:
    if modulus < 2 or not 0 <= fold < modulus:
        raise ValueError("invalid group-hash holdout")
    value = int(hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:16], 16)
    return value % modulus == fold


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "run_id", "experiment_id", "gpu", "trainer_kind", "dataset_id",
        "base_model", "base_model_revision", "train_candidates", "train_manifest",
        "train_candidates_sha256", "candidate_depth", "holdout_modulus",
        "holdout_fold", "epochs", "batch_size", "gradient_accumulation_steps",
        "learning_rate", "warmup_ratio", "max_length", "prediction_batch_size",
        "seed", "fp16", "view",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration lacks fields: {sorted(missing)}")
    if config["trainer_kind"] not in TRAINER_KINDS:
        raise ValueError("unknown trainer_kind")
    if config["dataset_id"] != DATASET_ID:
        raise ValueError("Dataset-v2 identity mismatch in configuration")
    if config["view"] not in {"plain", "schema"}:
        raise ValueError("view must be plain or schema")
    if int(config["candidate_depth"]) != 10:
        raise ValueError("Round 5 is frozen to candidate_depth=10")
    if bool(config["fp16"]):
        raise ValueError("Round-5 V100 controls use FP32")
    if not isinstance(config["gpu"], int) or config["gpu"] < 0:
        raise ValueError("gpu must be a non-negative physical index")
    if not 0 <= int(config["holdout_fold"]) < int(config["holdout_modulus"]):
        raise ValueError("invalid holdout fold")

    # A training process must be incapable of receiving dev/test labels by typo.
    for key, value in config.items():
        if not isinstance(value, str):
            continue
        name = Path(value).name.casefold()
        if key.endswith(("candidates", "manifest")) and (
            "dev" in name or "test" in name or "validation" in name
        ):
            raise ValueError(f"development/test input is prohibited: {key}={value}")
    if "reranker_train" not in Path(config["train_candidates"]).name:
        raise ValueError("train_candidates must identify reranker_train")
    if "reranker_train" not in Path(config["train_manifest"]).name:
        raise ValueError("train_manifest must identify reranker_train")

    kind = config["trainer_kind"]
    if kind == "pair":
        extra = {
            "group_size", "retention_repeat", "negative_policy",
            "negative_quotas", "pair_chars_per_side", "lambda_k",
        }
        if extra - set(config):
            raise ValueError(f"pair config lacks: {sorted(extra - set(config))}")
        if config["negative_policy"] not in PAIR_POLICIES:
            raise ValueError("unknown pair negative policy")
        if int(config["group_size"]) < 4:
            raise ValueError("pair group_size is too small")
    elif kind == "endpoint":
        extra = {"group_size", "retention_repeat", "sides", "fusion_alpha", "lambda_k"}
        if extra - set(config):
            raise ValueError(f"endpoint config lacks: {sorted(extra - set(config))}")
        if config["sides"] != ["trigger", "action"]:
            raise ValueError("endpoint experiment must train both declared sides")
        if not 0.0 <= float(config["fusion_alpha"]) <= 1.0:
            raise ValueError("fusion_alpha outside [0,1]")
    else:
        extra = {"num_labels", "class_weights"}
        if extra - set(config):
            raise ValueError(f"depth config lacks: {sorted(extra - set(config))}")
        if int(config["num_labels"]) != len(BUCKET_NAMES):
            raise ValueError("depth router requires four rank buckets")
        if len(config["class_weights"]) != len(BUCKET_NAMES):
            raise ValueError("one class weight is required per bucket")
    return config


def assert_single_v100(expected_physical_gpu: int) -> None:
    visible = [part.strip() for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible != [str(expected_physical_gpu)]:
        raise RuntimeError(
            f"expected CUDA_VISIBLE_DEVICES={expected_physical_gpu}, got {visible or '<unset>'}"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    if torch.cuda.get_device_capability(0) != (7, 0):
        major, minor = torch.cuda.get_device_capability(0)
        raise RuntimeError(f"Round 5 is V100-only; visible device is sm_{major}{minor}")


def load_training_artifact(config: dict[str, Any]) -> tuple[list[dict], dict]:
    candidates_path = Path(config["train_candidates"])
    manifest_path = Path(config["train_manifest"])
    manifest = read_json(manifest_path)
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("split") != "reranker_train":
        raise ValueError("only Dataset-v2 reranker_train is accepted")
    actual = sha256_file(candidates_path)
    expected = str(config["train_candidates_sha256"])
    if actual != expected or manifest.get("output_sha256") != expected:
        raise RuntimeError("frozen reranker_train candidate hash mismatch")
    rows = read_json(candidates_path)
    if len(rows) != int(manifest.get("rows", -1)):
        raise RuntimeError("candidate row count does not match manifest")
    for row in rows:
        if len(row.get("trigger_candidates", [])) < 10 or len(row.get("action_candidates", [])) < 10:
            raise ValueError("every training row needs the frozen top-10 on both sides")
    return rows, manifest


def valid_pair_set(row: dict) -> set[tuple[str, str]]:
    return {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}


def pair_identifier(pair: tuple[dict, dict]) -> tuple[str, str]:
    return pair[0]["url"], pair[1]["url"]


def pair_sort_key(pair: tuple[dict, dict]) -> tuple[Any, ...]:
    trigger, action = pair
    trigger_rank = int(trigger["retrieval_rank"])
    action_rank = int(action["retrieval_rank"])
    return (
        trigger_rank + action_rank,
        max(trigger_rank, action_rank),
        trigger_rank,
        action_rank,
        trigger["url"],
        action["url"],
    )


def _append_unique(
    output: list[tuple[dict, dict]],
    seen: set[tuple[str, str]],
    candidates: Iterable[tuple[dict, dict]],
    limit: int,
) -> int:
    before = len(output)
    for pair in candidates:
        identity = pair_identifier(pair)
        if identity in seen:
            continue
        output.append(pair)
        seen.add(identity)
        if len(output) >= limit:
            break
    return len(output) - before


def select_pair_negatives(
    row: dict,
    *,
    policy: str,
    quotas: dict[str, int],
    limit: int,
    depth: int = 10,
) -> tuple[list[tuple[dict, dict]], dict[str, int]]:
    """Select deterministic, baseline-aware pair negatives without gold injection."""
    if policy not in PAIR_POLICIES:
        raise ValueError(f"unknown pair policy: {policy}")
    if limit <= 0:
        return [], {}
    triggers = row["trigger_candidates"][:depth]
    actions = row["action_candidates"][:depth]
    valid = valid_pair_set(row)
    gold_triggers = {trigger for trigger, _ in valid}
    gold_actions = {action for _, action in valid}
    trigger_gold_channels = {item["channel"] for item in triggers if item["url"] in gold_triggers}
    action_gold_channels = {item["channel"] for item in actions if item["url"] in gold_actions}
    all_negatives = sorted(
        (pair for pair in ((t, a) for t in triggers for a in actions) if pair_identifier(pair) not in valid),
        key=pair_sort_key,
    )
    categories = {
        "baseline": [
            (triggers[0], actions[0])
        ] if (triggers[0]["url"], actions[0]["url"]) not in valid else [],
        "action_confounder": [
            pair for pair in all_negatives
            if pair[0]["url"] in gold_triggers and pair[1]["url"] not in gold_actions
        ],
        "action_wrong": [pair for pair in all_negatives if pair[1]["url"] not in gold_actions],
        "trigger_confounder": [
            pair for pair in all_negatives
            if pair[0]["url"] not in gold_triggers and pair[1]["url"] in gold_actions
        ],
        "same_service_wrong_function": [
            pair for pair in all_negatives
            if (
                (pair[0]["channel"] in trigger_gold_channels and pair[0]["url"] not in gold_triggers)
                or (pair[1]["channel"] in action_gold_channels and pair[1]["url"] not in gold_actions)
            )
        ],
        "rank_sum": all_negatives,
    }
    order = (
        ("baseline", "action_confounder", "trigger_confounder", "same_service_wrong_function", "rank_sum")
        if policy == "balanced_guard"
        else (
            "baseline", "action_confounder", "action_wrong", "trigger_confounder",
            "same_service_wrong_function", "rank_sum",
        )
    )
    selected: list[tuple[dict, dict]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for category in order:
        category_limit = limit if category == "rank_sum" else min(limit, len(selected) + int(quotas.get(category, 0)))
        if category == "baseline":
            category_limit = min(limit, len(selected) + 1)
        added = _append_unique(selected, seen, categories[category], category_limit)
        counts[category] = added
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        _append_unique(selected, seen, all_negatives, limit)
    if any(pair_identifier(pair) in valid for pair in selected):
        raise AssertionError("negative selector admitted a valid pair")
    return selected, counts


def candidate_text(item: dict, view: str, side: str) -> str:
    value = item.get(f"text_{view}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate lacks {view} evidence: {item.get('url')}")
    return f"{side.upper()}: {value.strip()}"


def pair_document(trigger: dict, action: dict, view: str, per_side_chars: int) -> str:
    trigger_text = candidate_text(trigger, view, "trigger")[:per_side_chars]
    action_text = candidate_text(action, view, "action")[:per_side_chars]
    return f"{trigger_text}\n\n{action_text}"


def make_pair_training_columns(
    rows: Sequence[dict],
    config: dict[str, Any],
) -> tuple[dict[str, list], dict[str, Any]]:
    columns: dict[str, list] = {"query": [], "docs": [], "labels": []}
    excluded = 0
    repeats = 0
    category_totals: Counter[str] = Counter()
    depth = int(config["candidate_depth"])
    group_size = int(config["group_size"])
    for row in rows:
        valid = valid_pair_set(row)
        triggers = row["trigger_candidates"][:depth]
        actions = row["action_candidates"][:depth]
        positives = sorted(
            ((t, a) for t in triggers for a in actions if pair_identifier((t, a)) in valid),
            key=pair_sort_key,
        )
        if not positives:
            excluded += 1
            continue
        negative_limit = max(1, group_size - len(positives))
        negatives, counts = select_pair_negatives(
            row,
            policy=config["negative_policy"],
            quotas={key: int(value) for key, value in config["negative_quotas"].items()},
            limit=negative_limit,
            depth=depth,
        )
        selected = positives + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        if not negatives:
            raise ValueError("listwise pair group lacks a negative")
        baseline = (triggers[0]["url"], actions[0]["url"])
        repeat = int(config["retention_repeat"]) if baseline in valid else 1
        for _ in range(repeat):
            columns["query"].append(f"Select the exact trigger-to-action pair for: {row['query']}")
            columns["docs"].append([
                pair_document(t, a, config["view"], int(config["pair_chars_per_side"]))
                for t, a in selected
            ])
            columns["labels"].append(labels)
        repeats += repeat - 1
        category_totals.update(counts)
    return columns, {
        "source_rows": len(rows),
        "groups": len(columns["query"]),
        "unique_covered_groups": len(columns["query"]) - repeats,
        "retention_duplicates": repeats,
        "excluded_without_positive": excluded,
        "negative_categories": dict(sorted(category_totals.items())),
        "gold_injection": False,
    }


def select_endpoint_negatives(row: dict, side: str, limit: int, depth: int = 10) -> list[dict]:
    valid = valid_pair_set(row)
    gold = {pair[0 if side == "trigger" else 1] for pair in valid}
    candidates = row[f"{side}_candidates"][:depth]
    gold_channels = {item["channel"] for item in candidates if item["url"] in gold}
    baseline = [candidates[0]] if candidates[0]["url"] not in gold else []
    same_channel = [
        item for item in candidates
        if item["url"] not in gold and item["channel"] in gold_channels
    ]
    remaining = [item for item in candidates if item["url"] not in gold]
    order = baseline + (same_channel + same_channel if side == "action" else same_channel) + remaining
    output: list[dict] = []
    seen: set[str] = set()
    for item in order:
        if item["url"] in seen:
            continue
        output.append(item)
        seen.add(item["url"])
        if len(output) >= limit:
            break
    return output


def make_endpoint_training_columns(
    rows: Sequence[dict], side: str, config: dict[str, Any]
) -> tuple[dict[str, list], dict[str, Any]]:
    columns: dict[str, list] = {"query": [], "docs": [], "labels": []}
    excluded = 0
    repeats = 0
    same_channel_selected = 0
    depth = int(config["candidate_depth"])
    for row in rows:
        valid = valid_pair_set(row)
        gold = {pair[0 if side == "trigger" else 1] for pair in valid}
        candidates = row[f"{side}_candidates"][:depth]
        positives = [item for item in candidates if item["url"] in gold]
        if not positives:
            excluded += 1
            continue
        negatives = select_endpoint_negatives(
            row, side, max(1, int(config["group_size"]) - len(positives)), depth
        )
        if not negatives:
            raise ValueError("endpoint group lacks a negative")
        gold_channels = {item["channel"] for item in positives}
        same_channel_selected += sum(item["channel"] in gold_channels for item in negatives)
        selected = positives + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        repeat = int(config["retention_repeat"]) if candidates[0]["url"] in gold else 1
        for _ in range(repeat):
            columns["query"].append(f"Select the exact {side} for: {row['query']}")
            columns["docs"].append([candidate_text(item, config["view"], side) for item in selected])
            columns["labels"].append(labels)
        repeats += repeat - 1
    return columns, {
        "side": side,
        "source_rows": len(rows),
        "groups": len(columns["query"]),
        "unique_covered_groups": len(columns["query"]) - repeats,
        "retention_duplicates": repeats,
        "excluded_without_positive": excluded,
        "same_channel_negatives_selected": same_channel_selected,
        "gold_injection": False,
    }


def gold_pair_rank(row: dict, depth: int = 10) -> int | None:
    trigger_positions = {
        item["url"]: index for index, item in enumerate(row["trigger_candidates"][:depth], start=1)
    }
    action_positions = {
        item["url"]: index for index, item in enumerate(row["action_candidates"][:depth], start=1)
    }
    ranks = [
        max(trigger_positions[t], action_positions[a])
        for t, a in valid_pair_set(row)
        if t in trigger_positions and a in action_positions
    ]
    return min(ranks) if ranks else None


def depth_bucket(row: dict, depth: int = 10) -> int:
    rank = gold_pair_rank(row, depth)
    if rank == 1:
        return 0
    if rank is not None and rank <= 5:
        return 1
    if rank is not None and rank <= 10:
        return 2
    return 3


def _entropy(values: Sequence[float]) -> float:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum((weight / total) * math.log(max(weight / total, 1e-12)) for weight in weights)


def depth_context(row: dict, view: str) -> str:
    pieces = []
    for side in ("trigger", "action"):
        candidates = row[f"{side}_candidates"][:10]
        scores = [float(item["retrieval_score"]) for item in candidates]
        pieces.append(
            f"{side} top1 score={scores[0]:.8f}; gap12={scores[0]-scores[1]:.8f}; "
            f"gap15={scores[0]-scores[4]:.8f}; entropy5={_entropy(scores[:5]):.8f}; "
            f"services5={len({item['channel'] for item in candidates[:5]})}; "
            f"services10={len({item['channel'] for item in candidates})}.\n"
            f"{candidate_text(candidates[0], view, side)}"
        )
    return "\n\n".join(pieces)


def make_depth_training_columns(rows: Sequence[dict], config: dict[str, Any]) -> tuple[dict, dict]:
    labels = [depth_bucket(row, int(config["candidate_depth"])) for row in rows]
    return {
        "query": [row["query"] for row in rows],
        "context": [depth_context(row, config["view"]) for row in rows],
        "label": labels,
    }, {
        "source_rows": len(rows),
        "class_counts": {BUCKET_NAMES[key]: value for key, value in sorted(Counter(labels).items())},
        "target": "minimum max(trigger_rank,action_rank) over observed valid pairs",
        "gold_injection": False,
    }


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*") if output_dir.is_dir() else []:
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if (path / "trainer_state.json").is_file() and (path / "model.safetensors").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def _seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def train_cross_encoder(
    *,
    config: dict[str, Any],
    columns: dict[str, list],
    output_root: Path,
    binding_hash: str,
    num_labels: int = 1,
    class_weights: Sequence[float] | None = None,
) -> tuple[Any, float, bool]:
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import CrossEntropyLoss, LambdaLoss

    final_root = output_root / "final"
    metadata_path = final_root / "round5_checkpoint.json"
    if final_root.is_dir():
        if not metadata_path.is_file() or not (final_root / "model.safetensors").is_file():
            raise RuntimeError(f"incomplete final checkpoint: {final_root}")
        metadata = read_json(metadata_path)
        if metadata.get("binding_hash") != binding_hash:
            raise RuntimeError("existing checkpoint binding mismatch")
        model = CrossEncoder(str(final_root), num_labels=num_labels, max_length=int(config["max_length"]), device="cuda:0")
        return model, float(metadata["training_seconds"]), True

    model_kwargs = {"ignore_mismatched_sizes": True} if num_labels > 1 else None
    model = CrossEncoder(
        config["base_model"],
        num_labels=num_labels,
        max_length=int(config["max_length"]),
        device="cuda:0",
        model_kwargs=model_kwargs,
    )
    trainer_root = output_root / "trainer"
    trainer_root.mkdir(parents=True, exist_ok=True)
    arguments = CrossEncoderTrainingArguments(
        output_dir=str(trainer_root),
        num_train_epochs=float(config["epochs"]),
        per_device_train_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        learning_rate=float(config["learning_rate"]),
        warmup_ratio=float(config["warmup_ratio"]),
        fp16=False,
        bf16=False,
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=25,
        dataloader_num_workers=2,
        report_to=[],
    )
    if num_labels == 1:
        loss = LambdaLoss(model, k=int(config.get("lambda_k", 10)))
    else:
        weights = torch.tensor(list(class_weights or [1.0] * num_labels), dtype=torch.float32, device=model.device)
        loss = CrossEntropyLoss(model, weight=weights)
    trainer = CrossEncoderTrainer(
        model=model,
        args=arguments,
        train_dataset=Dataset.from_dict(columns),
        loss=loss,
    )
    resume = _latest_checkpoint(trainer_root)
    started = time.monotonic()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    training_seconds = time.monotonic() - started
    staging = final_root.parent / f".final.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    model.save_pretrained(str(staging))
    write_json_atomic(staging / "round5_checkpoint.json", {
        "binding_hash": binding_hash,
        "training_seconds": training_seconds,
        "num_labels": num_labels,
        "resumed_from": str(resume) if resume else None,
    })
    os.replace(staging, final_root)
    return model, training_seconds, False


def _binary_comparison(before: Sequence[int], after: Sequence[int]) -> dict[str, Any]:
    recoveries = sum(left == 0 and right == 1 for left, right in zip(before, after))
    regressions = sum(left == 1 and right == 0 for left, right in zip(before, after))
    discordant = recoveries + regressions
    if discordant:
        tail = sum(math.comb(discordant, k) for k in range(min(recoveries, regressions) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    else:
        p_value = 1.0
    return {
        "recoveries": recoveries,
        "regressions": regressions,
        "discordant": discordant,
        "exact_mcnemar_p": p_value,
        "delta": (sum(after) - sum(before)) / len(before),
    }


def evaluate_pair_model(model: Any, rows: Sequence[dict], config: dict[str, Any]) -> tuple[dict, list[dict]]:
    depth = int(config["candidate_depth"])
    hits = {1: 0, 5: 0, 10: 0}
    baseline_hits: list[int] = []
    model_hits: list[int] = []
    details = []
    for row in rows:
        valid = valid_pair_set(row)
        candidates = [
            (trigger, action)
            for trigger in row["trigger_candidates"][:depth]
            for action in row["action_candidates"][:depth]
        ]
        values = model.predict(
            [
                (
                    f"Select the exact trigger-to-action pair for: {row['query']}",
                    pair_document(t, a, config["view"], int(config["pair_chars_per_side"])),
                )
                for t, a in candidates
            ],
            batch_size=int(config["prediction_batch_size"]),
            show_progress_bar=False,
        )
        scores = [float(value) for value in values]
        if not all(math.isfinite(value) for value in scores):
            raise RuntimeError("non-finite pair score")
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], *pair_identifier(candidates[index])))
        ranking = [pair_identifier(candidates[index]) for index in order]
        relevant_rank = next((index for index, pair in enumerate(ranking, 1) if pair in valid), None)
        for cutoff in hits:
            hits[cutoff] += int(relevant_rank is not None and relevant_rank <= cutoff)
        baseline_pair = pair_identifier((row["trigger_candidates"][0], row["action_candidates"][0]))
        baseline_hits.append(int(baseline_pair in valid))
        model_hits.append(int(ranking[0] in valid))
        details.append({
            "group_id": row["group_id"],
            "gold_rank": relevant_rank,
            "top_pair": {"trigger_url": ranking[0][0], "action_url": ranking[0][1]},
            "top_score": scores[order[0]],
            "baseline_exact": bool(baseline_hits[-1]),
            "model_exact": bool(model_hits[-1]),
        })
    metrics = {f"pair_R@{cutoff}": hits[cutoff] / len(rows) for cutoff in hits}
    metrics["retrieval_top1"] = sum(baseline_hits) / len(rows)
    metrics["comparison"] = _binary_comparison(baseline_hits, model_hits)
    return metrics, details


def _minmax(values: Sequence[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def evaluate_endpoint_models(
    models: dict[str, Any], rows: Sequence[dict], config: dict[str, Any]
) -> tuple[dict, list[dict]]:
    before: list[int] = []
    after: list[int] = []
    trigger_hits = action_hits = 0
    details = []
    alpha = float(config["fusion_alpha"])
    for row in rows:
        valid = valid_pair_set(row)
        selected: dict[str, str] = {}
        ranks: dict[str, list[str]] = {}
        for side in ("trigger", "action"):
            candidates = row[f"{side}_candidates"][: int(config["candidate_depth"])]
            values = models[side].predict(
                [(f"Select the exact {side} for: {row['query']}", candidate_text(item, config["view"], side)) for item in candidates],
                batch_size=int(config["prediction_batch_size"]),
                show_progress_bar=False,
            )
            ce = _minmax([float(value) for value in values])
            retrieval = _minmax([float(item["retrieval_score"]) for item in candidates])
            fused = [(1.0 - alpha) * retrieval[i] + alpha * ce[i] for i in range(len(candidates))]
            order = sorted(range(len(candidates)), key=lambda i: (-fused[i], candidates[i]["url"]))
            ranks[side] = [candidates[i]["url"] for i in order]
            selected[side] = ranks[side][0]
        pair = (selected["trigger"], selected["action"])
        baseline = (row["trigger_candidates"][0]["url"], row["action_candidates"][0]["url"])
        before.append(int(baseline in valid))
        after.append(int(pair in valid))
        trigger_hits += int(any(pair[0] == gold_trigger for gold_trigger, _ in valid))
        action_hits += int(any(pair[1] == gold_action for _, gold_action in valid))
        details.append({
            "group_id": row["group_id"],
            "selected_pair": {"trigger_url": pair[0], "action_url": pair[1]},
            "baseline_exact": bool(before[-1]),
            "model_exact": bool(after[-1]),
            "trigger_ranking": ranks["trigger"],
            "action_ranking": ranks["action"],
        })
    return {
        "trigger_R@1": trigger_hits / len(rows),
        "action_R@1": action_hits / len(rows),
        "joint_independent_R@1": sum(after) / len(rows),
        "retrieval_top1": sum(before) / len(rows),
        "comparison": _binary_comparison(before, after),
        "fusion": "per-row minmax: (1-alpha)*RRF + alpha*CE",
        "fusion_alpha": alpha,
    }, details


def evaluate_depth_model(model: Any, rows: Sequence[dict], config: dict[str, Any]) -> tuple[dict, list[dict]]:
    import numpy as np

    gold = [depth_bucket(row, int(config["candidate_depth"])) for row in rows]
    logits = model.predict(
        [(row["query"], depth_context(row, config["view"])) for row in rows],
        batch_size=int(config["prediction_batch_size"]),
        show_progress_bar=False,
    )
    logits = np.asarray(logits, dtype=np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1).tolist()
    confusion = [[0 for _ in BUCKET_NAMES] for _ in BUCKET_NAMES]
    for expected, actual in zip(gold, predicted):
        confusion[expected][actual] += 1
    recalls, f1s = [], []
    for label in range(len(BUCKET_NAMES)):
        tp = confusion[label][label]
        fn = sum(confusion[label]) - tp
        fp = sum(confusion[row][label] for row in range(len(BUCKET_NAMES))) - tp
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    details = [
        {
            "group_id": row["group_id"],
            "gold_bucket": BUCKET_NAMES[expected],
            "predicted_bucket": BUCKET_NAMES[actual],
            "probabilities": {BUCKET_NAMES[i]: float(probabilities[index, i]) for i in range(len(BUCKET_NAMES))},
        }
        for index, (row, expected, actual) in enumerate(zip(rows, gold, predicted))
    ]
    return {
        "accuracy": sum(left == right for left, right in zip(gold, predicted)) / len(gold),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "majority_class_accuracy": max(Counter(gold).values()) / len(gold),
        "confusion_rows_gold_columns_predicted": confusion,
        "bucket_order": list(BUCKET_NAMES),
    }, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    config = validate_config(read_json(config_path))
    if run_root.name != config["run_id"]:
        raise ValueError("run-root basename does not match run_id")
    result_path = run_root / "results" / f"{config['experiment_id']}_train_holdout.json"
    progress_path = run_root / "manifests" / f"{config['experiment_id']}.progress.json"
    if result_path.exists():
        raise RuntimeError(f"refusing to overwrite final result: {result_path}")
    assert_single_v100(int(config["gpu"]))
    rows, manifest = load_training_artifact(config)
    training_rows = [
        row for row in rows
        if not stable_holdout(row["group_id"], int(config["holdout_modulus"]), int(config["holdout_fold"]))
    ]
    holdout_rows = [
        row for row in rows
        if stable_holdout(row["group_id"], int(config["holdout_modulus"]), int(config["holdout_fold"]))
    ]
    if not training_rows or not holdout_rows:
        raise RuntimeError("group-hash split produced an empty partition")
    binding = {
        "dataset_id": DATASET_ID,
        "split": "reranker_train",
        "candidate_sha256": manifest["output_sha256"],
        "config_sha256": sha256_file(config_path),
        "trainer_sha256": sha256_file(Path(__file__).resolve()),
        "train_rows": len(training_rows),
        "holdout_rows": len(holdout_rows),
        "holdout_policy": f"sha256(group_id) mod {config['holdout_modulus']} == {config['holdout_fold']}",
        "development_rows_opened": 0,
        "locked_test_rows_opened": 0,
    }
    binding_hash = hashlib.sha256(stable_json(binding).encode("utf-8")).hexdigest()
    _seed_everything(int(config["seed"]))

    if config["trainer_kind"] == "pair":
        columns, assembly = make_pair_training_columns(training_rows, config)
    elif config["trainer_kind"] == "endpoint":
        endpoint_columns = {}
        endpoint_assembly = {}
        for side in config["sides"]:
            endpoint_columns[side], endpoint_assembly[side] = make_endpoint_training_columns(training_rows, side, config)
        assembly = endpoint_assembly
    else:
        columns, assembly = make_depth_training_columns(training_rows, config)
    write_json_atomic(progress_path, {"phase": "assembled", "binding": binding, "assembly": assembly})
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "binding": binding, "assembly": assembly}, indent=2, sort_keys=True))
        return

    checkpoint_root = run_root / "checkpoints" / config["experiment_id"]
    reused: Any
    if config["trainer_kind"] == "pair":
        model, seconds, reused = train_cross_encoder(
            config=config,
            columns=columns,
            output_root=checkpoint_root,
            binding_hash=binding_hash,
        )
        metrics, details = evaluate_pair_model(model, holdout_rows, config)
        training_seconds: Any = seconds
    elif config["trainer_kind"] == "endpoint":
        models, seconds_by_side, reused_by_side = {}, {}, {}
        for side in config["sides"]:
            models[side], seconds_by_side[side], reused_by_side[side] = train_cross_encoder(
                config=config,
                columns=endpoint_columns[side],
                output_root=checkpoint_root / side,
                binding_hash=hashlib.sha256(f"{binding_hash}:{side}".encode()).hexdigest(),
            )
        metrics, details = evaluate_endpoint_models(models, holdout_rows, config)
        training_seconds = seconds_by_side
        reused = reused_by_side
    else:
        model, seconds, reused = train_cross_encoder(
            config=config,
            columns=columns,
            output_root=checkpoint_root,
            binding_hash=binding_hash,
            num_labels=int(config["num_labels"]),
            class_weights=[float(value) for value in config["class_weights"]],
        )
        metrics, details = evaluate_depth_model(model, holdout_rows, config)
        training_seconds = seconds

    result = {
        "status": "completed",
        "run_id": config["run_id"],
        "experiment_id": config["experiment_id"],
        "trainer_kind": config["trainer_kind"],
        "scientific_status": "reranker_train internal holdout; not a development or test claim",
        "binding": binding,
        "binding_hash": binding_hash,
        "assembly": assembly,
        "metrics": metrics,
        "training_seconds": training_seconds,
        "checkpoint_reused": reused,
        "rows_detail": details,
    }
    write_json_atomic(result_path, result)
    write_json_atomic(progress_path, {
        "phase": "complete",
        "binding": binding,
        "assembly": assembly,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
    })
    print(json.dumps({"status": "completed", "experiment_id": config["experiment_id"], "metrics": metrics}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise

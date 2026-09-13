#!/usr/bin/env python3
"""Train/evaluate one Dataset-v2 cross-encoder arm on a single declared V100."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from experiment_core import (
    DATASET_ID,
    independent_rank_metrics,
    load_candidate_artifact,
    pair_id,
    pair_rank_metrics,
    read_json,
    serialize_pair_document,
    sha256_file,
    write_json,
)


SELECTORS = {"independent", "pair"}
VIEWS = {"plain", "schema", "service"}


def find_resume_checkpoint(output_dir: Path) -> Path | None:
    candidates = []
    required = {
        "trainer_state.json", "model.safetensors", "optimizer.pt",
        "scheduler.pt", "rng_state.pth",
    }
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("checkpoint-"):
                continue
            suffix = path.name.removeprefix("checkpoint-")
            if suffix.isdigit() and all((path / name).is_file() for name in required):
                candidates.append((int(suffix), path))
    return max(candidates, default=(0, None))[1]


def validated_final_checkpoint_metadata(checkpoint: Path, expected_binding: dict) -> dict:
    metadata_path = checkpoint / "round3_checkpoint.json"
    if not checkpoint.is_dir() or not metadata_path.is_file() or not (checkpoint / "model.safetensors").is_file():
        raise RuntimeError("final checkpoint is incomplete")
    metadata = read_json(metadata_path)
    if metadata.get("binding") != expected_binding:
        raise RuntimeError("final checkpoint binding does not match the current run")
    return metadata


def load_config(path: Path) -> dict:
    config = read_json(path)
    required = {
        "run_id", "experiment_id", "gpu", "task_level", "selector", "view",
        "base_model", "base_model_revision", "candidate_depth", "group_size",
        "epochs", "batch_size", "gradient_accumulation_steps", "learning_rate",
        "warmup_ratio", "max_length", "seed", "fp16", "train_candidates",
        "train_manifest", "dev_candidates", "dev_manifest", "trigger_corpus",
        "action_corpus", "prediction_batch_size", "pair_chars_per_side",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration lacks fields: {sorted(missing)}")
    if config["selector"] not in SELECTORS or config["view"] not in VIEWS:
        raise ValueError("unknown selector or view")
    if config["task_level"] not in {"function", "service"}:
        raise ValueError("task_level must be function or service")
    if config["task_level"] == "service" and config["view"] != "service":
        raise ValueError("service experiments require the service view")
    if config["task_level"] == "function" and config["view"] == "service":
        raise ValueError("function experiments cannot use the service view")
    if not 2 <= int(config["candidate_depth"]) <= 20:
        raise ValueError("candidate depth outside the declared safety range")
    for field in ("train_candidates", "train_manifest", "dev_candidates", "dev_manifest"):
        if "test" in Path(str(config[field])).name.casefold():
            raise ValueError("locked test paths are prohibited")
    return config


def assert_single_gpu(expected: int) -> None:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible != [str(expected)]:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={expected}, got {visible}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) != (7, 0):
        raise RuntimeError(f"this run is pinned to V100 sm_70, got sm_{major}{minor}")


def corpus_map(path: Path, task_level: str, side: str) -> dict[str, dict]:
    rows = read_json(path)
    result = {}
    for row in rows:
        if task_level == "service":
            identifier = row["service_id"]
            text = row["text"]
            item = {
                "url": identifier, "channel": identifier, "function_name": text,
                "text_plain": text, "text_schema": text, "service_name": text,
                "retrieval_rank": 10**6, "retrieval_score": float("-inf"),
            }
        else:
            identifier = row["url"]
            item = dict(row)
            item.setdefault("retrieval_rank", 10**6)
            item.setdefault("retrieval_score", float("-inf"))
        if identifier in result:
            raise ValueError(f"duplicate {side} corpus identity: {identifier}")
        result[identifier] = item
    if not result:
        raise ValueError(f"empty {side} corpus")
    return result


def candidate_text(item: dict, view: str, side: str) -> str:
    value = item.get("service_name") if view == "service" else item.get(f"text_{view}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {view} text for {item.get('url')}")
    return f"{side.upper()}: {value.strip()}"


def valid_sets(row: dict) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    pairs = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
    return {t for t, _ in pairs}, {a for _, a in pairs}, pairs


def independent_groups(
    rows: list[dict], corpora: dict[str, dict[str, dict]], *, view: str, depth: int,
) -> tuple[dict[str, list], dict]:
    output = {"query": [], "docs": [], "labels": []}
    excluded_without_positive = 0
    for row in rows:
        trigger_gold, action_gold, _ = valid_sets(row)
        for side, gold in (("trigger", trigger_gold), ("action", action_gold)):
            candidates = row[f"{side}_candidates"][:depth]
            labels = [float(item["url"] in gold) for item in candidates]
            if not any(labels):
                excluded_without_positive += 1
                continue
            if all(labels):
                raise ValueError("each listwise group needs at least one negative")
            output["query"].append(f"Select the exact {side} for: {row['query']}")
            output["docs"].append([candidate_text(item, view, side) for item in candidates])
            output["labels"].append(labels)
    return output, {
        "groups": len(output["query"]),
        "excluded_without_positive": excluded_without_positive,
        "gold_injection": False,
    }


def _pair_document(trigger: dict, action: dict, view: str, per_side_chars: int) -> str:
    return serialize_pair_document(
        candidate_text(trigger, view, "trigger"),
        candidate_text(action, view, "action"),
        per_side_chars=per_side_chars,
    )


def pair_groups(
    rows: list[dict], corpora: dict[str, dict[str, dict]], *, view: str,
    depth: int, group_size: int, pair_chars_per_side: int,
) -> tuple[dict[str, list], dict]:
    output = {"query": [], "docs": [], "labels": []}
    excluded_without_positive = 0
    for row in rows:
        _, _, gold_pairs = valid_sets(row)
        triggers = row["trigger_candidates"][:depth]
        actions = row["action_candidates"][:depth]
        trigger_by_id = {item["url"]: item for item in triggers}
        action_by_id = {item["url"]: item for item in actions}
        positives = [
            (trigger_by_id[trigger], action_by_id[action])
            for trigger, action in sorted(gold_pairs)
            if trigger in trigger_by_id and action in action_by_id
        ]
        if not positives:
            excluded_without_positive += 1
            continue
        negatives = [
            (trigger, action)
            for trigger in triggers for action in actions
            if (trigger["url"], action["url"]) not in gold_pairs
        ]
        negatives.sort(key=lambda pair: (
            int(pair[0]["retrieval_rank"]) + int(pair[1]["retrieval_rank"]),
            max(int(pair[0]["retrieval_rank"]), int(pair[1]["retrieval_rank"])),
            pair[0]["url"], pair[1]["url"],
        ))
        keep_negatives = max(1, group_size - len(positives))
        selected = positives + negatives[:keep_negatives]
        labels = [1.0] * len(positives) + [0.0] * min(len(negatives), keep_negatives)
        if not positives or not negatives:
            raise ValueError("pair group needs positive and negative documents")
        output["query"].append(row["query"])
        output["docs"].append([_pair_document(t, a, view, pair_chars_per_side) for t, a in selected])
        output["labels"].append(labels)
    return output, {
        "groups": len(output["query"]),
        "excluded_without_positive": excluded_without_positive,
        "gold_injection": False,
    }


def first_relevant_rank(order: list[str], relevant: set[str]) -> int | None:
    return next((index for index, identifier in enumerate(order, start=1) if identifier in relevant), None)


def exact_mcnemar(before: list[int], after: list[int]) -> dict:
    recovered = sum(x == 0 and y == 1 for x, y in zip(before, after))
    regressed = sum(x == 1 and y == 0 for x, y in zip(before, after))
    discordant = recovered + regressed
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(recovered, regressed) + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {"recoveries": recovered, "regressions": regressed, "discordant": discordant, "exact_p": p_value}


def bootstrap_delta(before: list[int], after: list[int], *, seed: int, iterations: int = 3000) -> dict:
    rng = random.Random(seed)
    count = len(before)
    values = []
    for _ in range(iterations):
        indices = [rng.randrange(count) for _ in range(count)]
        values.append(sum(after[i] - before[i] for i in indices) / count)
    values.sort()
    return {
        "delta": sum(y - x for x, y in zip(before, after)) / count,
        "ci95": [values[int(0.025 * iterations)], values[int(0.975 * iterations) - 1]],
        "iterations": iterations,
    }


def score_independent(model: Any, rows: list[dict], view: str, depth: int, batch: int) -> tuple[list[dict], dict]:
    details = []
    for row in rows:
        rankings = {}
        scores = {}
        trigger_gold, action_gold, valid_pairs = valid_sets(row)
        for side, gold in (("trigger", trigger_gold), ("action", action_gold)):
            candidates = row[f"{side}_candidates"][:depth]
            pairs = [(f"Select the exact {side} for: {row['query']}", candidate_text(item, view, side)) for item in candidates]
            predicted = model.predict(pairs, batch_size=batch, show_progress_bar=False)
            if not all(math.isfinite(float(value)) for value in predicted):
                raise RuntimeError("cross-encoder produced non-finite scores")
            order = sorted(range(len(candidates)), key=lambda index: (-float(predicted[index]), candidates[index]["url"]))
            rankings[side] = [candidates[index]["url"] for index in order]
            scores[side] = [float(predicted[index]) for index in order]
        trigger_positions = {identifier: index for index, identifier in enumerate(rankings["trigger"], start=1)}
        action_positions = {identifier: index for index, identifier in enumerate(rankings["action"], start=1)}
        joint_rank = min(
            (max(trigger_positions[t], action_positions[a]) for t, a in valid_pairs if t in trigger_positions and a in action_positions),
            default=None,
        )
        details.append({
            "group_id": row["group_id"],
            "trigger_rank": first_relevant_rank(rankings["trigger"], trigger_gold),
            "action_rank": first_relevant_rank(rankings["action"], action_gold),
            "joint_rank": joint_rank,
            "trigger_ranking": rankings["trigger"], "action_ranking": rankings["action"],
            "trigger_scores": scores["trigger"], "action_scores": scores["action"],
        })
    return details, independent_rank_metrics(details)


def score_pairs(
    model: Any, rows: list[dict], view: str, depth: int, batch: int, pair_chars_per_side: int,
) -> tuple[list[dict], dict]:
    details = []
    for row in rows:
        _, _, valid = valid_sets(row)
        candidates = [
            (trigger, action)
            for trigger in row["trigger_candidates"][:depth]
            for action in row["action_candidates"][:depth]
        ]
        values = model.predict(
            [(row["query"], _pair_document(t, a, view, pair_chars_per_side)) for t, a in candidates],
            batch_size=batch, show_progress_bar=False,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("pair cross-encoder produced non-finite scores")
        order = sorted(
            range(len(candidates)),
            key=lambda index: (-float(values[index]), candidates[index][0]["url"], candidates[index][1]["url"]),
        )
        ranking = [pair_id(candidates[i][0]["url"], candidates[i][1]["url"]) for i in order]
        relevant = {pair_id(trigger, action) for trigger, action in valid}
        details.append({
            "group_id": row["group_id"],
            "pair_rank": first_relevant_rank(ranking, relevant),
            "pair_ranking": ranking,
            "pair_scores": [float(values[i]) for i in order],
        })
    return details, pair_rank_metrics(details)


def baseline_hits(rows: list[dict]) -> list[int]:
    output = []
    for row in rows:
        pair = (row["trigger_candidates"][0]["url"], row["action_candidates"][0]["url"])
        output.append(int(pair in valid_sets(row)[2]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    run_root = args.run_root.resolve()
    config = load_config(config_path)
    experiment_id = config["experiment_id"]
    progress_path = run_root / "manifests" / f"{experiment_id}.progress.json"
    result_path = run_root / "results" / f"{experiment_id}_dev.json"
    checkpoint_path = run_root / "checkpoints" / experiment_id / "final"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite an existing final result")

    assert_single_gpu(int(config["gpu"]))
    train_rows, train_manifest = load_candidate_artifact(
        Path(config["train_candidates"]), Path(config["train_manifest"]),
    )
    dev_rows, dev_manifest = load_candidate_artifact(
        Path(config["dev_candidates"]), Path(config["dev_manifest"]),
    )
    if train_manifest["split"] != "reranker_train" or dev_manifest["split"] != "dev":
        raise ValueError("reranker training requires reranker_train -> dev")
    corpora = {
        "trigger": corpus_map(Path(config["trigger_corpus"]), config["task_level"], "trigger"),
        "action": corpus_map(Path(config["action_corpus"]), config["task_level"], "action"),
    }
    if config["selector"] == "independent":
        dataset_columns, assembly = independent_groups(
            train_rows, corpora, view=config["view"], depth=int(config["candidate_depth"]),
        )
    else:
        dataset_columns, assembly = pair_groups(
            train_rows, corpora, view=config["view"], depth=int(config["candidate_depth"]),
            group_size=int(config["group_size"]),
            pair_chars_per_side=int(config["pair_chars_per_side"]),
        )
    binding = {
        "dataset_id": DATASET_ID,
        "config_sha256": sha256_file(config_path),
        "train_candidates_sha256": train_manifest["output_sha256"],
        "dev_candidates_sha256": dev_manifest["output_sha256"],
        "train_rows": len(train_rows), "dev_rows": len(dev_rows),
        "assembly": assembly, "config": config,
    }
    write_json(progress_path, {"phase": "assembled", "binding": binding})
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", **binding}, indent=2, sort_keys=True))
        return

    import numpy as np
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import LambdaLoss
    from transformers import TrainerCallback

    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if bool(config["fp16"]) and torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("fp16 configuration is pinned to V100")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_reused = checkpoint_path.exists()
    if checkpoint_reused:
        checkpoint_metadata = validated_final_checkpoint_metadata(checkpoint_path, binding)
        training_seconds = float(checkpoint_metadata["training_seconds"])
        model = CrossEncoder(
            str(checkpoint_path), num_labels=1, max_length=int(config["max_length"]), device="cuda:0",
        )
    else:
        model = CrossEncoder(
            config["base_model"], num_labels=1, max_length=int(config["max_length"]), device="cuda:0",
        )

        class ProgressCallback(TrainerCallback):
            def on_log(self, training_args, state, control, logs=None, **kwargs):
                write_json(progress_path, {
                    "phase": "training", "binding": binding, "global_step": int(state.global_step),
                    "epoch": state.epoch, "max_steps": int(state.max_steps), "latest_log": logs or {},
                })

        trainer_output = run_root / "checkpoints" / experiment_id / "trainer"
        resume_checkpoint = find_resume_checkpoint(trainer_output)
        training_args = CrossEncoderTrainingArguments(
            output_dir=str(trainer_output),
            num_train_epochs=float(config["epochs"]),
            per_device_train_batch_size=int(config["batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            learning_rate=float(config["learning_rate"]), warmup_ratio=float(config["warmup_ratio"]),
            fp16=bool(config["fp16"]), bf16=False, seed=int(config["seed"]), data_seed=int(config["seed"]),
            save_strategy="epoch", save_total_limit=1, logging_steps=25,
            dataloader_num_workers=2, report_to=[],
        )
        started = time.monotonic()
        trainer = CrossEncoderTrainer(
            model=model, args=training_args, train_dataset=Dataset.from_dict(dataset_columns),
            loss=LambdaLoss(model), callbacks=[ProgressCallback()],
        )
        trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
        training_seconds = time.monotonic() - started
        staging = checkpoint_path.parent / f".final.staging-{os.getpid()}"
        if staging.exists():
            raise RuntimeError(f"checkpoint staging path already exists: {staging}")
        model.save_pretrained(str(staging))
        write_json(staging / "round3_checkpoint.json", {
            "binding": binding,
            "training_seconds": training_seconds,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        })
        os.replace(staging, checkpoint_path)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    write_json(progress_path, {
        "phase": "evaluating", "binding": binding,
        "training_seconds": training_seconds, "checkpoint_reused": checkpoint_reused,
    })

    if config["selector"] == "independent":
        details, metrics = score_independent(
            model, dev_rows, config["view"], int(config["candidate_depth"]), int(config["prediction_batch_size"]),
        )
        after = [int((row["joint_rank"] or 10**9) == 1) for row in details]
        metric_unit = "independently-ranked exact function/service sides"
    else:
        details, metrics = score_pairs(
            model, dev_rows, config["view"], int(config["candidate_depth"]), int(config["prediction_batch_size"]),
            int(config["pair_chars_per_side"]),
        )
        after = [int((row["pair_rank"] or 10**9) == 1) for row in details]
        metric_unit = "ordered exact trigger-action pair list"
    before = baseline_hits(dev_rows)
    comparison = exact_mcnemar(before, after) | bootstrap_delta(before, after, seed=int(config["seed"]))
    result = {
        "status": "completed", "dataset_id": DATASET_ID, "split": "dev",
        "task_level": config["task_level"], "selector": config["selector"], "view": config["view"],
        "metric_unit": metric_unit, "rows": len(dev_rows), "metrics": metrics,
        "retrieval_top1": sum(before) / len(before), "top1_comparison": comparison,
        "training_seconds": training_seconds, "parameter_count": parameter_count,
        "checkpoint_reused_for_evaluation": checkpoint_reused,
        "checkpoint": str(checkpoint_path), "binding": binding, "rows_detail": details,
    }
    write_json(result_path, result)
    write_json(progress_path, {
        "phase": "complete", "binding": binding, "training_seconds": training_seconds,
        "output": str(result_path), "output_sha256": sha256_file(result_path),
        "checkpoint": str(checkpoint_path),
    })
    print(json.dumps({"status": "completed", "experiment_id": experiment_id, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            config_arg = Path(sys.argv[sys.argv.index("--config") + 1])
            root_arg = Path(sys.argv[sys.argv.index("--run-root") + 1])
            config = load_config(config_arg)
            write_json(root_arg / "manifests" / f"{config['experiment_id']}.progress.json", {
                "phase": "failed", "error_type": type(exc).__name__,
            })
        except Exception:
            pass
        raise

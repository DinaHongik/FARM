#!/usr/bin/env python3
"""Train and reload-validate one side of one isolated FARM condition."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    assert_one_visible_gpu,
    assert_new_or_resumable_output,
    config_from_path,
    load_training_rows,
    prompt_text,
    read_json,
    saved_model_kwargs,
    sha256_file,
    utc_now,
    validate_dataset,
    validate_training_rows,
    write_json,
)


def latest_complete_checkpoint(trainer_root: Path) -> Path | None:
    candidates = []
    for path in trainer_root.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
        if all((path / name).is_file() for name in required) and not (
            path / "checkpoint-is-incomplete.txt"
        ).exists():
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=("trigger", "action"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--scope", choices=("full", "smoke"), default="full")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    config = config_from_path(args.config.resolve())
    physical_gpu = assert_one_visible_gpu(config["gpu"])
    manifest = validate_dataset(data_root)

    import numpy as np
    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers
    from transformers import TrainerCallback

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False

    rows = load_training_rows(config, data_root, run_root, args.kind)
    if args.max_rows:
        rows = rows[: args.max_rows]
    row_validation = validate_training_rows(config, data_root, run_root, args.kind, rows)
    if len(rows) < config["batch_size"]:
        raise RuntimeError("training subset is smaller than one batch")

    scope_root = run_root if args.scope == "full" else run_root / "smoke"
    side_root = scope_root / "checkpoints" / config["experiment_id"] / args.kind
    trainer_root = side_root / "trainer"
    final_root = side_root / "final"
    done_path = scope_root / "manifests" / config["experiment_id"] / f"train_{args.kind}.done.json"
    progress_path = scope_root / "manifests" / config["experiment_id"] / f"train_{args.kind}.progress.json"
    if done_path.exists():
        print(json.dumps({"status": "already_complete", "done": str(done_path)}), flush=True)
        return
    assert_new_or_resumable_output(final_root, args.resume)
    trainer_root.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(
        config["base_model"],
        revision=config["base_model_revision"],
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.float32},
    )
    model.max_seq_length = config["max_seq_length"]
    if "query" not in model.prompts or "document" not in model.prompts:
        raise RuntimeError("base model lacks required query/document prompts")

    columns: dict[str, list[str]] = {
        "anchor": [prompt_text(model, "query", row["anchor"]) for row in rows],
        "positive": [prompt_text(model, "document", row["positive"]) for row in rows],
    }
    for index in range(config["hard_negatives"]):
        columns[f"negative_{index + 1}"] = [
            prompt_text(model, "document", row["negatives"][index]) for row in rows
        ]
    dataset = Dataset.from_dict(columns)

    if config.get("cached_loss"):
        loss = losses.CachedMultipleNegativesRankingLoss(
            model,
            scale=config["scale"],
            mini_batch_size=config["mini_batch_size"],
        )
    else:
        loss = losses.MultipleNegativesRankingLoss(model, scale=config["scale"])

    started = time.monotonic()

    class FiniteProgressCallback(TrainerCallback):
        def on_log(self, training_args, state, control, logs=None, **kwargs):
            logs = dict(logs or {})
            loss_value = logs.get("loss")
            if loss_value is not None and not math.isfinite(float(loss_value)):
                raise RuntimeError(f"non-finite loss at step {state.global_step}: {loss_value}")
            record = {
                "status": "running",
                "updated_at": utc_now(),
                "experiment_id": config["experiment_id"],
                "kind": args.kind,
                "scope": args.scope,
                "pid": os.getpid(),
                "physical_gpu": int(physical_gpu),
                "step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
                "elapsed_seconds": time.monotonic() - started,
                "loss": float(loss_value) if loss_value is not None else None,
                "learning_rate": logs.get("learning_rate"),
            }
            write_json(progress_path, record)

    training_kwargs = dict(
        output_dir=str(trainer_root),
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        seed=config["seed"],
        data_seed=config["seed"],
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=5,
        logging_first_step=True,
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        report_to=[],
    )
    if args.max_steps:
        training_kwargs["max_steps"] = args.max_steps
    training_args = SentenceTransformerTrainingArguments(**training_kwargs)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
        callbacks=[FiniteProgressCallback()],
    )
    resume_path = latest_complete_checkpoint(trainer_root) if args.resume else None
    result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    training_loss = float(result.training_loss)
    if not math.isfinite(training_loss):
        raise RuntimeError(f"non-finite final training loss: {training_loss}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(final_root))

    sentinel_query = "When a new item appears, archive it"
    sentinel_document = "channel: example | function: archive\nArchive a new item."
    before = np.vstack([
        model.encode_query(sentinel_query, normalize_embeddings=True, convert_to_numpy=True),
        model.encode_document(sentinel_document, normalize_embeddings=True, convert_to_numpy=True),
    ])
    del trainer, loss
    torch.cuda.empty_cache()
    reload_model = SentenceTransformer(
        str(final_root),
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.float32},
        **saved_model_kwargs(final_root),
    )
    after = np.vstack([
        reload_model.encode_query(sentinel_query, normalize_embeddings=True, convert_to_numpy=True),
        reload_model.encode_document(sentinel_document, normalize_embeddings=True, convert_to_numpy=True),
    ])
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise RuntimeError("checkpoint reload produced non-finite embeddings")
    reload_max_abs_diff = float(np.max(np.abs(before - after)))
    if reload_max_abs_diff > 1e-5:
        raise RuntimeError(f"checkpoint reload changed sentinel embeddings: {reload_max_abs_diff}")

    final_files = sorted(path for path in final_root.rglob("*") if path.is_file())
    record = {
        "status": "passed",
        "completed_at": utc_now(),
        "dataset_id": manifest["dataset_id"],
        "experiment_id": config["experiment_id"],
        "kind": args.kind,
        "scope": args.scope,
        "pid": os.getpid(),
        "physical_gpu": int(physical_gpu),
        "base_model": config["base_model"],
        "base_model_revision": config["base_model_revision"],
        "config_sha256": sha256_file(args.config.resolve()),
        "prompt_policy": {"query": model.prompts["query"], "document": model.prompts["document"]},
        "seed": config["seed"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "max_steps_override": args.max_steps or None,
        "learning_rate": config["learning_rate"],
        "max_seq_length": config["max_seq_length"],
        "loss_class": "CachedMultipleNegativesRankingLoss" if config.get("cached_loss") else "MultipleNegativesRankingLoss",
        "scale": config["scale"],
        "explicit_negative_count": config["hard_negatives"],
        "negative_source": config["negative_source"],
        "hardness_weighting": None,
        "rows": row_validation,
        "global_steps": int(result.global_step),
        "training_loss": training_loss,
        "training_seconds": time.monotonic() - started,
        "resumed_from": str(resume_path) if resume_path else None,
        "checkpoint": str(final_root),
        "checkpoint_bytes": sum(path.stat().st_size for path in final_files),
        "checkpoint_file_hashes": {
            str(path.relative_to(final_root)): sha256_file(path) for path in final_files
        },
        "reload_max_abs_diff": reload_max_abs_diff,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "sentence-transformers", "transformers", "datasets")
        },
    }
    write_json(done_path, record)
    write_json(progress_path, {**record, "step": record["global_steps"]})
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Five-step GPU smoke test for one Dataset v2 Stage-1 arm.

This is intentionally separate from the historical trainer so syncing it cannot
overwrite server-only fixes or legacy run directories.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from farm.dataset_v2 import CORPUS_FILENAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("trigger", "action"))
    parser.add_argument("--view", choices=("plain", "schema"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.training_args import BatchSamplers

    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        raise RuntimeError(f"smoke process requires exactly one CUDA_VISIBLE_DEVICES entry, got {visible}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible CUDA device, got {torch.cuda.device_count()}")

    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    pair_name = f"pairs/{args.kind}_encoder_train_{args.view}.json"
    pair_path = args.data_root / pair_name
    rows = json.loads(pair_path.read_text(encoding="utf-8"))
    rows = rows[: args.max_rows]
    if len(rows) < args.batch_size:
        raise RuntimeError(f"not enough rows for a batch: {len(rows)}")

    corpus = {
        row["url"]: row
        for row in json.loads(
            (args.data_root / "corpus" / CORPUS_FILENAMES[args.kind]).read_text(encoding="utf-8")
        )
    }
    for row in rows:
        expected = corpus[row["label_url"]][f"text_{args.view}"]
        if row["positive"] != expected:
            raise AssertionError(f"positive/document mismatch for {row['label_url']}")

    torch.manual_seed(args.seed)
    dataset = Dataset.from_dict({
        "anchor": [row["anchor"] for row in rows],
        "positive": [row["positive"] for row in rows],
    })
    model = SentenceTransformer(
        args.model,
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.float32},
    )
    model.max_seq_length = 512
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)
    args.out.mkdir(parents=True, exist_ok=True)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.out / "trainer"),
        max_steps=args.steps,
        num_train_epochs=1,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        seed=args.seed,
        fp16=False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    result = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=loss,
    ).train()
    training_loss = float(result.training_loss)
    if not math.isfinite(training_loss):
        raise RuntimeError(f"non-finite training loss: {training_loss}")
    model.save(str(args.out / "final"))

    run_manifest = {
        "status": "passed",
        "dataset_id": manifest["dataset_id"],
        "dataset_recipe_id": manifest["recipe_id"],
        "pair_artifact": pair_name,
        "pair_sha256": manifest["artifacts"][pair_name]["sha256"],
        "kind": args.kind,
        "view": args.view,
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "rows_loaded": len(rows),
        "batch_size": args.batch_size,
        "visible_gpu": visible[0],
        "training_loss": training_loss,
    }
    (args.out / "smoke_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(run_manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

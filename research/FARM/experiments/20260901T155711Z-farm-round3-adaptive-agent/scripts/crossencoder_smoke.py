#!/usr/bin/env python3
"""One-device finite-logit and reload smoke test for the pinned BGE base."""
from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

from experiment_core import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite cross-encoder smoke output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if visible != [str(args.expected_gpu)]:
        raise RuntimeError("cross-encoder smoke requires exactly the declared physical GPU")

    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import LambdaLoss

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("cross-encoder smoke is pinned to one V100")
    pairs = [
        ("When a photo is posted, save it", "TRIGGER: new social photo\nACTION: upload file"),
        ("When a photo is posted, save it", "TRIGGER: hourly clock\nACTION: send thermostat command"),
    ]
    model = CrossEncoder(str(args.model), num_labels=1, max_length=512, device="cuda:0")
    logits_first = [float(value) for value in model.predict(pairs, show_progress_bar=False)]
    if not all(math.isfinite(value) for value in logits_first):
        raise RuntimeError("first-load logits are non-finite")

    positive = "TRIGGER: new social photo " + "event field " * 100 + "\nACTION: upload file " + "mapping field " * 100
    negatives = [
        f"TRIGGER: unrelated event {index} " + "event field " * 100
        + f"\nACTION: unrelated action {index} " + "mapping field " * 100
        for index in range(1, 25)
    ]
    training = Dataset.from_dict({
        "query": ["When a photo is posted, save it"],
        "docs": [[positive, *negatives]],
        "labels": [[1.0, *([0.0] * 24)]],
    })
    with tempfile.TemporaryDirectory(prefix="farm-r3-ce-smoke-", dir=args.output.parent) as directory:
        work = Path(directory)
        arguments = CrossEncoderTrainingArguments(
            output_dir=str(work / "trainer"),
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            fp16=False,
            bf16=False,
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            disable_tqdm=True,
            seed=42,
            data_seed=42,
        )
        trainer = CrossEncoderTrainer(
            model=model,
            args=arguments,
            train_dataset=training,
            loss=LambdaLoss(model),
        )
        train_result = trainer.train()
        if int(trainer.state.global_step) != 1:
            raise RuntimeError("cross-encoder smoke did not complete exactly one optimizer step")
        if not math.isfinite(float(train_result.training_loss)):
            raise RuntimeError("cross-encoder smoke training loss is non-finite")
        token_ids_before = model.tokenizer(
            [query for query, _ in pairs],
            [document for _, document in pairs],
            padding=True,
            truncation=True,
            max_length=512,
        )["input_ids"]
        saved = work / "trained"
        model.save_pretrained(str(saved))
        del trainer, model
        torch.cuda.empty_cache()
        reloaded = CrossEncoder(
            str(saved), num_labels=1, max_length=512, device="cuda:0",
        )
        token_ids_after = reloaded.tokenizer(
            [query for query, _ in pairs],
            [document for _, document in pairs],
            padding=True,
            truncation=True,
            max_length=512,
        )["input_ids"]
        if token_ids_before != token_ids_after:
            raise RuntimeError("saved/reloaded tokenizer changed the smoke token IDs")
        logits_second = [float(value) for value in reloaded.predict(pairs, show_progress_bar=False)]
    if not all(math.isfinite(value) for value in logits_second):
        raise RuntimeError("reload logits are non-finite")
    if not any(abs(before - after) > 1e-10 for before, after in zip(logits_first, logits_second)):
        raise RuntimeError("one-step smoke produced no detectable parameter effect")
    write_json(args.output, {
        "status": "passed",
        "physical_gpu": args.expected_gpu,
        "cuda_capability": "sm_70",
        "model": str(args.model),
        "finite_first_load": True,
        "finite_after_reload": True,
        "tokenization_identical_after_reload": True,
        "optimizer_steps": 1,
        "training_loss": float(train_result.training_loss),
        "logits_first": logits_first,
        "logits_after_reload": logits_second,
    })


if __name__ == "__main__":
    main()

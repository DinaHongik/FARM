"""
Trigger Encoder Training with InfoNCE Loss

Uses MultipleNegativesRankingLoss (InfoNCE) for contrastive learning.
Saves training logs to JSON for paper analysis.
"""

import json
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from transformers import TrainerCallback

# Local imports
from .config import config, TrainingConfig
from .dataset import build_trigger_pairs


class JSONLoggingCallback(TrainerCallback):
    """Callback to save training logs to JSON file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.logs: Dict[str, Any] = {
            "config": {},
            "training_start": None,
            "training_end": None,
            "steps": [],
            "epochs": [],
            "final_metrics": {}
        }

    def on_train_begin(self, args, state, control, **kwargs):
        self.logs["training_start"] = datetime.now().isoformat()
        self.logs["config"] = {
            "base_model": config.base_model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "warmup_ratio": config.warmup_ratio,
            "temperature": config.temperature,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step_log = {
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            }
            self.logs["steps"].append(step_log)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_log = {
            "epoch": int(state.epoch),
            "global_step": state.global_step,
        }
        self.logs["epochs"].append(epoch_log)

    def on_train_end(self, args, state, control, **kwargs):
        self.logs["training_end"] = datetime.now().isoformat()
        self._save_logs()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_log = {
                "step": state.global_step,
                "epoch": round(state.epoch, 2) if state.epoch else 0,
                "metrics": {k: v for k, v in metrics.items()}
            }
            if "eval_loss" in metrics:
                # Add to epochs if available
                for epoch in self.logs["epochs"]:
                    if epoch["global_step"] == state.global_step:
                        epoch["eval_metrics"] = metrics
                        break

    def _save_logs(self):
        """Save logs to JSON file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to: {self.log_path}")

    def set_final_metrics(self, metrics: Dict[str, Any]):
        """Set final evaluation metrics."""
        self.logs["final_metrics"] = metrics
        self._save_logs()


def train_trigger_encoder(cfg: TrainingConfig = None):
    """
    Train the trigger encoder using InfoNCE loss.

    Args:
        cfg: Training configuration. Uses default if not provided.
    """
    cfg = cfg or config

    print("=" * 60)
    print("FARM Trigger Encoder Training")
    print("=" * 60)
    print(f"Base model: {cfg.base_model}")
    print(f"Output dir: {cfg.trigger_output_dir}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Temperature: {cfg.temperature}")
    print("=" * 60)

    # 1. Load base model (float32 for V100 compatibility)
    print("\n[1/5] Loading base model...")
    dtype = torch.float32 if cfg.dtype == "float32" else torch.bfloat16
    model = SentenceTransformer(
        cfg.base_model,
        model_kwargs={"torch_dtype": dtype}
    )
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # =========================================================================
    # LAYER FREEZING: Preserve pretrained semantic knowledge
    # =========================================================================
    # Problem: Full fine-tuning causes "catastrophic forgetting" - model loses
    # semantic understanding (e.g., "log" ~= "add row" ~= "record")
    #
    # Solution: Freeze lower layers that contain general semantics:
    # - embed_tokens (201M params): Word-level semantics
    # - layers 0-11 (50M params): General language understanding
    #
    # Train top 12 layers (50M params) for task-specific discrimination
    # With 12K pairs, we can train more layers without overfitting
    # =========================================================================
    print("\n[1.5/5] Freezing lower layers to preserve semantics...")
    base_model = model[0].auto_model  # Access underlying transformer

    # Freeze embedding layer (201M params - 67% of model)
    for param in base_model.embed_tokens.parameters():
        param.requires_grad = False

    # Freeze layers 0-11 (keep top 12 layers trainable)
    freeze_layers = 12
    for i in range(freeze_layers):
        for param in base_model.layers[i].parameters():
            param.requires_grad = False

    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total params: {total_params:,}")
    print(f"Frozen params: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # 2. Build training pairs from applets
    print("\n[2/5] Building training pairs from applets...")
    train_dataset, val_dataset = build_trigger_pairs(
        cfg.applets_path,
        val_split=cfg.val_split,
        seed=cfg.seed
    )

    # 3. Define InfoNCE loss (MultipleNegativesRankingLoss)
    print("\n[3/5] Setting up InfoNCE loss...")
    loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=1.0 / cfg.temperature,  # scale = 1/τ
    )
    print(f"Loss: MultipleNegativesRankingLoss (InfoNCE)")
    print(f"Temperature τ = {cfg.temperature}, scale = {1.0/cfg.temperature}")

    # 4. Setup training arguments
    print("\n[4/5] Configuring trainer...")
    output_dir = cfg.project_root / cfg.trigger_output_dir

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=False,  # V100 compatibility
        bf16=False,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.evaluation_strategy,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=cfg.seed,
        report_to="none",  # We use custom JSON logging
    )

    # Setup JSON logging callback
    json_logger = JSONLoggingCallback(cfg.trigger_logs_path)

    # Create trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        callbacks=[json_logger],
    )

    # 5. Train
    print("\n[5/5] Starting training...")
    print("-" * 60)
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {elapsed/60:.2f} minutes")

    # Save final model
    final_path = output_dir / "final"
    model.save(str(final_path))
    print(f"Model saved to: {final_path}")

    # Save final metrics
    json_logger.set_final_metrics({
        "training_time_seconds": elapsed,
        "training_time_minutes": elapsed / 60,
        "total_steps": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
    })

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model: {final_path}")
    print(f"Logs:  {cfg.trigger_logs_path}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    train_trigger_encoder()

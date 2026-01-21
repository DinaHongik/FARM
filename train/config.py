"""
Training Configuration for FARM Contrastive Learning

Uses MultipleNegativesRankingLoss (InfoNCE) as specified in README.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for contrastive training."""

    # Model
    base_model: str = "google/embeddinggemma-300m"

    # Output paths
    trigger_output_dir: str = "./models/trigger-encoder"
    action_output_dir: str = "./models/action-encoder"
    logs_dir: str = "./logs"

    # Data paths (relative to project root)
    # Training data: Train split only (90% of 16K applets)
    # Run `python data/split_data.py` first to create this file
    applets_data: str = "./data/train_applets.json"
    # RAG index data (unique APIs for retrieval)
    triggers_rag_data: str = "./data/triggers_rag.json"
    actions_rag_data: str = "./data/actions_rag.json"

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 16  # V100 32GB safe
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # InfoNCE / MultipleNegativesRankingLoss settings
    # Temperature τ for softmax (lower = sharper distribution)
    temperature: float = 0.05

    # Hardware settings (V100 compatibility)
    dtype: str = "float32"  # V100 doesn't support bfloat16 well
    gradient_checkpointing: bool = False

    # Logging
    logging_steps: int = 50
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"

    # Train/val split
    val_split: float = 0.1
    seed: int = 42

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent

    @property
    def applets_path(self) -> Path:
        """Full path to applets training data."""
        return self.project_root / self.applets_data

    @property
    def triggers_rag_path(self) -> Path:
        """Full path to triggers RAG data."""
        return self.project_root / self.triggers_rag_data

    @property
    def actions_rag_path(self) -> Path:
        """Full path to actions RAG data."""
        return self.project_root / self.actions_rag_data

    @property
    def trigger_logs_path(self) -> Path:
        """Path for trigger training logs."""
        return self.project_root / self.logs_dir / "trigger_training.json"

    @property
    def action_logs_path(self) -> Path:
        """Path for action training logs."""
        return self.project_root / self.logs_dir / "action_training.json"


# Default configuration instance
config = TrainingConfig()

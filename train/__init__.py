"""
FARM Training Module

Contrastive training for Trigger and Action encoders using InfoNCE loss.
"""

from .config import TrainingConfig, config
from .dataset import build_trigger_pairs, build_action_pairs

__all__ = [
    "TrainingConfig",
    "config",
    "build_trigger_pairs",
    "build_action_pairs",
]

"""
Configuration for FARM RAG System
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class QdrantConfig:
    """Qdrant vector database configuration."""
    # Use local file storage (no Docker needed)
    # Set to None for in-memory, or path string for persistent storage
    # Path inside rag/ folder (rag/qdrant_data)
    storage_path: str = field(default_factory=lambda: str(Path(__file__).parent / "qdrant_data"))
    # Collection names - using FINE-TUNED (layer_freeze from ablation study)
    triggers_collection: str = "farm_triggers_finetuned"
    actions_collection: str = "farm_actions_finetuned"
    # Collection names for BASELINE (pretrained) models
    triggers_collection_baseline: str = "farm_triggers_baseline"
    actions_collection_baseline: str = "farm_actions_baseline"
    # Collection names for FINE-TUNED (if models are retrained)
    triggers_collection_finetuned: str = "farm_triggers_finetuned"
    actions_collection_finetuned: str = "farm_actions_finetuned"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    # Base model (used when fine-tuned models are not available)
    model_name: str = "google/embeddinggemma-300m"
    # Fine-tuned models for triggers and actions (paths relative to project root)
    trigger_model_path: str = "./models/ablation_layer_freeze_trigger/final"
    action_model_path: str = "./models/ablation_layer_freeze_action/final"
    # Use fine-tuned models (layer_freeze from ablation study)
    use_finetuned: bool = True
    dimension: int = 768
    max_length: int = 2048  # EmbeddingGemma supports up to 2048 tokens
    batch_size: int = 32
    normalize: bool = True
    # EmbeddingGemma requires bfloat16 or float32 (no float16 support)
    dtype: str = "bfloat16"


@dataclass
class DataConfig:
    """Data paths configuration."""
    # Paths relative to project root
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    # Data directory inside FARM folder
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")

    triggers_file: str = "triggers_rag.json"
    actions_file: str = "actions_rag.json"

    @property
    def triggers_path(self) -> Path:
        return self.data_dir / self.triggers_file

    @property
    def actions_path(self) -> Path:
        return self.data_dir / self.actions_file


@dataclass
class RetrievalConfig:
    """Retrieval settings."""
    top_k: int = 10
    score_threshold: float = 0.3  # Lowered from 0.5 to work with baseline embeddings
    # Categories for filtering (from dataset analysis)
    trigger_categories: List[str] = field(default_factory=lambda: [
        "Smart home & IoT",
        "News & information",
        "Popular services",
        "Other",
        "Business tools",
        "Project management & to-dos",
        "New services",
        "Mobile devices & accessories",
        "Gaming & Entertainment",
        "Developer tools",
        "Finance & payments",
        "Health & fitness",
        "Social media",
        "Photo & video",
        "Website & blog",
        "Music",
        "YouTube Channels",
        "Podcasts",
        "Notifications"
    ])
    action_categories: List[str] = field(default_factory=lambda: [
        "Smart home & IoT",
        "Popular services",
        "Other",
        "Business tools",
        "New services",
        "Project management & to-dos",
        "News & information",
        "Social media",
        "Mobile devices & accessories",
        "Music",
        "Developer tools",
        "Website & blog",
        "Finance & payments",
        "Health & fitness",
        "Gaming & Entertainment",
        "Photo & video",
        "Notifications"
    ])


@dataclass
class FARMConfig:
    """Main configuration combining all settings."""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


# Default configuration instance (fine-tuned)
config = FARMConfig()


def get_baseline_config() -> FARMConfig:
    """Get configuration for BASELINE (pretrained) RAG."""
    cfg = FARMConfig()
    cfg.embedding.use_finetuned = False
    # Use baseline collection names
    cfg.qdrant.triggers_collection = cfg.qdrant.triggers_collection_baseline
    cfg.qdrant.actions_collection = cfg.qdrant.actions_collection_baseline
    return cfg


def get_finetuned_config() -> FARMConfig:
    """Get configuration for FINE-TUNED RAG."""
    cfg = FARMConfig()
    cfg.embedding.use_finetuned = True
    # Use fine-tuned collection names
    cfg.qdrant.triggers_collection = cfg.qdrant.triggers_collection_finetuned
    cfg.qdrant.actions_collection = cfg.qdrant.actions_collection_finetuned
    return cfg

"""
Embedding generation using fine-tuned FARM encoders

Supports separate models for triggers and actions, falling back to
base EmbeddingGemma if fine-tuned models not available.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer

# Handle imports for both direct run and module import
try:
    from .config import EmbeddingConfig, config
except ImportError:
    from config import EmbeddingConfig, config


class EmbeddingModel:
    """
    Wrapper for FARM embedding models.

    Supports separate fine-tuned models for triggers and actions,
    with fallback to base model if fine-tuned not available.
    """

    def __init__(self, embedding_config: EmbeddingConfig = None):
        """
        Initialize the embedding model.

        Args:
            embedding_config: Configuration for the embedding model.
                             Uses default config if not provided.
        """
        self.config = embedding_config or config.embedding
        self._base_model = None
        self._trigger_model = None
        self._action_model = None

        # Resolve model paths relative to project root
        self.project_root = Path(__file__).parent.parent

    def _load_model(self, model_path: str) -> Optional[SentenceTransformer]:
        """Load a SentenceTransformer model."""
        # Resolve relative paths
        if not os.path.isabs(model_path):
            full_path = self.project_root / model_path
        else:
            full_path = Path(model_path)

        if full_path.exists():
            print(f"Loading fine-tuned model: {full_path}")
            return SentenceTransformer(str(full_path))
        else:
            print(f"Model not found at {full_path}, using base model")
            return None

    @property
    def base_model(self) -> SentenceTransformer:
        """Load base model (lazy)."""
        if self._base_model is None:
            print(f"Loading base embedding model: {self.config.model_name}")
            dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float32
            self._base_model = SentenceTransformer(
                self.config.model_name,
                model_kwargs={"torch_dtype": dtype}
            )
        return self._base_model

    @property
    def trigger_model(self) -> SentenceTransformer:
        """Load trigger encoder (lazy)."""
        if self._trigger_model is None:
            if self.config.use_finetuned:
                self._trigger_model = self._load_model(self.config.trigger_model_path)
            if self._trigger_model is None:
                self._trigger_model = self.base_model
        return self._trigger_model

    @property
    def action_model(self) -> SentenceTransformer:
        """Load action encoder (lazy)."""
        if self._action_model is None:
            if self.config.use_finetuned:
                self._action_model = self._load_model(self.config.action_model_path)
            if self._action_model is None:
                self._action_model = self.base_model
        return self._action_model

    def encode_triggers(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode trigger texts using the trigger encoder.

        Args:
            texts: Single text or list of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        embeddings = self.trigger_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        return embeddings

    def encode_actions(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode action texts using the action encoder.

        Args:
            texts: Single text or list of texts to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

        Returns:
            numpy array of embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = batch_size or self.config.batch_size

        embeddings = self.action_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        return embeddings

    def encode_trigger_query(self, query: str) -> np.ndarray:
        """
        Encode a query for trigger retrieval.

        Args:
            query: Query text.

        Returns:
            numpy array of shape (dimension,)
        """
        embedding = self.trigger_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        return embedding

    def encode_action_query(self, query: str) -> np.ndarray:
        """
        Encode a query for action retrieval.

        Args:
            query: Query text.

        Returns:
            numpy array of shape (dimension,)
        """
        embedding = self.action_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize
        )
        return embedding

    # Legacy methods for backward compatibility
    def encode_document(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generic document encoding (uses base model)."""
        if isinstance(texts, str):
            texts = [texts]
        batch_size = batch_size or self.config.batch_size

        model = self.base_model
        # Try encode_document if available (EmbeddingGemma)
        if hasattr(model, 'encode_document'):
            return model.encode_document(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize
            )
        else:
            return model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize
            )

    def encode_query(self, query: str) -> np.ndarray:
        """Generic query encoding (uses base model)."""
        model = self.base_model
        if hasattr(model, 'encode_query'):
            return model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize
            )
        else:
            return model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize
            )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generic encode (uses base model)."""
        return self.encode_document(texts, batch_size, show_progress)

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.dimension


# Singleton instance
_embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create the singleton embedding model instance."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel()
    return _embedding_model


def embed_triggers(texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
    """Encode trigger texts."""
    model = get_embedding_model()
    return model.encode_triggers(texts, show_progress=show_progress)


def embed_actions(texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
    """Encode action texts."""
    model = get_embedding_model()
    return model.encode_actions(texts, show_progress=show_progress)


def embed_trigger_query(query: str) -> np.ndarray:
    """Encode query for trigger retrieval."""
    model = get_embedding_model()
    return model.encode_trigger_query(query)


def embed_action_query(query: str) -> np.ndarray:
    """Encode query for action retrieval."""
    model = get_embedding_model()
    return model.encode_action_query(query)


# Legacy functions for backward compatibility
def embed_documents(texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
    """Generic document encoding."""
    model = get_embedding_model()
    return model.encode_document(texts, show_progress=show_progress)


def embed_texts(texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
    """Alias for embed_documents."""
    return embed_documents(texts, show_progress)


def embed_query(query: str) -> np.ndarray:
    """Generic query encoding."""
    model = get_embedding_model()
    return model.encode_query(query)

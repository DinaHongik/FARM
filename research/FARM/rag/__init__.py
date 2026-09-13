"""
FARM RAG Module

Retrieval-Augmented Generation for Trigger-Action APIs using Qdrant.

Supports both BASELINE (pretrained) and FINE-TUNED models.
"""

from .config import FARMConfig, config, get_baseline_config, get_finetuned_config
from .embeddings import EmbeddingModel, get_embedding_model, embed_texts, embed_documents, embed_query
from .indexer import QdrantIndexer
from .retriever import (
    FARMRetriever,
    SearchResult,
    get_retriever,
    get_baseline_retriever,
    get_finetuned_retriever,
    search_triggers,
    search_actions,
    print_results
)

__all__ = [
    # Config
    "FARMConfig",
    "config",
    "get_baseline_config",
    "get_finetuned_config",
    # Embeddings
    "EmbeddingModel",
    "get_embedding_model",
    "embed_texts",
    "embed_documents",
    "embed_query",
    # Indexer
    "QdrantIndexer",
    # Retriever
    "FARMRetriever",
    "SearchResult",
    "get_retriever",
    "get_baseline_retriever",
    "get_finetuned_retriever",
    "search_triggers",
    "search_actions",
    "print_results",
]

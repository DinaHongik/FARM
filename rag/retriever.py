"""
Retriever for FARM RAG System

Provides two-level retrieval:
1. Optional filter by category (metadata)
2. Semantic search within filtered results
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# Handle imports for both direct run and module import
try:
    from .config import FARMConfig, config, get_baseline_config, get_finetuned_config
    from .embeddings import EmbeddingModel
except ImportError:
    from config import FARMConfig, config, get_baseline_config, get_finetuned_config
    from embeddings import EmbeddingModel


@dataclass
class SearchResult:
    """A single search result."""
    id: int
    score: float
    service_name: str
    category: str
    channel: str
    description: str
    api_info: Dict[str, Any]
    embedding_text: str

    @classmethod
    def from_qdrant_result(cls, result) -> "SearchResult":
        """Create from Qdrant search result."""
        return cls(
            id=result.id,
            score=result.score,
            service_name=result.payload.get("service_name", ""),
            category=result.payload.get("category", ""),
            channel=result.payload.get("channel", ""),
            description=result.payload.get("description", ""),
            api_info=result.payload.get("api_info", {}),
            embedding_text=result.payload.get("embedding_text", "")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "service_name": self.service_name,
            "category": self.category,
            "channel": self.channel,
            "description": self.description,
            "api_info": self.api_info
        }


class FARMRetriever:
    """
    Retriever for FARM triggers and actions.

    Supports two-level retrieval:
    1. Filter by category (optional)
    2. Semantic search
    """

    def __init__(self, farm_config: FARMConfig = None):
        """
        Initialize the retriever.

        Args:
            farm_config: Configuration. Uses default if not provided.
        """
        self.config = farm_config or config

        # Use local storage (no Docker needed)
        storage_path = self.config.qdrant.storage_path
        if storage_path:
            # Persistent local storage
            self.client = QdrantClient(path=storage_path)
        else:
            # In-memory (for testing)
            self.client = QdrantClient(":memory:")

        # Create embedding model with this config
        self.embedding_model = EmbeddingModel(self.config.embedding)
        self.is_finetuned = self.config.embedding.use_finetuned

    def _build_filter(
        self,
        category: Optional[str] = None,
        channel: Optional[str] = None
    ) -> Optional[Filter]:
        """
        Build Qdrant filter for category and/or channel.

        Args:
            category: Filter by category (exact match).
            channel: Filter by channel (exact match).

        Returns:
            Qdrant Filter object or None if no filters.
        """
        conditions = []

        if category:
            conditions.append(
                FieldCondition(
                    key="category",
                    match=MatchValue(value=category)
                )
            )

        if channel:
            conditions.append(
                FieldCondition(
                    key="channel",
                    match=MatchValue(value=channel)
                )
            )

        if conditions:
            return Filter(must=conditions)
        return None

    def search_triggers(
        self,
        query: str,
        top_k: int = None,
        category: Optional[str] = None,
        channel: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for triggers matching the query.

        Args:
            query: Search query text.
            top_k: Number of results to return. Uses config default if not provided.
            category: Filter by category (optional).
            channel: Filter by channel (optional).
            score_threshold: Minimum score threshold (optional).

        Returns:
            List of SearchResult objects.
        """
        top_k = top_k if top_k is not None else self.config.retrieval.top_k
        score_threshold = score_threshold if score_threshold is not None else self.config.retrieval.score_threshold

        # Embed query using appropriate encoder
        if self.is_finetuned:
            query_vector = self.embedding_model.encode_trigger_query(query)
        else:
            query_vector = self.embedding_model.encode_query(query)

        # Build filter
        query_filter = self._build_filter(category=category, channel=channel)

        # Search
        results = self.client.search(
            collection_name=self.config.qdrant.triggers_collection,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        )

        return [SearchResult.from_qdrant_result(r) for r in results]

    def search_actions(
        self,
        query: str,
        top_k: int = None,
        category: Optional[str] = None,
        channel: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for actions matching the query.

        Args:
            query: Search query text.
            top_k: Number of results to return. Uses config default if not provided.
            category: Filter by category (optional).
            channel: Filter by channel (optional).
            score_threshold: Minimum score threshold (optional).

        Returns:
            List of SearchResult objects.
        """
        top_k = top_k if top_k is not None else self.config.retrieval.top_k
        score_threshold = score_threshold if score_threshold is not None else self.config.retrieval.score_threshold

        # Embed query using appropriate encoder
        if self.is_finetuned:
            query_vector = self.embedding_model.encode_action_query(query)
        else:
            query_vector = self.embedding_model.encode_query(query)

        # Build filter
        query_filter = self._build_filter(category=category, channel=channel)

        # Search
        results = self.client.search(
            collection_name=self.config.qdrant.actions_collection,
            query_vector=query_vector.tolist(),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold
        )

        return [SearchResult.from_qdrant_result(r) for r in results]

    def search_both(
        self,
        trigger_query: str,
        action_query: str,
        top_k: int = None,
        trigger_category: Optional[str] = None,
        action_category: Optional[str] = None
    ) -> Dict[str, List[SearchResult]]:
        """
        Search for both triggers and actions.

        Args:
            trigger_query: Query for trigger search.
            action_query: Query for action search.
            top_k: Number of results for each.
            trigger_category: Filter triggers by category.
            action_category: Filter actions by category.

        Returns:
            Dictionary with 'triggers' and 'actions' keys.
        """
        triggers = self.search_triggers(
            query=trigger_query,
            top_k=top_k,
            category=trigger_category
        )
        actions = self.search_actions(
            query=action_query,
            top_k=top_k,
            category=action_category
        )

        return {
            "triggers": triggers,
            "actions": actions
        }


# Singleton instances for baseline and fine-tuned
_baseline_retriever = None
_finetuned_retriever = None


def get_baseline_retriever() -> FARMRetriever:
    """Get or create the BASELINE (pretrained) retriever instance."""
    global _baseline_retriever
    if _baseline_retriever is None:
        _baseline_retriever = FARMRetriever(get_baseline_config())
    return _baseline_retriever


def get_finetuned_retriever() -> FARMRetriever:
    """Get or create the FINE-TUNED retriever instance."""
    global _finetuned_retriever
    if _finetuned_retriever is None:
        _finetuned_retriever = FARMRetriever(get_finetuned_config())
    return _finetuned_retriever


def get_retriever(use_finetuned: bool = True) -> FARMRetriever:
    """
    Get retriever instance.

    Args:
        use_finetuned: If True, use fine-tuned models. If False, use baseline.

    Returns:
        FARMRetriever instance.
    """
    if use_finetuned:
        return get_finetuned_retriever()
    else:
        return get_baseline_retriever()


def search_triggers(
    query: str,
    top_k: int = 10,
    category: Optional[str] = None,
    use_finetuned: bool = True
) -> List[SearchResult]:
    """
    Search triggers.

    Args:
        query: Search query text.
        top_k: Number of results.
        category: Optional category filter.
        use_finetuned: Use fine-tuned (True) or baseline (False) model.

    Returns:
        List of SearchResult objects.
    """
    retriever = get_retriever(use_finetuned=use_finetuned)
    return retriever.search_triggers(query, top_k=top_k, category=category)


def search_actions(
    query: str,
    top_k: int = 10,
    category: Optional[str] = None,
    use_finetuned: bool = True
) -> List[SearchResult]:
    """
    Search actions.

    Args:
        query: Search query text.
        top_k: Number of results.
        category: Optional category filter.
        use_finetuned: Use fine-tuned (True) or baseline (False) model.

    Returns:
        List of SearchResult objects.
    """
    retriever = get_retriever(use_finetuned=use_finetuned)
    return retriever.search_actions(query, top_k=top_k, category=category)


def print_results(results: List[SearchResult], title: str = "Results"):
    """Pretty print search results."""
    print(f"\n{title}")
    print("=" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.service_name}")
        print(f"   Category: {r.category}")
        print(f"   Channel: {r.channel or 'N/A'}")
        print(f"   {r.description[:100]}...")
        print()

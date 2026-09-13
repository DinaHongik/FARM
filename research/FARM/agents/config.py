"""
Configuration for FARM Agentic AI System
========================================

LLM factory and configuration constants for the negotiation agents.
Uses Ollama with granite4:small-h as the default model.

Note: Granite 4 models work best with temperature=0 for most tasks.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from langchain_ollama import ChatOllama


@dataclass
class AgentConfig:
    """
    Configuration for the agentic negotiation system.

    Attributes:
        model_name: Ollama model identifier
        temperature: LLM temperature (0 recommended for Granite 4)
        max_retries: Maximum LLM call retries on parse failure
        max_pairs: Maximum trigger-action pairs to try before failing
        verifier_threshold: Minimum score for applet acceptance
        top_k_candidates: Number of candidates to retrieve from RAG
        verbose: Enable debug logging
    """

    # LLM Configuration
    model_name: str = "granite4:small-h"
    temperature: float = 0.0  # Granite 4 works best with temperature=0

    # Retry Configuration
    max_retries: int = 3
    max_pairs: int = 9  # 3x3 grid of trigger-action combinations

    # Verifier Configuration
    # NOTE: Verifier no longer rejects pairs - we trust RAG selection
    # Verifier only generates bindings (1 LLM call total)
    # No arbitrary weights needed - RAG score ranking is used directly
    verifier_threshold: float = 0.6  # Used by verifier_node for is_executable flag

    # RAG Configuration
    top_k_candidates: int = 5  # Match R@5 evaluation (was 3, missed rank 4-5)

    # Debug
    verbose: bool = False

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent
    )

    @property
    def rag_module_path(self) -> Path:
        """Path to the RAG module."""
        return self.project_root / "rag"


# Default configuration instance
config = AgentConfig()


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None
) -> ChatOllama:
    """
    Factory function to create LLM instance.

    Uses Ollama with granite4:small-h by default. The model can be
    easily swapped by passing a different model_name.

    Args:
        model_name: Ollama model identifier. Defaults to config value.
        temperature: Sampling temperature. Defaults to config value.

    Returns:
        ChatOllama instance configured for the specified model.

    Example:
        # Default (Granite 4)
        llm = get_llm()

        # Alternative models
        llm = get_llm("llama3.1:8b")
        llm = get_llm("qwen2.5:7b")
    """
    return ChatOllama(
        model=model_name or config.model_name,
        temperature=temperature if temperature is not None else config.temperature,
    )


# NOTE: PAIR_ORDER removed - we now use RAG score ranking directly
# Pairs are ranked by: trigger_rag_score + action_rag_score
# This is computed in trigger_selector_node._compute_pair_ranking()
# Benefits:
# - Uses all 25 pairs (5x5), not just 9 (3x3)
# - Ranking based on trained RAG embeddings
# - No hardcoded magic order

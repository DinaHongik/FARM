#!/usr/bin/env python3
"""Debug RAG retrieval - compare fine-tuned vs baseline."""

from rag.retriever import FARMRetriever
from rag.config import FARMConfig

def test_rag(use_finetuned: bool):
    """Test RAG with specific model setting."""
    label = "FINE-TUNED" if use_finetuned else "BASELINE"
    print(f"\n{'='*60}")
    print(f"  {label} MODEL")
    print(f"{'='*60}")

    # Create config with specific setting
    config = FARMConfig()
    config.embedding.use_finetuned = use_finetuned
    if use_finetuned:
        config.qdrant.triggers_collection = "farm_triggers_finetuned"
        config.qdrant.actions_collection = "farm_actions_finetuned"
    else:
        config.qdrant.triggers_collection = "farm_triggers_baseline"
        config.qdrant.actions_collection = "farm_actions_baseline"

    retriever = FARMRetriever(config)

    # Test queries - semantic queries (not exact service names)
    queries = [
        ("log to spreadsheet", "action"),
        ("add row to google sheets", "action"),
        ("send notification", "action"),
        ("darkness detected", "trigger"),
        ("when it gets dark", "trigger"),
        ("new email arrives", "trigger"),
    ]

    for query, search_type in queries:
        print(f"\n--- {search_type.upper()}: '{query}' ---")
        if search_type == "action":
            results = retriever.search_actions(query, top_k=3, score_threshold=0.0)
        else:
            results = retriever.search_triggers(query, top_k=3, score_threshold=0.0)

        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.service_name}")

if __name__ == "__main__":
    print("\nComparing BASELINE vs FINE-TUNED RAG results...")

    # Test baseline first
    test_rag(use_finetuned=False)

    # Then fine-tuned
    test_rag(use_finetuned=True)

    print("\n" + "="*60)
    print("  COMPARISON COMPLETE")
    print("="*60)

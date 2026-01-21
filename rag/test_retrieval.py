"""
Test script for FARM RAG retrieval
"""

from retriever import get_retriever, search_triggers, search_actions, print_results


def test_retrieval():
    """Test trigger and action retrieval."""

    print("=" * 60)
    print("FARM RAG Retrieval Test")
    print("=" * 60)

    # Test queries
    test_queries = [
        "turn on lights when I arrive home",
        "send email notification",
        "weather temperature alert",
        "post to social media",
        "smart thermostat control",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        # Search triggers
        print("\nTop 3 Triggers:")
        trigger_results = search_triggers(query, top_k=3)
        for i, r in enumerate(trigger_results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.service_name}")
            print(f"     {r.description[:80]}...")
            print(f"     Category: {r.category}")

        # Search actions
        print("\nTop 3 Actions:")
        action_results = search_actions(query, top_k=3)
        for i, r in enumerate(action_results, 1):
            print(f"  {i}. [{r.score:.3f}] {r.service_name}")
            print(f"     {r.description[:80]}...")
            print(f"     Category: {r.category}")

        print()


if __name__ == "__main__":
    test_retrieval()

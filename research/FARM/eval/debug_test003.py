#!/usr/bin/env python3
"""
Debug script for test_003: SMS2Email
Shows RAG candidates and runs selector with LLM re-ranking.

Usage:
    python -m eval.debug_test003           # Show RAG candidates only
    python -m eval.debug_test003 --run     # Run full selector
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def show_rag_candidates(query: str, expected_trigger: str, expected_action: str):
    """Show what RAG retrieves (Stage 1)."""
    from rag.retriever import search_triggers, search_actions

    print("\n" + "=" * 70)
    print("STAGE 1: RAG RETRIEVAL")
    print("=" * 70)

    # Get trigger candidates
    print("\n📍 TRIGGER CANDIDATES (top 5):")
    trigger_candidates = search_triggers(query, top_k=5)
    for i, t in enumerate(trigger_candidates):
        name = t.service_name
        score = t.score
        match = "✓ MATCH" if expected_trigger.lower() in name.lower() or name.lower() in expected_trigger.lower() else ""
        print(f"  T{i}: {name} (score={score:.3f}) {match}")

    # Check if expected trigger is in candidates
    trigger_found = any(
        expected_trigger.lower() in t.service_name.lower() or
        t.service_name.lower() in expected_trigger.lower()
        for t in trigger_candidates
    )
    print(f"\n  Expected trigger in top-5? {'✓ YES' if trigger_found else '✗ NO - RAG FAILURE'}")

    # Get action candidates
    print("\n🎯 ACTION CANDIDATES (top 5):")
    action_candidates = search_actions(query, top_k=5)
    for i, a in enumerate(action_candidates):
        name = a.service_name
        score = a.score
        match = "✓ MATCH" if expected_action.lower() in name.lower() or name.lower() in expected_action.lower() else ""
        print(f"  A{i}: {name} (score={score:.3f}) {match}")

    # Check if expected action is in candidates
    action_found = any(
        expected_action.lower() in a.service_name.lower() or
        a.service_name.lower() in expected_action.lower()
        for a in action_candidates
    )
    print(f"\n  Expected action in top-5? {'✓ YES' if action_found else '✗ NO - RAG FAILURE'}")

    return trigger_found, action_found


def run_selector(query: str, expected_trigger: str, expected_action: str):
    """Run full selector with LLM re-ranking (Stage 2)."""
    from agents.graph import run_selector_negotiation
    from eval.ragas_metrics import service_matches

    print("\n" + "=" * 70)
    print("STAGE 2: LLM RE-RANKING + SELECTION")
    print("=" * 70)

    result = run_selector_negotiation(
        query=query,
        verbose=True,
        show_applet=False,
    )

    if result:
        final_applet = result.get("final_applet", {})
        trigger = final_applet.get("trigger", {})
        action = final_applet.get("action", {})

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Selected Trigger: {trigger.get('service_name', 'N/A')}")
        print(f"Selected Action: {action.get('service_name', 'N/A')}")

        # Check match using word overlap (same as evaluation)
        trigger_match = service_matches(
            trigger.get('service_name', ''),
            expected_trigger
        )
        action_match = service_matches(
            action.get('service_name', ''),
            expected_action
        )

        print(f"\nTrigger Match: {'✓' if trigger_match else '✗'}")
        print(f"Action Match: {'✓' if action_match else '✗'}")

        return trigger_match, action_match
    else:
        print("ERROR: No result returned")
        return False, False


def main():
    parser = argparse.ArgumentParser(description="Debug test_003: SMS2Email")
    parser.add_argument("--run", action="store_true", help="Run full selector (not just RAG)")
    args = parser.parse_args()

    query = "SMS2Email: Forward your text messages to your email"
    expected_trigger = "Any new SMS received"
    expected_action = "Send an email"

    print("=" * 70)
    print("DEBUG: test_003 - SMS2Email")
    print("=" * 70)
    print(f"\nQuery: {query}")
    print(f"Expected Trigger: {expected_trigger}")
    print(f"Expected Action: {expected_action}")

    if args.run:
        # Run full selector (includes RAG + LLM re-ranking)
        run_selector(query, expected_trigger, expected_action)
    else:
        # Just show RAG candidates
        trigger_found, action_found = show_rag_candidates(query, expected_trigger, expected_action)

        print("\n" + "=" * 70)
        print("DIAGNOSIS")
        print("=" * 70)
        if not trigger_found:
            print("⚠️  RAG retrieval failed - correct trigger not in top-5")
            print("   → Need to improve trigger embeddings")
        elif trigger_found:
            print("✓ Correct trigger is in top-5 (RAG working)")
            print("  → Run with --run to see if LLM re-ranker picks it")

        print("\nTo run full selector with LLM re-ranking:")
        print("  python -m eval.debug_test003 --run")


if __name__ == "__main__":
    main()

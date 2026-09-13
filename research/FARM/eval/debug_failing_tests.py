#!/usr/bin/env python3
"""
Debug script for failing tests in evaluation.
Shows what's expected vs what's being selected.

Usage:
    python -m eval.debug_failing_tests           # Show all failing tests
    python -m eval.debug_failing_tests --test 0  # Debug specific test index
"""

import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_test(test_data: dict, idx: int, run_selector: bool = False):
    """Analyze a single test case."""
    from eval.ragas_metrics import service_matches, _embedding_similarity, word_overlap_score

    query = test_data.get("query", test_data.get("description", ""))
    trigger = test_data.get("trigger", {})
    action = test_data.get("action", {})

    expected_trigger = trigger.get("service_name", "N/A")
    expected_action = action.get("service_name", "N/A")
    expected_action_desc = action.get("description", "N/A")

    print(f"\n{'='*70}")
    print(f"TEST {idx:03d}")
    print("=" * 70)
    print(f"Query: {query}")
    print(f"\nExpected Trigger: {expected_trigger}")
    print(f"Expected Action: {expected_action}")
    print(f"Expected Action Desc: {expected_action_desc}")

    if run_selector:
        from agents.graph import run_selector_negotiation

        print(f"\n{'-'*70}")
        print("Running selector...")
        print("-" * 70)

        result = run_selector_negotiation(
            query=query,
            verbose=True,
            show_applet=False,
        )

        if result:
            final_applet = result.get("final_applet", {})
            sel_trigger = final_applet.get("trigger", {})
            sel_action = final_applet.get("action", {})

            selected_trigger = sel_trigger.get('service_name', 'N/A')
            selected_action = sel_action.get('service_name', 'N/A')

            trigger_match = service_matches(selected_trigger, expected_trigger)
            action_match = service_matches(selected_action, expected_action)

            print(f"\n{'-'*70}")
            print("RESULT")
            print("-" * 70)
            print(f"Selected Trigger: {selected_trigger}")
            print(f"Expected Trigger: {expected_trigger}")
            print(f"Trigger Match: {'✓' if trigger_match else '✗'}")
            print()
            print(f"Selected Action: {selected_action}")
            print(f"Expected Action: {expected_action}")
            print(f"Action Match: {'✓' if action_match else '✗'}")

            # Show similarity analysis for action mismatch
            if not action_match:
                print(f"\n{'-'*70}")
                print("ACTION MISMATCH ANALYSIS")
                print("-" * 70)

                # Get selected action description
                selected_action_desc = sel_action.get('description', 'N/A')
                print(f"Selected Action Desc: {selected_action_desc}")
                print(f"Expected Action Desc: {expected_action_desc}")

                # Calculate similarities
                embed_sim = _embedding_similarity(selected_action.lower(), expected_action.lower())
                word_sim = word_overlap_score(selected_action.lower(), expected_action.lower())
                print(f"\nSimilarity Scores:")
                print(f"  Embedding similarity: {embed_sim:.3f} (threshold: 0.5)")
                print(f"  Word overlap score:   {word_sim:.3f} (threshold: 0.6)")

                # Show action candidates from RAG
                action_candidates = result.get("action_candidates", [])
                if action_candidates:
                    print(f"\nRAG Action Candidates (top 5):")
                    for i, cand in enumerate(action_candidates[:5]):
                        cand_name = cand.get("service_name", "N/A")
                        cand_score = cand.get("score", 0)
                        cand_desc = cand.get("description", "")[:50]
                        marker = "← SELECTED" if cand_name == selected_action else ""
                        marker = "← EXPECTED" if cand_name == expected_action else marker
                        print(f"  {i+1}. [{cand_score:.3f}] {cand_name} {marker}")
                        print(f"       Desc: {cand_desc}...")

            return trigger_match, action_match
        else:
            print("ERROR: No result")
            return False, False

    return None, None


def main():
    parser = argparse.ArgumentParser(description="Debug failing tests")
    parser.add_argument("--test", type=int, help="Specific test index to debug")
    parser.add_argument("--run", action="store_true", help="Run selector (not just show expected)")
    parser.add_argument("--limit", type=int, default=5, help="Number of tests to analyze")
    args = parser.parse_args()

    # Load test data
    gold_path = Path(__file__).parent.parent / "data" / "test" / "gold.json"
    with open(gold_path) as f:
        data = json.load(f)

    if args.test is not None:
        # Debug specific test
        if args.test < len(data):
            analyze_test(data[args.test], args.test, run_selector=args.run)
        else:
            print(f"Error: Test index {args.test} out of range (max: {len(data)-1})")
    else:
        # Show first N tests
        print(f"Showing first {args.limit} tests:")
        for i in range(min(args.limit, len(data))):
            analyze_test(data[i], i, run_selector=args.run)


if __name__ == "__main__":
    main()

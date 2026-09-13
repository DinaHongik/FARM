#!/usr/bin/env python3
"""
Selector Debugger
=================

Debug why the selector isn't picking the correct trigger-action pairs.
Shows RAG scores, pair rankings, and what the system actually selects.

Usage:
    # Just analyze RAG scores and pair ranking (fast, no LLM)
    python -m eval.debug_selector --test-idx 0

    # Run full selector with LLM (slow but complete)
    python -m eval.debug_selector --test-idx 0 --run-selector

    # Analyze multiple tests
    python -m eval.debug_selector --all --limit 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_test_data(split: str = "gold") -> list:
    """Load test data."""
    path = Path(__file__).parent.parent / "data" / "test" / f"{split}.json"
    with open(path) as f:
        return json.load(f)


def analyze_rag_scores(query: str, expected_trigger: str, expected_action: str):
    """Analyze RAG retrieval scores without running selector."""
    try:
        from rag.retriever import FARMRetriever
        retriever = FARMRetriever()
    except ImportError as e:
        print(f"\n  RAG not available: {e}")
        return None, None, {}

    triggers = retriever.search_triggers(query, top_k=5)
    actions = retriever.search_actions(query, top_k=5)

    print("\n" + "=" * 70)
    print("STEP 1: RAG RETRIEVAL")
    print("=" * 70)

    # Analyze triggers
    trigger_pos = -1
    print("\nTRIGGER CANDIDATES:")
    for i, t in enumerate(triggers):
        is_correct = t.service_name == expected_trigger
        if is_correct:
            trigger_pos = i
        mark = "✓" if is_correct else " "
        print(f"  {mark} [{i}] {t.service_name[:45]:<45} score={t.score:.4f}")

    # Analyze actions
    action_pos = -1
    print("\nACTION CANDIDATES:")
    for i, a in enumerate(actions):
        is_correct = a.service_name == expected_action
        if is_correct:
            action_pos = i
        mark = "✓" if is_correct else " "
        print(f"  {mark} [{i}] {a.service_name[:45]:<45} score={a.score:.4f}")

    # Summary
    print("\nRETRIEVAL SUMMARY:")
    issues = {}
    if trigger_pos >= 0:
        print(f"  ✓ Correct trigger at rank {trigger_pos} (score: {triggers[trigger_pos].score:.4f})")
        issues["trigger_in_top5"] = True
    else:
        print(f"  ✗ Correct trigger NOT in top-5!")
        issues["trigger_in_top5"] = False

    if action_pos >= 0:
        print(f"  ✓ Correct action at rank {action_pos} (score: {actions[action_pos].score:.4f})")
        issues["action_in_top5"] = True
    else:
        print(f"  ✗ Correct action NOT in top-5!")
        issues["action_in_top5"] = False

    return triggers, actions, issues


def analyze_pair_ranking(triggers, actions, expected_trigger: str, expected_action: str):
    """Analyze pair ranking based on combined RAG scores."""
    if not triggers or not actions:
        return None, {}

    print("\n" + "=" * 70)
    print("STEP 2: PAIR RANKING (trigger_score + action_score)")
    print("=" * 70)

    # Compute all pairs
    pairs = []
    for t_idx, t in enumerate(triggers):
        for a_idx, a in enumerate(actions):
            combined = t.score + a.score
            t_correct = t.service_name == expected_trigger
            a_correct = a.service_name == expected_action
            pairs.append({
                "t_idx": t_idx,
                "a_idx": a_idx,
                "t_name": t.service_name,
                "a_name": a.service_name,
                "t_score": t.score,
                "a_score": a.score,
                "combined": combined,
                "t_correct": t_correct,
                "a_correct": a_correct,
                "is_correct_pair": t_correct and a_correct,
            })

    # Sort by combined score (this is what selector does)
    pairs.sort(key=lambda x: x["combined"], reverse=True)

    # Find correct pair position
    correct_pair_pos = -1
    for i, p in enumerate(pairs):
        if p["is_correct_pair"]:
            correct_pair_pos = i
            break

    print(f"\nTop 10 pairs (of {len(pairs)}):")
    for i, p in enumerate(pairs[:10]):
        if p["is_correct_pair"]:
            marker = "✓✓"
        elif p["t_correct"]:
            marker = "T "
        elif p["a_correct"]:
            marker = " A"
        else:
            marker = "  "
        print(f"  {marker} [{i:2}] T{p['t_idx']}+A{p['a_idx']} = {p['combined']:.4f}  "
              f"({p['t_name'][:22]:<22} + {p['a_name'][:22]})")

    # Analysis
    print("\nPAIR RANKING ANALYSIS:")
    issues = {}

    if correct_pair_pos == 0:
        print(f"  ✓ Correct pair is RANK 0 - will be selected!")
        issues["correct_pair_rank"] = 0
        issues["should_succeed"] = True
    elif correct_pair_pos > 0:
        cp = pairs[correct_pair_pos]
        gap = pairs[0]["combined"] - cp["combined"]
        print(f"  ⚠ Correct pair is at RANK {correct_pair_pos}")
        print(f"    Score gap from rank 0: {gap:.4f}")
        if gap < 0.05:
            print(f"    → Gap < 0.05: LLM TIEBREAKER will be used!")
            print(f"    → LLM will choose between T{pairs[0]['t_idx']} and T{cp['t_idx']}")
            issues["small_gap"] = True
            issues["tiebreaker_needed"] = True
        else:
            print(f"    → Rank 0 pair has significantly higher score")
        issues["correct_pair_rank"] = correct_pair_pos
        issues["should_succeed"] = gap < 0.05  # Might succeed with tiebreaker
    else:
        print(f"  ✗ Correct pair NOT POSSIBLE (trigger or action missing from top-5)")
        issues["correct_pair_rank"] = -1
        issues["should_succeed"] = False

    return pairs, issues


def run_full_selector(query: str, expected_trigger: str, expected_action: str):
    """Run the actual selector with LLM."""
    print("\n" + "=" * 70)
    print("STEP 3: SELECTOR EXECUTION (with LLM)")
    print("=" * 70)

    try:
        from agents.graph import run_selector_negotiation
        from agents.config import config
        config.verbose = True

        result = run_selector_negotiation(query, verbose=True, show_applet=False)
    except Exception as e:
        print(f"\n  Selector failed: {e}")
        return None

    if not result:
        print("\n  Selector returned no result")
        return None

    # What was selected?
    applet = result.get("final_applet", {})
    selected_trigger = (applet.get("trigger") or {}).get("service_name", "?")
    selected_action = (applet.get("action") or {}).get("service_name", "?")

    print("\n" + "-" * 40)
    print("SELECTION RESULT:")
    print("-" * 40)
    t_match = selected_trigger == expected_trigger
    a_match = selected_action == expected_action

    print(f"  Selected trigger: {selected_trigger}")
    print(f"  Expected trigger: {expected_trigger}")
    print(f"  Match: {'✓' if t_match else '✗'}")
    print()
    print(f"  Selected action:  {selected_action}")
    print(f"  Expected action:  {expected_action}")
    print(f"  Match: {'✓' if a_match else '✗'}")
    print()
    print(f"  Negotiation rounds: {result.get('pair_attempt', 0) + 1}")
    print(f"  Tried pairs: {result.get('tried_pairs', [])}")
    print(f"  Verifier score: {result.get('verifier_score', 'N/A')}")

    return {
        "selected_trigger": selected_trigger,
        "selected_action": selected_action,
        "trigger_correct": t_match,
        "action_correct": a_match,
        "rounds": result.get('pair_attempt', 0) + 1,
    }


def debug_test(test_case: dict, idx: int, run_selector: bool = False):
    """Debug a single test case."""
    query = test_case["query"]
    expected_trigger = test_case["trigger"]["service_name"]
    expected_action = test_case["action"]["service_name"]

    print("\n" + "#" * 70)
    print(f"# TEST {idx}")
    print("#" * 70)
    print(f"\nQuery: {query}")
    print(f"Expected: {expected_trigger} -> {expected_action}")

    # Step 1: Analyze RAG
    triggers, actions, rag_issues = analyze_rag_scores(query, expected_trigger, expected_action)

    # Step 2: Analyze pair ranking
    pairs, pair_issues = analyze_pair_ranking(triggers, actions, expected_trigger, expected_action)

    # Step 3: Run selector (optional)
    selector_result = None
    if run_selector:
        selector_result = run_full_selector(query, expected_trigger, expected_action)

    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    if not rag_issues.get("trigger_in_top5"):
        print("  ✗ RAG PROBLEM: Correct trigger not retrieved in top-5")
        print("    → Need to improve trigger encoder or retrieval")
    elif not rag_issues.get("action_in_top5"):
        print("  ✗ RAG PROBLEM: Correct action not retrieved in top-5")
        print("    → Need to improve action encoder or retrieval")
    elif pair_issues.get("correct_pair_rank", -1) == 0:
        print("  ✓ SHOULD WORK: Correct pair is rank 0")
        if selector_result and not (selector_result["trigger_correct"] and selector_result["action_correct"]):
            print("    ⚠ But selector still failed - check bindings")
    elif pair_issues.get("correct_pair_rank", -1) > 0:
        rank = pair_issues["correct_pair_rank"]
        if pair_issues.get("tiebreaker_needed"):
            print(f"  🎯 TIEBREAKER CASE: Correct pair at rank {rank}, gap < 0.05")
            print(f"    → LLM tiebreaker will decide between top-2 triggers")
            print(f"    → If LLM picks correctly, this test will PASS")
            if selector_result:
                if selector_result["trigger_correct"] and selector_result["action_correct"]:
                    print(f"    ✓ Tiebreaker SUCCESS!")
                else:
                    print(f"    ✗ Tiebreaker picked wrong option")
        else:
            print(f"  ⚠ RAG RANKING: Correct pair at rank {rank}, not rank 0")
            print(f"    → Wrong pair has significantly higher score (gap >= 0.05)")
            print(f"    → No tiebreaker used - RAG ranking is trusted")
            print(f"    → Improvement: fine-tune RAG embeddings for this pattern")

    return {
        "rag_issues": rag_issues,
        "pair_issues": pair_issues,
        "selector_result": selector_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Debug selector - analyze why it picks wrong pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick analysis (no LLM, just RAG scores + pair ranking)
    python -m eval.debug_selector --test-idx 0

    # Full analysis with selector execution
    python -m eval.debug_selector --test-idx 0 --run-selector

    # Analyze multiple tests
    python -m eval.debug_selector --all --limit 5
        """
    )
    parser.add_argument("--test-idx", type=int, default=0, help="Test index")
    parser.add_argument("--split", type=str, default="gold", help="Test split")
    parser.add_argument("--run-selector", action="store_true", help="Run full selector with LLM")
    parser.add_argument("--all", action="store_true", help="Analyze all tests")
    parser.add_argument("--limit", type=int, default=5, help="Limit for --all")
    args = parser.parse_args()

    test_data = load_test_data(args.split)

    if args.all:
        stats = {"total": 0, "trigger_in_top5": 0, "action_in_top5": 0, "pair_rank_0": 0}

        for i, test_case in enumerate(test_data[:args.limit]):
            result = debug_test(test_case, i, run_selector=args.run_selector)
            stats["total"] += 1
            if result["rag_issues"].get("trigger_in_top5"):
                stats["trigger_in_top5"] += 1
            if result["rag_issues"].get("action_in_top5"):
                stats["action_in_top5"] += 1
            if result["pair_issues"].get("correct_pair_rank", -1) == 0:
                stats["pair_rank_0"] += 1

        print("\n" + "=" * 70)
        print("OVERALL STATISTICS")
        print("=" * 70)
        n = stats["total"]
        print(f"  Total tests:           {n}")
        print(f"  Trigger in top-5:      {stats['trigger_in_top5']}/{n} ({100*stats['trigger_in_top5']/n:.0f}%)")
        print(f"  Action in top-5:       {stats['action_in_top5']}/{n} ({100*stats['action_in_top5']/n:.0f}%)")
        print(f"  Correct pair rank 0:   {stats['pair_rank_0']}/{n} ({100*stats['pair_rank_0']/n:.0f}%)")
    else:
        if args.test_idx >= len(test_data):
            print(f"Error: test-idx {args.test_idx} out of range (max: {len(test_data) - 1})")
            return
        debug_test(test_data[args.test_idx], args.test_idx, run_selector=args.run_selector)


if __name__ == "__main__":
    main()

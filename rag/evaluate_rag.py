"""
RAG Evaluation Script for FARM

Tests the actual RAG pipeline (Qdrant retrieval) comparing:
- BASELINE: pretrained encoder + farm_*_baseline collections
- FINE-TUNED: fine-tuned encoders + farm_*_finetuned collections

Computes retrieval metrics:
- Recall@K (K=1, 3, 5, 10)
- MRR@3 (Mean Reciprocal Rank)

Usage:
  python -m rag.evaluate_rag
  python -m rag.evaluate_rag --baseline   # Only baseline
  python -m rag.evaluate_rag --finetuned  # Only fine-tuned
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# Handle imports
try:
    from .config import config, get_baseline_config, get_finetuned_config
    from .retriever import FARMRetriever
    from .indexer import extract_trigger_text, extract_action_text
except ImportError:
    from config import config, get_baseline_config, get_finetuned_config
    from retriever import FARMRetriever
    from indexer import extract_trigger_text, extract_action_text


@dataclass
class RAGEvalResult:
    """RAG evaluation result."""
    mode: str  # "baseline" or "finetuned"
    dataset: str  # "trigger" or "action"
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_3: float
    num_queries: int
    avg_score: float

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "dataset": self.dataset,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr@3": round(self.mrr_at_3, 4),
            "num_queries": self.num_queries,
            "avg_score": round(self.avg_score, 4),
        }


def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """Compute Recall@K."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def compute_mrr_at_k(ranks: List[int], k: int = 3) -> float:
    """Compute Mean Reciprocal Rank @ K."""
    if not ranks:
        return 0.0
    rr_sum = 0.0
    for r in ranks:
        if r <= k:
            rr_sum += 1.0 / r
    return rr_sum / len(ranks)


def load_triggers() -> List[Dict]:
    """Load trigger data."""
    with open(config.data.triggers_path, 'r') as f:
        return json.load(f)


def load_actions() -> List[Dict]:
    """Load action data."""
    with open(config.data.actions_path, 'r') as f:
        return json.load(f)


def evaluate_rag_triggers(
    retriever: FARMRetriever,
    triggers: List[Dict],
    mode: str,
    top_k: int = 10
) -> RAGEvalResult:
    """
    Evaluate RAG retrieval for triggers.

    For each trigger, query with its description and check if the
    correct trigger (same service_name) is in the top-K results.
    """
    print(f"\nEvaluating {mode.upper()} RAG on triggers...")

    ranks = []
    scores = []

    for trigger in tqdm(triggers, desc=f"Triggers ({mode})"):
        description = trigger.get("description", "")
        if not description:
            continue

        target_service = trigger.get("service_name", "")

        # Search using RAG
        results = retriever.search_triggers(description, top_k=top_k)

        # Find rank of correct result
        found = False
        for rank, result in enumerate(results, 1):
            if result.service_name == target_service:
                ranks.append(rank)
                scores.append(result.score)
                found = True
                break

        if not found:
            ranks.append(top_k + 1)  # Not found in top-K

    return RAGEvalResult(
        mode=mode,
        dataset="trigger",
        recall_at_1=compute_recall_at_k(ranks, 1),
        recall_at_3=compute_recall_at_k(ranks, 3),
        recall_at_5=compute_recall_at_k(ranks, 5),
        recall_at_10=compute_recall_at_k(ranks, 10),
        mrr_at_3=compute_mrr_at_k(ranks, 3),
        num_queries=len(ranks),
        avg_score=sum(scores) / len(scores) if scores else 0.0
    )


def evaluate_rag_actions(
    retriever: FARMRetriever,
    actions: List[Dict],
    mode: str,
    top_k: int = 10
) -> RAGEvalResult:
    """
    Evaluate RAG retrieval for actions.
    """
    print(f"\nEvaluating {mode.upper()} RAG on actions...")

    ranks = []
    scores = []

    for action in tqdm(actions, desc=f"Actions ({mode})"):
        description = action.get("description", "")
        if not description:
            continue

        target_service = action.get("service_name", "")

        # Search using RAG
        results = retriever.search_actions(description, top_k=top_k)

        # Find rank of correct result
        found = False
        for rank, result in enumerate(results, 1):
            if result.service_name == target_service:
                ranks.append(rank)
                scores.append(result.score)
                found = True
                break

        if not found:
            ranks.append(top_k + 1)

    return RAGEvalResult(
        mode=mode,
        dataset="action",
        recall_at_1=compute_recall_at_k(ranks, 1),
        recall_at_3=compute_recall_at_k(ranks, 3),
        recall_at_5=compute_recall_at_k(ranks, 5),
        recall_at_10=compute_recall_at_k(ranks, 10),
        mrr_at_3=compute_mrr_at_k(ranks, 3),
        num_queries=len(ranks),
        avg_score=sum(scores) / len(scores) if scores else 0.0
    )


def print_result(result: RAGEvalResult):
    """Print a single result."""
    print(f"  R@1:  {result.recall_at_1:.4f}")
    print(f"  R@3:  {result.recall_at_3:.4f}")
    print(f"  R@5:  {result.recall_at_5:.4f}")
    print(f"  R@10: {result.recall_at_10:.4f}")
    print(f"  MRR@3: {result.mrr_at_3:.4f}")
    print(f"  Avg Score: {result.avg_score:.4f}")
    print(f"  Queries: {result.num_queries}")


def print_comparison(baseline: RAGEvalResult, finetuned: RAGEvalResult):
    """Print side-by-side comparison."""
    def delta(b, f):
        d = f - b
        return f"+{d:.4f}" if d > 0 else f"{d:.4f}"

    print(f"\n{'Metric':<12} {'Baseline':<12} {'Fine-tuned':<12} {'Delta':<12}")
    print("-" * 48)
    print(f"{'R@1':<12} {baseline.recall_at_1:<12.4f} {finetuned.recall_at_1:<12.4f} {delta(baseline.recall_at_1, finetuned.recall_at_1):<12}")
    print(f"{'R@3':<12} {baseline.recall_at_3:<12.4f} {finetuned.recall_at_3:<12.4f} {delta(baseline.recall_at_3, finetuned.recall_at_3):<12}")
    print(f"{'R@5':<12} {baseline.recall_at_5:<12.4f} {finetuned.recall_at_5:<12.4f} {delta(baseline.recall_at_5, finetuned.recall_at_5):<12}")
    print(f"{'R@10':<12} {baseline.recall_at_10:<12.4f} {finetuned.recall_at_10:<12.4f} {delta(baseline.recall_at_10, finetuned.recall_at_10):<12}")
    print(f"{'MRR@3':<12} {baseline.mrr_at_3:<12.4f} {finetuned.mrr_at_3:<12.4f} {delta(baseline.mrr_at_3, finetuned.mrr_at_3):<12}")
    print(f"{'Avg Score':<12} {baseline.avg_score:<12.4f} {finetuned.avg_score:<12.4f} {delta(baseline.avg_score, finetuned.avg_score):<12}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FARM RAG System - Baseline vs Fine-tuned",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", action="store_true", help="Evaluate baseline only")
    parser.add_argument("--finetuned", action="store_true", help="Evaluate fine-tuned only")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval (default: 10)")
    parser.add_argument("--sample", type=int, default=100, help="Sample size per dataset (default: 100, use 0 for all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    parser.add_argument("--save", type=str, help="Save results to JSON file")

    args = parser.parse_args()

    # Default to both if none specified
    if not args.baseline and not args.finetuned:
        args.baseline = True
        args.finetuned = True

    print("=" * 70)
    print("FARM RAG Evaluation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    triggers = load_triggers()
    actions = load_actions()
    print(f"  Triggers: {len(triggers)}")
    print(f"  Actions: {len(actions)}")

    # Sample data if requested
    if args.sample > 0:
        random.seed(args.seed)
        if len(triggers) > args.sample:
            triggers = random.sample(triggers, args.sample)
        if len(actions) > args.sample:
            actions = random.sample(actions, args.sample)
        print(f"  Sampled: {len(triggers)} triggers, {len(actions)} actions")

    all_results = []
    baseline_trigger = None
    baseline_action = None
    finetuned_trigger = None
    finetuned_action = None

    # Evaluate baseline
    if args.baseline:
        print("\n" + "=" * 70)
        print("BASELINE RAG (pretrained encoder)")
        print("=" * 70)

        baseline_config = get_baseline_config()
        baseline_retriever = FARMRetriever(baseline_config)

        baseline_trigger = evaluate_rag_triggers(
            baseline_retriever, triggers, "baseline", args.top_k
        )
        print("\nTrigger Results:")
        print_result(baseline_trigger)
        all_results.append(baseline_trigger)

        baseline_action = evaluate_rag_actions(
            baseline_retriever, actions, "baseline", args.top_k
        )
        print("\nAction Results:")
        print_result(baseline_action)
        all_results.append(baseline_action)

        baseline_retriever.client.close()

    # Evaluate fine-tuned
    if args.finetuned:
        print("\n" + "=" * 70)
        print("FINE-TUNED RAG (fine-tuned encoders)")
        print("=" * 70)

        finetuned_config = get_finetuned_config()
        finetuned_retriever = FARMRetriever(finetuned_config)

        finetuned_trigger = evaluate_rag_triggers(
            finetuned_retriever, triggers, "finetuned", args.top_k
        )
        print("\nTrigger Results:")
        print_result(finetuned_trigger)
        all_results.append(finetuned_trigger)

        finetuned_action = evaluate_rag_actions(
            finetuned_retriever, actions, "finetuned", args.top_k
        )
        print("\nAction Results:")
        print_result(finetuned_action)
        all_results.append(finetuned_action)

        finetuned_retriever.client.close()

    # Print comparison if both were evaluated
    if args.baseline and args.finetuned:
        print("\n" + "=" * 70)
        print("COMPARISON: BASELINE vs FINE-TUNED")
        print("=" * 70)

        print("\n>>> TRIGGERS <<<")
        print_comparison(baseline_trigger, finetuned_trigger)

        print("\n>>> ACTIONS <<<")
        print_comparison(baseline_action, finetuned_action)

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        trigger_improvement = finetuned_trigger.recall_at_1 - baseline_trigger.recall_at_1
        action_improvement = finetuned_action.recall_at_1 - baseline_action.recall_at_1

        print(f"\nTrigger R@1 improvement: {baseline_trigger.recall_at_1:.4f} -> {finetuned_trigger.recall_at_1:.4f} (+{trigger_improvement:.4f})")
        print(f"Action R@1 improvement:  {baseline_action.recall_at_1:.4f} -> {finetuned_action.recall_at_1:.4f} (+{action_improvement:.4f})")
        print(f"\nFine-tuning improved R@1 by {((trigger_improvement + action_improvement) / 2) * 100:.1f}% on average!")

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nResults saved to: {save_path}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
RAG Evaluation Script for FARM

Evaluates the embedding model + RAG retrieval performance on test sets.
This is SEPARATE from the agentic evaluation (evaluate_e2e.py).

Tests:
- Does RAG retrieve the correct trigger API?
- Does RAG retrieve the correct action API?

Metrics (well-established from TREC, MS MARCO, BEIR):
- Recall@K (R@1, R@3, R@5): Is correct API in top-K?
- MRR@K: Mean Reciprocal Rank
- EM: Exact Match (same as R@1)

Test Sets:
- data/test/gold.json: Clear descriptions (500 samples)
- data/test/noisy.json: Vague descriptions (500 samples)
- data/test/oneshot.json: Rare APIs (200 samples)

Usage:
    python -m eval.evaluate_rag                     # Run all test sets
    python -m eval.evaluate_rag --test-set gold     # Run only gold
    python -m eval.evaluate_rag --baseline          # Use baseline (pretrained) model
    python -m eval.evaluate_rag --finetuned         # Use fine-tuned model
    python -m eval.evaluate_rag --all-models        # Compare both models
"""

import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RAGMetrics:
    """
    RAG retrieval metrics at two granularity levels.

    Service-Level Metrics (R@K, MRR@K):
        Matches by service_name only (e.g., "Gmail", "Slack").
        This is compatible with prior work like TARGE for fair comparison.

    Schema-Level Metrics (Schema_R@K, Schema_MRR@K):
        Matches by service_name AND field-name signature (ingredients/action fields).
        This is a stricter proxy for API-variant correctness because the same service
        can have multiple API variants with different field structures.
        Note: This is field-name matching, not a guaranteed unique API identifier.

        Example - YouTube has different trigger schemas:
          - "New video in channel" -> Ingredients: {channel_id, video_title}
          - "New liked video" -> Ingredients: {video_id, liked_at}

        Service-level would count both as correct if service_name matches.
        Schema-level requires the exact field structure to match.
    """
    # Service-level metrics (TARGE-compatible)
    # Matches service_name only - used for comparison with prior work
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr_at_3: float = 0.0
    mrr_at_5: float = 0.0
    # Schema-level metrics (field-signature match)
    # Matches service_name + field names - stricter proxy for API-variant correctness
    # Note: This is a field-name signature match, not a guaranteed unique API identifier
    schema_recall_at_1: float = 0.0
    schema_recall_at_3: float = 0.0
    schema_recall_at_5: float = 0.0
    schema_mrr_at_3: float = 0.0
    schema_mrr_at_5: float = 0.0  # Added for consistency with service-level
    # Common
    num_samples: int = 0
    avg_score: float = 0.0  # Mean retrieval score at correct hit rank

    def to_dict(self) -> Dict:
        return {
            "R@1 (EM)": round(self.recall_at_1, 4),
            "R@3": round(self.recall_at_3, 4),
            "R@5": round(self.recall_at_5, 4),
            "MRR@3": round(self.mrr_at_3, 4),
            "MRR@5": round(self.mrr_at_5, 4),
            "Schema_R@1": round(self.schema_recall_at_1, 4),
            "Schema_R@3": round(self.schema_recall_at_3, 4),
            "Schema_R@5": round(self.schema_recall_at_5, 4),
            "Schema_MRR@3": round(self.schema_mrr_at_3, 4),
            "Schema_MRR@5": round(self.schema_mrr_at_5, 4),
            "Samples": self.num_samples,
            "Avg_Score": round(self.avg_score, 4),
        }


@dataclass
class TestSetResults:
    """Results for a single test set."""
    test_set: str
    model_type: str  # "baseline" or "finetuned"
    trigger_metrics: RAGMetrics
    action_metrics: RAGMetrics
    # Joint R@1: Both trigger AND action correct at rank 1
    joint_r1: float = 0.0  # Service-level (name only)
    joint_schema_r1: float = 0.0  # Schema-level (name + field signature)

    def to_dict(self) -> Dict:
        return {
            "test_set": self.test_set,
            "model": self.model_type,
            "trigger": self.trigger_metrics.to_dict(),
            "action": self.action_metrics.to_dict(),
            "Joint_R@1": round(self.joint_r1, 4),
            "Joint_Schema_R@1": round(self.joint_schema_r1, 4),
        }


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """Compute Recall@K: fraction where correct is in top-K."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def compute_mrr_at_k(ranks: List[int], k: int) -> float:
    """Compute MRR@K: Mean Reciprocal Rank for top-K."""
    if not ranks:
        return 0.0
    rr_sum = 0.0
    for r in ranks:
        if r <= k:
            rr_sum += 1.0 / r
    return rr_sum / len(ranks)


def normalize_name(name: str) -> str:
    """Normalize service name for comparison."""
    return name.lower().strip()


def get_schema_key(api_info: Dict, is_trigger: bool = True) -> str:
    """
    Extract a unique schema key from api_info for schema-level matching.

    Schema-level matching is more rigorous than service-level matching because
    the same service can have multiple API variants with different field structures.

    For triggers: extracts sorted Ingredients field names
        Example: {"subject": "...", "from": "...", "body": "..."} -> "body|from|subject"

    For actions: extracts sorted Action fields names
        Example: {"channel": "...", "message": "..."} -> "channel|message"

    Why this matters:
        - Gmail trigger "New email" has Ingredients: {from, subject, body}
        - Gmail trigger "New starred email" has Ingredients: {from, subject, starred_at}
        - Service-level treats both as "Gmail" (correct)
        - Schema-level distinguishes them by field structure (more precise)

    Args:
        api_info: API metadata containing Ingredients or Action fields
        is_trigger: True for triggers (use Ingredients), False for actions (use Action fields)

    Returns:
        Pipe-separated sorted field names as schema key (e.g., "body|from|subject")
    """
    if is_trigger:
        # Triggers use "Ingredients" - the data fields provided by the trigger
        ingredients = api_info.get("Ingredients", {})
        keys = sorted(ingredients.keys()) if isinstance(ingredients, dict) else []
    else:
        # Actions use "Action fields" - the parameters required by the action
        fields = api_info.get("Action fields", {})
        keys = sorted(fields.keys()) if isinstance(fields, dict) else []
    return "|".join(keys).lower()


def find_rank(target: str, results: List[Any], max_k: int = 10) -> int:
    """Find rank of target in results (1-indexed). Returns max_k+1 if not found.

    NOTE: This matches service_name ONLY (for TARGE-compatible comparison).
    """
    target_norm = normalize_name(target)
    for i, result in enumerate(results[:max_k]):
        if normalize_name(result.service_name) == target_norm:
            return i + 1
    return max_k + 1


def find_rank_schema(target_name: str, target_api_info: Dict,
                     results: List[Any], is_trigger: bool = True,
                     max_k: int = 10) -> int:
    """
    Find rank matching BOTH service_name AND schema (ingredients/fields).

    Schema-level matching requires two conditions:
        1. service_name must match (same as service-level)
        2. Field structure must match (schema key comparison)

    This is more rigorous than service_name-only matching because
    the same service can have multiple API variants:

        Example - YouTube triggers:
          - "New video in channel" -> schema: "channel_id|video_title"
          - "New liked video" -> schema: "liked_at|video_id"

        If ground truth is "New video in channel" but RAG retrieves
        "New liked video", service-level counts it correct (both are YouTube),
        but schema-level counts it wrong (different field structures).

    This matters for downstream binding generation - if we retrieve the
    wrong API variant, the LLM will generate incorrect field bindings.

    Args:
        target_name: Ground truth service name to find
        target_api_info: Ground truth API metadata with Ingredients/Action fields
        results: Retrieved results from RAG (must have .service_name and .api_info)
        is_trigger: True for triggers, False for actions
        max_k: Maximum rank to search

    Returns:
        1-indexed rank if found with matching schema, max_k+1 if not found
    """
    target_norm = normalize_name(target_name)
    target_schema = get_schema_key(target_api_info, is_trigger)

    for i, result in enumerate(results[:max_k]):
        # First check: service_name must match
        if normalize_name(result.service_name) == target_norm:
            # Second check: schema (field structure) must also match
            result_schema = get_schema_key(result.api_info, is_trigger)
            if result_schema == target_schema:
                return i + 1
    return max_k + 1


# =============================================================================
# RAG EVALUATOR
# =============================================================================

class RAGEvaluator:
    """Evaluates RAG retrieval on test sets."""

    def __init__(self, use_finetuned: bool = True, verbose: bool = False):
        """
        Initialize evaluator.

        Args:
            use_finetuned: Use fine-tuned model (True) or baseline (False)
            verbose: Print detailed output
        """
        self.use_finetuned = use_finetuned
        self.verbose = verbose
        self.model_type = "finetuned" if use_finetuned else "baseline"
        self.retriever = None

    def _init_retriever(self):
        """Initialize RAG retriever."""
        if self.retriever is not None:
            return

        from rag.retriever import FARMRetriever
        from rag.config import get_baseline_config, get_finetuned_config

        if self.use_finetuned:
            config = get_finetuned_config()
            print(f"Using FINE-TUNED model")
        else:
            config = get_baseline_config()
            print(f"Using BASELINE (pretrained) model")

        self.retriever = FARMRetriever(config)

    def load_test_set(self, name: str) -> List[Dict]:
        """Load test set by name (gold, noisy, oneshot)."""
        test_path = Path(__file__).parent.parent / "data" / "test" / f"{name}.json"
        if not test_path.exists():
            raise FileNotFoundError(f"Test set not found: {test_path}")

        with open(test_path, 'r') as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples from {name}.json")
        return data

    def evaluate_test_set(self, test_name: str, top_k: int = 10) -> TestSetResults:
        """
        Evaluate RAG on a test set.

        Args:
            test_name: Name of test set (gold, noisy, oneshot)
            top_k: Maximum K for retrieval

        Returns:
            TestSetResults with metrics
        """
        self._init_retriever()
        test_data = self.load_test_set(test_name)

        # Service-level ranks (TARGE-compatible)
        trigger_ranks = []
        action_ranks = []
        # Schema-level ranks (field-signature match)
        trigger_schema_ranks = []
        action_schema_ranks = []
        # Scores at correct hit rank
        trigger_scores = []
        action_scores = []
        # Joint R@1 counters
        both_correct = 0
        both_correct_schema = 0

        desc = f"Evaluating {test_name} ({self.model_type})"
        for sample in tqdm(test_data, desc=desc):
            query = sample["query"]
            gt_trigger = sample["trigger"]["service_name"]
            gt_trigger_api = sample["trigger"].get("api_info", {})
            gt_action = sample["action"]["service_name"]
            gt_action_api = sample["action"].get("api_info", {})

            # Search triggers
            trigger_results = self.retriever.search_triggers(query, top_k=top_k)
            # Service-level match
            trigger_rank = find_rank(gt_trigger, trigger_results, top_k)
            trigger_ranks.append(trigger_rank)
            # Schema-level match (service_name + ingredients)
            trigger_schema_rank = find_rank_schema(
                gt_trigger, gt_trigger_api, trigger_results,
                is_trigger=True, max_k=top_k
            )
            trigger_schema_ranks.append(trigger_schema_rank)

            if trigger_results and trigger_rank <= top_k:
                trigger_scores.append(trigger_results[trigger_rank - 1].score)

            # Search actions
            action_results = self.retriever.search_actions(query, top_k=top_k)
            # Service-level match
            action_rank = find_rank(gt_action, action_results, top_k)
            action_ranks.append(action_rank)
            # Schema-level match (service_name + fields)
            action_schema_rank = find_rank_schema(
                gt_action, gt_action_api, action_results,
                is_trigger=False, max_k=top_k
            )
            action_schema_ranks.append(action_schema_rank)

            if action_results and action_rank <= top_k:
                action_scores.append(action_results[action_rank - 1].score)

            # Joint R@1 (both trigger and action correct at rank 1)
            if trigger_rank == 1 and action_rank == 1:
                both_correct += 1
            if trigger_schema_rank == 1 and action_schema_rank == 1:
                both_correct_schema += 1

            if self.verbose:
                print(f"\nQuery: {query[:60]}...")
                print(f"  Trigger: {gt_trigger} -> Rank {trigger_rank} (schema: {trigger_schema_rank})")
                print(f"  Action: {gt_action} -> Rank {action_rank} (schema: {action_schema_rank})")

        # Compute metrics
        trigger_metrics = RAGMetrics(
            # Service-level (TARGE-compatible)
            recall_at_1=compute_recall_at_k(trigger_ranks, 1),
            recall_at_3=compute_recall_at_k(trigger_ranks, 3),
            recall_at_5=compute_recall_at_k(trigger_ranks, 5),
            mrr_at_3=compute_mrr_at_k(trigger_ranks, 3),
            mrr_at_5=compute_mrr_at_k(trigger_ranks, 5),
            # Schema-level (field-signature match)
            schema_recall_at_1=compute_recall_at_k(trigger_schema_ranks, 1),
            schema_recall_at_3=compute_recall_at_k(trigger_schema_ranks, 3),
            schema_recall_at_5=compute_recall_at_k(trigger_schema_ranks, 5),
            schema_mrr_at_3=compute_mrr_at_k(trigger_schema_ranks, 3),
            schema_mrr_at_5=compute_mrr_at_k(trigger_schema_ranks, 5),
            num_samples=len(trigger_ranks),
            avg_score=sum(trigger_scores) / len(trigger_scores) if trigger_scores else 0.0,
        )

        action_metrics = RAGMetrics(
            # Service-level (TARGE-compatible)
            recall_at_1=compute_recall_at_k(action_ranks, 1),
            recall_at_3=compute_recall_at_k(action_ranks, 3),
            recall_at_5=compute_recall_at_k(action_ranks, 5),
            mrr_at_3=compute_mrr_at_k(action_ranks, 3),
            mrr_at_5=compute_mrr_at_k(action_ranks, 5),
            # Schema-level (field-signature match)
            schema_recall_at_1=compute_recall_at_k(action_schema_ranks, 1),
            schema_recall_at_3=compute_recall_at_k(action_schema_ranks, 3),
            schema_recall_at_5=compute_recall_at_k(action_schema_ranks, 5),
            schema_mrr_at_3=compute_mrr_at_k(action_schema_ranks, 3),
            schema_mrr_at_5=compute_mrr_at_k(action_schema_ranks, 5),
            num_samples=len(action_ranks),
            avg_score=sum(action_scores) / len(action_scores) if action_scores else 0.0,
        )

        joint_r1 = both_correct / len(test_data) if test_data else 0.0
        joint_schema_r1 = both_correct_schema / len(test_data) if test_data else 0.0

        return TestSetResults(
            test_set=test_name,
            model_type=self.model_type,
            trigger_metrics=trigger_metrics,
            action_metrics=action_metrics,
            joint_r1=joint_r1,
            joint_schema_r1=joint_schema_r1,
        )

    def close(self):
        """Close retriever connection."""
        if self.retriever:
            self.retriever.client.close()


# =============================================================================
# PRINTING RESULTS
# =============================================================================

def print_metrics(metrics: RAGMetrics, title: str):
    """Print metrics in a formatted way."""
    print(f"\n  {title}:")
    print(f"    --- Service-Level (TARGE-compatible) ---")
    print(f"    R@1 (EM):  {metrics.recall_at_1:.4f}")
    print(f"    R@3:       {metrics.recall_at_3:.4f}")
    print(f"    R@5:       {metrics.recall_at_5:.4f}")
    print(f"    MRR@3:     {metrics.mrr_at_3:.4f}")
    print(f"    MRR@5:     {metrics.mrr_at_5:.4f}")
    print(f"    --- Schema-Level (Field-Signature Match) ---")
    print(f"    Schema R@1:   {metrics.schema_recall_at_1:.4f}")
    print(f"    Schema R@3:   {metrics.schema_recall_at_3:.4f}")
    print(f"    Schema R@5:   {metrics.schema_recall_at_5:.4f}")
    print(f"    Schema MRR@3: {metrics.schema_mrr_at_3:.4f}")
    print(f"    Schema MRR@5: {metrics.schema_mrr_at_5:.4f}")


def print_results(results: TestSetResults):
    """Print full test set results."""
    print(f"\n{'='*60}")
    print(f"TEST SET: {results.test_set.upper()} | MODEL: {results.model_type.upper()}")
    print(f"{'='*60}")

    print_metrics(results.trigger_metrics, "TRIGGER Retrieval")
    print_metrics(results.action_metrics, "ACTION Retrieval")

    print(f"\n  JOINT R@1 (Both trigger AND action correct at rank 1):")
    print(f"    Service-Level: {results.joint_r1:.4f}")
    print(f"    Schema-Level:  {results.joint_schema_r1:.4f}")


def print_comparison_table(all_results: List[TestSetResults]):
    """Print comparison table across test sets and models."""
    print(f"\n{'='*100}")
    print("COMPARISON TABLE - SERVICE LEVEL (TARGE-compatible)")
    print(f"{'='*100}")

    # Service-level header
    print(f"\n{'Test Set':<12} {'Model':<12} {'Trig R@1':<10} {'Trig MRR@3':<12} {'Act R@1':<10} {'Act MRR@3':<12} {'Joint R@1':<10}")
    print("-" * 80)

    for r in all_results:
        print(f"{r.test_set:<12} {r.model_type:<12} "
              f"{r.trigger_metrics.recall_at_1:<10.4f} {r.trigger_metrics.mrr_at_3:<12.4f} "
              f"{r.action_metrics.recall_at_1:<10.4f} {r.action_metrics.mrr_at_3:<12.4f} "
              f"{r.joint_r1:<10.4f}")

    # Schema-level table
    print(f"\n{'='*100}")
    print("COMPARISON TABLE - SCHEMA LEVEL (Field-Signature Match)")
    print(f"{'='*100}")

    print(f"\n{'Test Set':<12} {'Model':<12} {'Trig R@1':<10} {'Trig MRR@3':<12} {'Act R@1':<10} {'Act MRR@3':<12} {'Joint R@1':<10}")
    print("-" * 80)

    for r in all_results:
        print(f"{r.test_set:<12} {r.model_type:<12} "
              f"{r.trigger_metrics.schema_recall_at_1:<10.4f} {r.trigger_metrics.schema_mrr_at_3:<12.4f} "
              f"{r.action_metrics.schema_recall_at_1:<10.4f} {r.action_metrics.schema_mrr_at_3:<12.4f} "
              f"{r.joint_schema_r1:<10.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FARM RAG on test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test-set", type=str, default="all",
        choices=["gold", "noisy", "oneshot", "all"],
        help="Which test set to evaluate (default: all)"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Use baseline (pretrained) model"
    )
    parser.add_argument(
        "--finetuned", action="store_true",
        help="Use fine-tuned model"
    )
    parser.add_argument(
        "--all-models", action="store_true",
        help="Compare both baseline and fine-tuned"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Top-K for retrieval (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--save", type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Determine which models to test
    if args.all_models:
        model_configs = [False, True]  # baseline, then finetuned
    elif args.baseline:
        model_configs = [False]
    elif args.finetuned:
        model_configs = [True]
    else:
        model_configs = [True]  # Default to finetuned

    # Determine which test sets
    if args.test_set == "all":
        test_sets = ["gold", "noisy", "oneshot"]
    else:
        test_sets = [args.test_set]

    print("=" * 60)
    print("FARM RAG EVALUATION")
    print("=" * 60)
    print(f"Test sets: {test_sets}")
    print(f"Models: {['baseline' if not m else 'finetuned' for m in model_configs]}")
    print(f"Top-K: {args.top_k}")

    all_results = []

    for use_finetuned in model_configs:
        evaluator = RAGEvaluator(use_finetuned=use_finetuned, verbose=args.verbose)

        for test_name in test_sets:
            try:
                results = evaluator.evaluate_test_set(test_name, top_k=args.top_k)
                all_results.append(results)
                print_results(results)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
            except Exception as e:
                print(f"Error evaluating {test_name}: {e}")
                raise

        evaluator.close()

    # Print comparison table
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nResults saved to: {save_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()

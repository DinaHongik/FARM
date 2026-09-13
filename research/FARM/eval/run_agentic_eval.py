#!/usr/bin/env python3
"""
FARM Agentic AI Evaluation Script
==================================

Runs end-to-end evaluation of the WoT multi-agent system using
RAGAS-style metrics across gold/noise/oneshot test sets.

Usage:
    python -m eval.run_agentic_eval
    python -m eval.run_agentic_eval --splits gold,noisy
    python -m eval.run_agentic_eval --limit 10 --verbose
    python -m eval.run_agentic_eval --save results/eval_results.json

Metrics (RAGAS-based):
    - Goal Accuracy: AgentGoalAccuracyWithReference
    - Binding F1: ToolCallF1
    - Faithfulness: Reasoning grounded in context
    - Context Recall: RAG retrieval quality
    - Topic Adherence: Stays on WoT domain
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import asdict
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.ragas_metrics import (
    RAGASMetrics,
    EvaluationResult,
    evaluate_single_sample,
    evaluate_dataset,
    save_results_json,
    print_results,
    get_ragas_llm,
    RAGAS_AVAILABLE,
)


class AgenticEvaluator:
    """
    End-to-end evaluator for FARM Agentic System.

    Uses RAGAS-style metrics for comprehensive evaluation.
    """

    def __init__(
        self,
        verbose: bool = False,
        use_planner: bool = True,
        use_mock: bool = False,
        use_selector: bool = False,
        use_reference: bool = False,
        save_interval: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Initialize evaluator.

        Args:
            verbose: Print detailed progress
            use_planner: Use full agentic mode with planner
            use_mock: Use mock data (for testing evaluator itself)
            use_selector: Use NEW multi-candidate selector design
            use_reference: Use reference data for candidates (no RAG needed)
            save_interval: Save intermediate results every N samples (e.g., 5)
            save_path: Path for intermediate saves (required if save_interval set)
        """
        self.verbose = verbose
        self.use_planner = use_planner
        self.use_mock = use_mock
        self.use_selector = use_selector
        self.use_reference = use_reference
        self.save_interval = save_interval
        self.save_path = save_path
        self._test_data = None  # Stored for reference mode

        # Initialize RAGAS LLM once for consistent metrics
        self._ragas_llm = None
        self._use_ragas = False
        if RAGAS_AVAILABLE:
            try:
                self._ragas_llm = get_ragas_llm()
                self._use_ragas = True
                if verbose:
                    print("RAGAS initialized with granite4:small-h")
            except Exception as e:
                print(f"Warning: Could not initialize RAGAS: {e}")

    def load_test_set(self, path: str) -> List[Dict]:
        """Load test set from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def run_agent(self, query: str) -> Optional[Dict]:
        """
        Run the FARM agentic system on a query.

        Args:
            query: User query

        Returns:
            Agent output dict or None on failure
        """
        try:
            from agents.graph import (
                run_agentic_negotiation,
                run_negotiation,
                run_selector_negotiation,
                run_selector_with_reference,
            )

            if self.use_reference and self._test_data:
                # Reference mode: use test data for candidates (no RAG needed)
                result = run_selector_with_reference(
                    query=query,
                    test_data=self._test_data,
                    verbose=False,
                )
            elif self.use_mock:
                # Use mock mode for testing
                result = run_negotiation(
                    query=query,
                    use_mock=True,
                    use_simple=True,
                    use_memory=False,
                    verbose=False,
                )
            elif self.use_selector:
                # NEW DESIGN: Multi-candidate scoring with verifier-driven fallback
                result = run_selector_negotiation(
                    query=query,
                    verbose=self.verbose,
                    show_applet=False,
                )
            else:
                # Original agentic mode
                result = run_agentic_negotiation(
                    query=query,
                    verbose=False,
                    show_applet=False,
                )

            return result

        except Exception as e:
            if self.verbose:
                print(f"    Agent error: {e}")
            return None

    def evaluate_single(
        self,
        test_case: Dict,
        index: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate a single test case.

        Args:
            test_case: Test case with query, trigger, action
            index: Sample index for logging

        Returns:
            Dict with prediction and metrics
        """
        query = test_case["query"]
        test_id = test_case.get("id", f"test_{index:03d}")

        if self.verbose:
            print(f"\n[{test_id}] {query[:60]}...")

        # Run agent
        start_time = time.time()
        prediction = self.run_agent(query)
        elapsed = time.time() - start_time

        if prediction is None:
            if self.verbose:
                print(f"    FAILED: No agent output")
            return {
                "test_id": test_id,
                "query": query,
                "success": False,
                "error": "Agent returned None",
                "elapsed_time": elapsed,
                "metrics": RAGASMetrics().to_dict(),
            }

        # Evaluate against reference (use RAGAS if available)
        metrics = evaluate_single_sample(
            prediction, test_case, query,
            use_ragas=self._use_ragas,
            ragas_llm=self._ragas_llm,
        )

        success = metrics.goal_accuracy >= 0.5

        # Extract applet info safely (handle None)
        final_applet = prediction.get("final_applet") or {}
        pred_trigger = (final_applet.get("trigger") or {}).get("service_name", "")
        pred_action = (final_applet.get("action") or {}).get("service_name", "")
        ref_trigger = test_case.get("trigger", {}).get("service_name", "")
        ref_action = test_case.get("action", {}).get("service_name", "")

        if self.verbose:
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"    {status}: goal={metrics.goal_accuracy:.0%}, "
                  f"trigger={metrics.trigger_accuracy:.0%}, "
                  f"action={metrics.action_accuracy:.0%}, "
                  f"joint={metrics.joint_accuracy:.0%}, "
                  f"faith={metrics.faithfulness:.0%}, "
                  f"rounds={metrics.negotiation_rounds}, "
                  f"time={elapsed:.1f}s")

            # Ground truth comparison
            t_match = "✓" if metrics.trigger_accuracy > 0 else "✗"
            a_match = "✓" if metrics.action_accuracy > 0 else "✗"
            print(f"    [COMPARISON]")
            print(f"      Trigger: {pred_trigger} {t_match}")
            print(f"        Expected: {ref_trigger}")
            print(f"      Action: {pred_action} {a_match}")
            print(f"        Expected: {ref_action}")

        return {
            "test_id": test_id,
            "query": query,
            "success": success,
            "elapsed_time": elapsed,
            "prediction": {
                "trigger": pred_trigger,
                "action": pred_action,
                "negotiation_rounds": prediction.get("pair_attempt", 0) + 1,
                "verifier_score": prediction.get("verifier_score", 0),
            },
            "reference": {
                "trigger": ref_trigger,
                "action": ref_action,
            },
            "metrics": metrics.to_dict(),
            "raw_prediction": prediction,  # Full state for debugging
        }

    def _save_intermediate_results(
        self,
        per_sample: List[Dict],
        current: int,
        total: int,
    ) -> None:
        """
        Save intermediate evaluation results to JSON.

        Args:
            per_sample: List of per-sample results so far
            current: Current sample number
            total: Total number of samples
        """
        if not self.save_path:
            return

        # Generate intermediate filename
        base_path = Path(self.save_path)
        intermediate_path = base_path.parent / f"{base_path.stem}_intermediate_{current:03d}{base_path.suffix}"

        # Compute running metrics
        success_count = sum(1 for s in per_sample if s.get("success", False))

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "status": "in_progress",
                "samples_completed": current,
                "samples_total": total,
                "success_rate": success_count / current if current > 0 else 0,
            },
            "per_sample": per_sample,
        }

        # Save
        intermediate_path.parent.mkdir(parents=True, exist_ok=True)
        with open(intermediate_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        if self.verbose:
            print(f"\n  [Saved intermediate results: {intermediate_path}]")

    def evaluate_test_set(
        self,
        test_set: List[Dict],
        limit: Optional[int] = None,
        full_test_data: Optional[List[Dict]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a full test set.

        Args:
            test_set: List of test cases
            limit: Optional limit on number of samples
            full_test_data: Full test data for reference mode (includes distractors)

        Returns:
            EvaluationResult with aggregated metrics
        """
        # Store full test data for reference mode (used to build candidates)
        if full_test_data:
            self._test_data = full_test_data
        else:
            self._test_data = test_set

        if limit:
            test_set = test_set[:limit]

        print(f"\nEvaluating {len(test_set)} samples...")
        print("=" * 70)

        predictions = []
        references = []
        queries = []
        per_sample = []

        for i, test_case in enumerate(tqdm(test_set, desc="Evaluating")):
            result = self.evaluate_single(test_case, i)
            per_sample.append(result)

            # Save intermediate results every N samples (if configured)
            if self.save_interval and self.save_path and (i + 1) % self.save_interval == 0:
                self._save_intermediate_results(per_sample, i + 1, len(test_set))

        # Aggregate metrics from per-sample results (already computed with RAGAS)
        eval_result = EvaluationResult()
        eval_result.num_samples = len(per_sample)
        eval_result.per_sample_results = per_sample

        # Collect metrics from successful samples
        all_metrics = []
        for sample in per_sample:
            if sample.get("metrics"):
                metrics_dict = sample["metrics"]
                m = RAGASMetrics(
                    goal_accuracy=metrics_dict.get("goal_accuracy", 0),
                    faithfulness=metrics_dict.get("faithfulness", 0),
                    context_recall_trigger=metrics_dict.get("context_recall_trigger", 0),
                    context_recall_action=metrics_dict.get("context_recall_action", 0),
                    context_precision_trigger=metrics_dict.get("context_precision_trigger", 0),
                    context_precision_action=metrics_dict.get("context_precision_action", 0),
                    trigger_accuracy=metrics_dict.get("trigger_accuracy", 0),
                    action_accuracy=metrics_dict.get("action_accuracy", 0),
                    joint_accuracy=metrics_dict.get("joint_accuracy", 0),
                    topic_adherence=metrics_dict.get("topic_adherence", 0),
                    negotiation_rounds=metrics_dict.get("negotiation_rounds", 1),
                    first_try_success=metrics_dict.get("first_try_success", False),
                )
                all_metrics.append(m)

                if m.goal_accuracy >= 0.5:
                    eval_result.num_success += 1
                else:
                    eval_result.num_failed += 1

        # Compute averages (handle NaN values)
        if all_metrics:
            import math
            n = len(all_metrics)

            def safe_avg(values):
                """Average that handles NaN values."""
                valid = [v for v in values if not math.isnan(v)]
                return sum(valid) / len(valid) if valid else 0.0

            eval_result.metrics = RAGASMetrics(
                goal_accuracy=safe_avg([m.goal_accuracy for m in all_metrics]),
                faithfulness=safe_avg([m.faithfulness for m in all_metrics]),
                context_recall_trigger=safe_avg([m.context_recall_trigger for m in all_metrics]),
                context_recall_action=safe_avg([m.context_recall_action for m in all_metrics]),
                context_precision_trigger=safe_avg([m.context_precision_trigger for m in all_metrics]),
                context_precision_action=safe_avg([m.context_precision_action for m in all_metrics]),
                trigger_accuracy=safe_avg([m.trigger_accuracy for m in all_metrics]),
                action_accuracy=safe_avg([m.action_accuracy for m in all_metrics]),
                joint_accuracy=safe_avg([m.joint_accuracy for m in all_metrics]),
                topic_adherence=safe_avg([m.topic_adherence for m in all_metrics]),
                negotiation_rounds=sum(m.negotiation_rounds for m in all_metrics) // n,
                first_try_success=sum(1 for m in all_metrics if m.first_try_success) > n // 2,
            )

        return eval_result


def _save_incremental_results(
    save_path: str,
    results: Dict[str, EvaluationResult],
    all_per_sample: List[Dict],
    splits: List[str],
    limit: Optional[int],
    use_mock: bool,
    use_selector: bool,
    use_reference: bool,
) -> None:
    """Save results incrementally after each split completes."""
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "splits_completed": list(results.keys()),
            "splits_total": splits,
            "limit_per_split": limit,
            "use_mock": use_mock,
            "use_selector": use_selector,
            "use_reference": use_reference,
            "status": "in_progress" if len(results) < len(splits) else "complete",
        },
        "summary": {},
        "per_split": {},
        "per_sample": all_per_sample,
    }

    for split, result in results.items():
        output["summary"][split] = {
            "num_samples": result.num_samples,
            "num_success": result.num_success,
            "success_rate": result.num_success / result.num_samples if result.num_samples > 0 else 0,
            "metrics": result.metrics.to_dict(),
        }
        output["per_split"][split] = result.to_dict()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)


def run_full_evaluation(
    splits: List[str] = ["gold", "noisy", "oneshot"],
    limit: Optional[int] = None,
    verbose: bool = False,
    use_mock: bool = False,
    use_selector: bool = False,
    use_reference: bool = False,
    save_path: Optional[str] = None,
    save_interval: Optional[int] = None,
) -> Dict[str, EvaluationResult]:
    """
    Run evaluation across multiple test splits.

    Args:
        splits: List of test set names
        limit: Optional limit per split
        verbose: Print detailed progress
        use_mock: Use mock mode
        use_selector: Use NEW multi-candidate selector design
        use_reference: Use reference data for candidates (no RAG needed)
        save_path: Path to save results
        save_interval: Save intermediate results every N samples (e.g., 5)

    Returns:
        Dict mapping split name to EvaluationResult
    """
    evaluator = AgenticEvaluator(
        verbose=verbose,
        use_mock=use_mock,
        use_selector=use_selector,
        use_reference=use_reference,
        save_interval=save_interval,
        save_path=save_path,
    )
    data_dir = Path(__file__).parent.parent / "data" / "test"

    results = {}
    all_per_sample = []

    for split in splits:
        test_path = data_dir / f"{split}.json"

        if not test_path.exists():
            print(f"\nWarning: {test_path} not found, skipping {split}")
            continue

        print(f"\n{'='*70}")
        print(f"EVALUATING: {split.upper()}")
        print(f"{'='*70}")

        test_set = evaluator.load_test_set(str(test_path))
        result = evaluator.evaluate_test_set(test_set, limit=limit)
        results[split] = result

        # Print results for this split
        print_results(result, split_name=split)

        # Collect per-sample results
        for sample in result.per_sample_results:
            sample["split"] = split
            all_per_sample.append(sample)

        # Save after each split (incremental save to avoid data loss)
        if save_path:
            _save_incremental_results(save_path, results, all_per_sample, splits, limit, use_mock, use_selector, use_reference)
            print(f"  [Saved progress after {split}]")

    # Print combined summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("COMBINED SUMMARY")
        print(f"{'='*70}")

        print(f"\n{'Split':<12} {'Success':<10} {'Goal Acc':<10} {'Joint Acc':<10} {'Topic Adh':<10}")
        print("-" * 55)

        for split, result in results.items():
            m = result.metrics
            rate = f"{result.num_success}/{result.num_samples}"
            print(f"{split:<12} {rate:<10} {m.goal_accuracy:<10.1%} {m.joint_accuracy:<10.1%} {m.topic_adherence:<10.1%}")

    # Save results
    if save_path:
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "splits": splits,
                "limit_per_split": limit,
                "use_mock": use_mock,
                "use_selector": use_selector,
                "use_reference": use_reference,
            },
            "summary": {},
            "per_split": {},
            "per_sample": all_per_sample,
        }

        for split, result in results.items():
            output["summary"][split] = {
                "num_samples": result.num_samples,
                "num_success": result.num_success,
                "success_rate": result.num_success / result.num_samples if result.num_samples > 0 else 0,
                "metrics": result.metrics.to_dict(),
            }
            output["per_split"][split] = result.to_dict()

        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"\n✓ Results saved to: {save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FARM Agentic AI Evaluation with RAGAS Metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run on all test sets
    python -m eval.run_agentic_eval

    # Run on specific splits
    python -m eval.run_agentic_eval --splits gold,noisy

    # Quick test with mock mode
    python -m eval.run_agentic_eval --mock --limit 5

    # Full evaluation with results saved
    python -m eval.run_agentic_eval --save results/full_eval.json

Metrics (RAGAS-based):
    - Goal Accuracy: Did applet achieve user intent?
    - Binding F1: Partial credit for correct bindings
    - Faithfulness: Is reasoning grounded in context?
    - Context Recall: RAG retrieval quality
    - Joint Accuracy: Trigger AND action correct
        """
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="gold,noisy,oneshot",
        help="Comma-separated list of test splits (default: gold,noisy,oneshot)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per split (for quick testing)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-sample results"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock mode (no LLM, for testing evaluator)"
    )
    parser.add_argument(
        "--selector",
        action="store_true",
        help="Use NEW multi-candidate selector design (verifier-driven fallback)"
    )
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Use reference data for candidates (test selector logic without RAG/LLM)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load test sets without running evaluation"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save intermediate results every N samples (e.g., 5). Requires --save."
    )

    args = parser.parse_args()

    # Parse splits
    splits = [s.strip() for s in args.splits.split(",")]

    if args.dry_run:
        data_dir = Path(__file__).parent.parent / "data" / "test"
        print("Dry run - checking test sets:")
        for split in splits:
            test_path = data_dir / f"{split}.json"
            if test_path.exists():
                with open(test_path) as f:
                    data = json.load(f)
                print(f"  ✓ {split}: {len(data)} samples")
            else:
                print(f"  ✗ {split}: NOT FOUND")
        return

    # Run evaluation
    results = run_full_evaluation(
        splits=splits,
        limit=args.limit,
        verbose=args.verbose,
        use_mock=args.mock,
        use_selector=args.selector,
        use_reference=args.reference,
        save_path=args.save,
        save_interval=args.save_interval,
    )

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()

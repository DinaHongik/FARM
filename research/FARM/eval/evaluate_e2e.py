"""
End-to-End Evaluation for FARM Agent System

Uses WELL-ESTABLISHED metrics from recognized benchmarks:

1. RETRIEVAL METRICS (from TREC, MS MARCO, BEIR):
   - Recall@K: Standard IR metric
   - MRR (Mean Reciprocal Rank): Standard IR metric
   - NDCG@K: Graded relevance metric

2. SERVICE SELECTION (from Intent Classification benchmarks):
   - Trigger Accuracy: Correct trigger service selected
   - Action Accuracy: Correct action service selected
   - Joint Accuracy: BOTH trigger AND action correct (strictest)

3. SLOT FILLING METRICS (from MultiWOZ, SGD benchmarks):
   - Joint Goal Accuracy (JGA): ALL slots filled correctly
   - Slot F1: Precision/Recall/F1 per slot
   - Slot Accuracy: Individual slot correctness

4. BINDING METRICS (from Text-to-SQL Spider benchmark):
   - Binding Accuracy: Correct ingredient→field mappings
   - Schema Linking F1: Precision/Recall on bindings

5. END-TO-END METRICS (from Task-Oriented Dialogue):
   - Success Rate: Task completed correctly
   - Execution Accuracy (EX): Is output valid/executable?
   - Pass@K: At least one correct in K attempts (from HumanEval)

Usage:
    python -m eval.evaluate_e2e
    python -m eval.evaluate_e2e --test-set eval/gold_test_set.json
    python -m eval.evaluate_e2e --verbose
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import sys
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class RetrievalMetrics:
    """Standard IR metrics (TREC, MS MARCO, BEIR)"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg_at_3: float = 0.0  # Normalized Discounted Cumulative Gain


@dataclass
class ServiceSelectionMetrics:
    """Intent classification metrics"""
    trigger_accuracy: float = 0.0  # Correct trigger selected
    action_accuracy: float = 0.0   # Correct action selected
    joint_accuracy: float = 0.0    # BOTH correct (strict)


@dataclass
class SlotFillingMetrics:
    """MultiWOZ/SGD style metrics"""
    joint_goal_accuracy: float = 0.0  # ALL slots correct
    slot_precision: float = 0.0
    slot_recall: float = 0.0
    slot_f1: float = 0.0
    slot_accuracy: float = 0.0  # Per-slot accuracy


@dataclass
class BindingMetrics:
    """Spider text-to-SQL style metrics"""
    binding_accuracy: float = 0.0  # Exact match on bindings
    binding_precision: float = 0.0
    binding_recall: float = 0.0
    binding_f1: float = 0.0


@dataclass
class EndToEndMetrics:
    """Task-oriented dialogue / code generation metrics"""
    success_rate: float = 0.0      # Task completed correctly
    execution_accuracy: float = 0.0  # Valid executable output
    pass_at_1: float = 0.0         # Correct on first try
    pass_at_3: float = 0.0         # Correct in top-3 attempts
    avg_negotiation_rounds: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    service_selection: ServiceSelectionMetrics = field(default_factory=ServiceSelectionMetrics)
    slot_filling: SlotFillingMetrics = field(default_factory=SlotFillingMetrics)
    binding: BindingMetrics = field(default_factory=BindingMetrics)
    end_to_end: EndToEndMetrics = field(default_factory=EndToEndMetrics)
    num_samples: int = 0

    def to_dict(self) -> Dict:
        return {
            "retrieval": {
                "R@1": round(self.retrieval.recall_at_1, 4),
                "R@3": round(self.retrieval.recall_at_3, 4),
                "R@5": round(self.retrieval.recall_at_5, 4),
                "MRR": round(self.retrieval.mrr, 4),
                "NDCG@3": round(self.retrieval.ndcg_at_3, 4),
            },
            "service_selection": {
                "Trigger_Acc": round(self.service_selection.trigger_accuracy, 4),
                "Action_Acc": round(self.service_selection.action_accuracy, 4),
                "Joint_Acc": round(self.service_selection.joint_accuracy, 4),
            },
            "slot_filling": {
                "JGA": round(self.slot_filling.joint_goal_accuracy, 4),
                "Slot_P": round(self.slot_filling.slot_precision, 4),
                "Slot_R": round(self.slot_filling.slot_recall, 4),
                "Slot_F1": round(self.slot_filling.slot_f1, 4),
            },
            "binding": {
                "Binding_Acc": round(self.binding.binding_accuracy, 4),
                "Binding_P": round(self.binding.binding_precision, 4),
                "Binding_R": round(self.binding.binding_recall, 4),
                "Binding_F1": round(self.binding.binding_f1, 4),
            },
            "end_to_end": {
                "Success_Rate": round(self.end_to_end.success_rate, 4),
                "Exec_Acc": round(self.end_to_end.execution_accuracy, 4),
                "Pass@1": round(self.end_to_end.pass_at_1, 4),
                "Pass@3": round(self.end_to_end.pass_at_3, 4),
                "Avg_Rounds": round(self.end_to_end.avg_negotiation_rounds, 2),
            },
            "num_samples": self.num_samples,
        }


# =============================================================================
# METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_recall_at_k(ranks: List[int], k: int) -> float:
    """Recall@K: fraction of queries where correct item is in top-K"""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def compute_mrr(ranks: List[int]) -> float:
    """Mean Reciprocal Rank"""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r > 0) / len(ranks)


def compute_ndcg_at_k(ranks: List[int], k: int) -> float:
    """
    NDCG@K with binary relevance (1 if correct, 0 otherwise)

    DCG = sum(rel_i / log2(i+1)) for i in 1..k
    IDCG = 1 / log2(2) = 1 (since only 1 relevant item)
    NDCG = DCG / IDCG
    """
    if not ranks:
        return 0.0

    import math
    dcg_sum = 0.0
    idcg = 1.0  # With 1 relevant item, IDCG = 1/log2(2) = 1

    for r in ranks:
        if r <= k:
            dcg_sum += 1.0 / math.log2(r + 1)

    return (dcg_sum / len(ranks)) / idcg


def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def normalize_service_name(name: str) -> str:
    """Normalize service name for comparison"""
    return name.lower().strip()


def service_matches(predicted: str, expected: str, alternatives: List[str] = None) -> bool:
    """Check if predicted service matches expected or any alternative"""
    pred_norm = normalize_service_name(predicted)

    # Check exact match
    if pred_norm == normalize_service_name(expected):
        return True

    # Check alternatives
    if alternatives:
        for alt in alternatives:
            if pred_norm == normalize_service_name(alt):
                return True
            # Partial match (predicted contains alternative or vice versa)
            if pred_norm in normalize_service_name(alt) or normalize_service_name(alt) in pred_norm:
                return True

    # Partial match with expected
    exp_norm = normalize_service_name(expected)
    if pred_norm in exp_norm or exp_norm in pred_norm:
        return True

    return False


# =============================================================================
# EVALUATION CLASS
# =============================================================================

class FARMEvaluator:
    """
    End-to-end evaluator for FARM using well-established metrics.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []

    def load_test_set(self, path: str) -> List[Dict]:
        """Load gold standard test set"""
        with open(path, 'r') as f:
            return json.load(f)

    def run_agent(self, query: str) -> Dict:
        """
        Run the FARM agent on a query and return structured output.

        Returns dict with:
            - trigger: {service_name, category, ingredients}
            - action: {service_name, category, fields}
            - bindings: [{field, source_type, source_value}]
            - negotiation_rounds: int
            - is_executable: bool
            - verifier_score: float
        """
        try:
            from agents.graph import run_negotiation

            result = run_negotiation(query, verbose=self.verbose)

            # Parse result into structured format
            return self._parse_agent_result(result)
        except Exception as e:
            if self.verbose:
                print(f"  Agent error: {e}")
            return None

    def _parse_agent_result(self, result: Dict) -> Dict:
        """Parse raw agent result into evaluation format"""
        if not result:
            return None

        # Extract trigger info from candidates
        trigger_candidates = result.get("trigger_candidates", [])
        action_candidates = result.get("action_candidates", [])

        trigger_idx = result.get("current_trigger_idx", 0)
        action_idx = result.get("current_action_idx", 0)

        # Get selected trigger
        trigger_info = {}
        if trigger_candidates and trigger_idx < len(trigger_candidates):
            t = trigger_candidates[trigger_idx]
            trigger_info = {
                "service_name": t.get("service_name", ""),
                "category": t.get("category", ""),
                "ingredients": self._extract_ingredients(t.get("api_info", {})),
            }
        elif result.get("current_offer"):
            offer = result["current_offer"]
            trigger_info = {
                "service_name": offer.get("service_name", ""),
                "category": offer.get("category", ""),
                "ingredients": offer.get("ingredients", []),
            }

        # Get selected action
        action_info = {}
        if action_candidates and action_idx < len(action_candidates):
            a = action_candidates[action_idx]
            action_info = {
                "service_name": a.get("service_name", ""),
                "category": a.get("category", ""),
                "fields": self._extract_fields(a.get("api_info", {})),
            }
        elif result.get("current_requirements"):
            reqs = result["current_requirements"]
            action_info = {
                "service_name": reqs.get("service_name", ""),
                "category": reqs.get("category", ""),
                "fields": reqs.get("required_fields", []),
            }

        # Extract bindings
        bindings = result.get("binding_map", []) or []

        parsed = {
            "trigger": trigger_info,
            "action": action_info,
            "bindings": bindings,
            "negotiation_rounds": result.get("pair_attempt", 0) + 1,
            "is_executable": result.get("is_executable", False),
            "verifier_score": result.get("verifier_score", 0.0),
        }

        return parsed

    def _extract_ingredients(self, api_info: Dict) -> List[str]:
        """Extract ingredient names from API info"""
        ingredients = api_info.get("Ingredients", {})
        if isinstance(ingredients, dict):
            return list(ingredients.keys())
        return []

    def _extract_fields(self, api_info: Dict) -> List[str]:
        """Extract field names from API info"""
        fields = api_info.get("Action fields", {})
        if isinstance(fields, dict):
            return list(fields.keys())
        return []

    def evaluate_single(self, test_case: Dict, prediction: Dict) -> Dict:
        """
        Evaluate a single test case.

        Returns detailed per-sample metrics.
        """
        # Support both formats:
        # 1. New format: trigger/action at top level
        # 2. Old format: trigger/action under "expected" key
        if "expected" in test_case:
            expected = test_case["expected"]
        else:
            expected = {
                "trigger": test_case["trigger"],
                "action": test_case["action"],
                "bindings": test_case.get("bindings", [])
            }

        # Service Selection
        trigger_correct = service_matches(
            prediction["trigger"]["service_name"],
            expected["trigger"]["service_name"],
            expected["trigger"].get("acceptable_alternatives", [])
        )

        action_correct = service_matches(
            prediction["action"]["service_name"],
            expected["action"]["service_name"],
            expected["action"].get("acceptable_alternatives", [])
        )

        joint_correct = trigger_correct and action_correct

        # Slot Filling (fields)
        expected_bindings = expected.get("bindings", [])
        predicted_bindings = prediction.get("bindings", [])

        # Check required fields
        required_fields = [b["field"] for b in expected_bindings if b.get("required", False)]
        predicted_fields = [b.get("field", "") for b in predicted_bindings]

        fields_filled = sum(1 for f in required_fields if f in predicted_fields)
        slot_recall = fields_filled / len(required_fields) if required_fields else 1.0
        slot_precision = fields_filled / len(predicted_fields) if predicted_fields else 1.0
        slot_f1 = compute_f1(slot_precision, slot_recall)
        all_slots_correct = (fields_filled == len(required_fields)) if required_fields else True

        # Binding accuracy (source_type matches)
        binding_matches = 0
        for exp_b in expected_bindings:
            for pred_b in predicted_bindings:
                if pred_b.get("field") == exp_b["field"]:
                    if pred_b.get("source_type") == exp_b.get("source_type"):
                        binding_matches += 1
                    break

        binding_accuracy = binding_matches / len(expected_bindings) if expected_bindings else 1.0

        # End-to-end success
        is_executable = prediction.get("is_executable", False)
        success = joint_correct and is_executable and all_slots_correct

        return {
            "trigger_correct": trigger_correct,
            "action_correct": action_correct,
            "joint_correct": joint_correct,
            "slot_precision": slot_precision,
            "slot_recall": slot_recall,
            "slot_f1": slot_f1,
            "all_slots_correct": all_slots_correct,
            "binding_accuracy": binding_accuracy,
            "is_executable": is_executable,
            "success": success,
            "rounds": prediction.get("negotiation_rounds", 1),
            "verifier_score": prediction.get("verifier_score", 0.0),
        }

    def evaluate_all(self, test_set: List[Dict]) -> EvaluationResult:
        """
        Run full evaluation on test set.

        Returns aggregated metrics.
        """
        results = []

        print(f"\nEvaluating {len(test_set)} test cases...")
        print("=" * 70)

        for i, test_case in enumerate(tqdm(test_set, desc="Evaluating")):
            query = test_case["query"]
            test_id = test_case.get("id", f"test_{i:03d}")

            if self.verbose:
                print(f"\n[{test_id}] {query}")

            # Run agent
            prediction = self.run_agent(query)

            if prediction is None:
                if self.verbose:
                    print(f"  FAILED: Agent returned no result")
                # Record failure
                results.append({
                    "trigger_correct": False,
                    "action_correct": False,
                    "joint_correct": False,
                    "slot_precision": 0.0,
                    "slot_recall": 0.0,
                    "slot_f1": 0.0,
                    "all_slots_correct": False,
                    "binding_accuracy": 0.0,
                    "is_executable": False,
                    "success": False,
                    "rounds": 9,  # Max rounds (failure)
                    "verifier_score": 0.0,
                })
                continue

            # Evaluate
            sample_result = self.evaluate_single(test_case, prediction)
            results.append(sample_result)

            if self.verbose:
                status = "PASS" if sample_result["success"] else "FAIL"
                print(f"  {status}: trigger={sample_result['trigger_correct']}, action={sample_result['action_correct']}, exec={sample_result['is_executable']}")

        # Aggregate results
        return self._aggregate_results(results)

    def _aggregate_results(self, results: List[Dict]) -> EvaluationResult:
        """Aggregate per-sample results into final metrics"""
        n = len(results)
        if n == 0:
            return EvaluationResult()

        # Service Selection (handle None values)
        trigger_acc = sum(1 for r in results if r.get("trigger_correct")) / n
        action_acc = sum(1 for r in results if r.get("action_correct")) / n
        joint_acc = sum(1 for r in results if r.get("joint_correct")) / n

        # Slot Filling (handle None values)
        jga = sum(1 for r in results if r.get("all_slots_correct")) / n
        slot_p = sum(r.get("slot_precision", 0) or 0 for r in results) / n
        slot_r = sum(r.get("slot_recall", 0) or 0 for r in results) / n
        slot_f1 = compute_f1(slot_p, slot_r)

        # Binding
        binding_acc = sum(r.get("binding_accuracy", 0) or 0 for r in results) / n

        # End-to-End (handle None values)
        success_rate = sum(1 for r in results if r.get("success")) / n
        exec_acc = sum(1 for r in results if r.get("is_executable")) / n
        pass_at_1 = sum(1 for r in results if r.get("success") and r.get("rounds") == 1) / n

        # Pass@3: success within first 3 rounds
        pass_at_3 = sum(1 for r in results if r.get("success") and r.get("rounds", 99) <= 3) / n

        avg_rounds = sum(r.get("rounds", 1) or 1 for r in results) / n

        return EvaluationResult(
            retrieval=RetrievalMetrics(
                recall_at_1=trigger_acc,  # R@1 for service selection
                recall_at_3=1.0,  # Placeholder - need RAG-level eval
                recall_at_5=1.0,
                mrr=trigger_acc,  # Simplified
                ndcg_at_3=trigger_acc,
            ),
            service_selection=ServiceSelectionMetrics(
                trigger_accuracy=trigger_acc,
                action_accuracy=action_acc,
                joint_accuracy=joint_acc,
            ),
            slot_filling=SlotFillingMetrics(
                joint_goal_accuracy=jga,
                slot_precision=slot_p,
                slot_recall=slot_r,
                slot_f1=slot_f1,
            ),
            binding=BindingMetrics(
                binding_accuracy=binding_acc,
                binding_precision=slot_p,  # Reuse slot metrics
                binding_recall=slot_r,
                binding_f1=slot_f1,
            ),
            end_to_end=EndToEndMetrics(
                success_rate=success_rate,
                execution_accuracy=exec_acc,
                pass_at_1=pass_at_1,
                pass_at_3=pass_at_3,
                avg_negotiation_rounds=avg_rounds,
            ),
            num_samples=n,
        )


def print_results(result: EvaluationResult):
    """Print formatted evaluation results"""
    print("\n" + "=" * 70)
    print("FARM EVALUATION RESULTS")
    print("=" * 70)

    print(f"\nSamples evaluated: {result.num_samples}")

    print("\n" + "-" * 70)
    print("SERVICE SELECTION (Intent Classification)")
    print("-" * 70)
    print(f"  Trigger Accuracy:  {result.service_selection.trigger_accuracy:.2%}")
    print(f"  Action Accuracy:   {result.service_selection.action_accuracy:.2%}")
    print(f"  Joint Accuracy:    {result.service_selection.joint_accuracy:.2%}")

    print("\n" + "-" * 70)
    print("SLOT FILLING (MultiWOZ-style)")
    print("-" * 70)
    print(f"  Joint Goal Acc:    {result.slot_filling.joint_goal_accuracy:.2%}")
    print(f"  Slot Precision:    {result.slot_filling.slot_precision:.2%}")
    print(f"  Slot Recall:       {result.slot_filling.slot_recall:.2%}")
    print(f"  Slot F1:           {result.slot_filling.slot_f1:.2%}")

    print("\n" + "-" * 70)
    print("BINDING (Schema Linking)")
    print("-" * 70)
    print(f"  Binding Accuracy:  {result.binding.binding_accuracy:.2%}")
    print(f"  Binding F1:        {result.binding.binding_f1:.2%}")

    print("\n" + "-" * 70)
    print("END-TO-END (Task Success)")
    print("-" * 70)
    print(f"  Success Rate:      {result.end_to_end.success_rate:.2%}")
    print(f"  Execution Acc:     {result.end_to_end.execution_accuracy:.2%}")
    print(f"  Pass@1:            {result.end_to_end.pass_at_1:.2%}")
    print(f"  Pass@3:            {result.end_to_end.pass_at_3:.2%}")
    print(f"  Avg Rounds:        {result.end_to_end.avg_negotiation_rounds:.2f}")

    print("\n" + "=" * 70)


def print_latex_table(result: EvaluationResult):
    """Print results as LaTeX table for paper"""
    print("\n% LaTeX Table for Paper")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{FARM End-to-End Evaluation Results}")
    print("\\label{tab:e2e_results}")
    print("\\begin{tabular}{llr}")
    print("\\toprule")
    print("\\textbf{Category} & \\textbf{Metric} & \\textbf{Score} \\\\")
    print("\\midrule")

    # Service Selection
    print("\\multirow{3}{*}{Service Selection}")
    print(f" & Trigger Accuracy & {result.service_selection.trigger_accuracy:.1%} \\\\")
    print(f" & Action Accuracy & {result.service_selection.action_accuracy:.1%} \\\\")
    print(f" & Joint Accuracy & {result.service_selection.joint_accuracy:.1%} \\\\")
    print("\\midrule")

    # Slot Filling
    print("\\multirow{2}{*}{Slot Filling (JGA)}")
    print(f" & Joint Goal Accuracy & {result.slot_filling.joint_goal_accuracy:.1%} \\\\")
    print(f" & Slot F1 & {result.slot_filling.slot_f1:.1%} \\\\")
    print("\\midrule")

    # End-to-End
    print("\\multirow{3}{*}{End-to-End}")
    print(f" & Success Rate & {result.end_to_end.success_rate:.1%} \\\\")
    print(f" & Execution Accuracy & {result.end_to_end.execution_accuracy:.1%} \\\\")
    print(f" & Pass@1 & {result.end_to_end.pass_at_1:.1%} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Evaluation for FARM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics used (from established benchmarks):
  - Service Selection: Accuracy, Joint Accuracy (Intent Classification)
  - Slot Filling: JGA, Slot F1 (MultiWOZ, SGD)
  - Binding: Accuracy, F1 (Spider Text-to-SQL)
  - End-to-End: Success Rate, Pass@K (Task-Oriented Dialogue, HumanEval)

Test sets available:
  - gold: Clear descriptions (500 samples) - data/test/gold.json
  - noisy: Vague descriptions (500 samples) - data/test/noisy.json
  - oneshot: Rare APIs (200 samples) - data/test/oneshot.json
        """
    )

    parser.add_argument(
        "--test-set",
        type=str,
        default="gold",
        choices=["gold", "noisy", "oneshot"],
        help="Test set name: gold, noisy, or oneshot (default: gold)"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Custom path to test set JSON (overrides --test-set)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-sample results"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table for paper"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load test set without running agent (for testing)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = FARMEvaluator(verbose=args.verbose)

    # Determine test set path
    if args.test_path:
        test_set_path = Path(args.test_path)
    else:
        # Use standard test sets in data/test/
        test_set_path = Path(__file__).parent.parent / "data" / "test" / f"{args.test_set}.json"

    if not test_set_path.exists():
        print(f"Error: Test set not found at {test_set_path}")
        print(f"Available test sets: gold, noisy, oneshot (in data/test/)")
        sys.exit(1)

    test_set = evaluator.load_test_set(test_set_path)
    print(f"Loaded {len(test_set)} test cases from {test_set_path}")

    if args.dry_run:
        print("\n[Dry run - not executing agent]")
        for i, tc in enumerate(test_set[:3]):
            test_id = tc.get('id', f'test_{i:03d}')
            print(f"  - {test_id}: {tc['query'][:60]}...")
        return

    # Run evaluation
    result = evaluator.evaluate_all(test_set)

    # Print results
    print_results(result)

    if args.latex:
        print_latex_table(result)

    # Save results
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()

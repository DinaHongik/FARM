"""
Evaluate FARM at Service Level (for TARGE Comparison)

This script evaluates FARM's retrieval performance at the service level
(Channel + Function name) to enable fair comparison with TARGE.

Evaluation is done on FARM's test data converted to TARGE format.

Metrics:
- Trigger EM: Exact match of trigger service (channel + function)
- Action EM: Exact match of action service (channel + function)
- Recipe EM: Both trigger AND action correct (same as TARGE's main metric)

Usage:
    python evaluate_farm_service_level.py [--dataset gold|noisy|oneshot]
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_targe_output(output: str) -> Tuple[str, str, str, str]:
    """
    Parse TARGE format output to extract channel and function names.

    Input: "IF {TriggerChannel} {TriggerFunc} THEN {ActionChannel} {ActionFunc}"
    Returns: (trigger_channel, trigger_func, action_channel, action_func)
    """
    # Remove "IF " prefix and split by " THEN "
    parts = output.replace("IF ", "").split(" THEN ")
    if len(parts) != 2:
        return "", "", "", ""

    trigger_part = parts[0].strip()
    action_part = parts[1].strip()

    # First word is channel, rest is function
    trigger_words = trigger_part.split(maxsplit=1)
    action_words = action_part.split(maxsplit=1)

    trigger_channel = trigger_words[0] if trigger_words else ""
    trigger_func = trigger_words[1] if len(trigger_words) > 1 else ""
    action_channel = action_words[0] if action_words else ""
    action_func = action_words[1] if len(action_words) > 1 else ""

    return trigger_channel, trigger_func, action_channel, action_func


def extract_channel_from_trigger_result(result: Dict) -> str:
    """Extract channel name from FARM trigger search result."""
    api_info = result.get("api_info", {})
    ingredients = api_info.get("Ingredients", {})

    for name, info in ingredients.items():
        if isinstance(info, dict) and "Filter code" in info:
            filter_code = info["Filter code"]
            channel = filter_code.split('.')[0]
            if channel.startswith('_'):
                channel = channel[1:]
            return channel
    return ""


def extract_channel_from_action_result(result: Dict) -> str:
    """Extract channel name from FARM action search result."""
    api_info = result.get("api_info", {})
    action_fields = api_info.get("Action fields", {})

    for name, info in action_fields.items():
        if isinstance(info, dict) and "Filter code method" in info:
            filter_code = info["Filter code method"]
            channel = filter_code.split('.')[0]
            if channel.startswith('_'):
                channel = channel[1:]
            return channel
    return ""


def normalize_string(s: str) -> str:
    """Normalize string for comparison (lowercase, strip whitespace)."""
    return s.lower().strip()


def evaluate_sample(
    query: str,
    expected_trigger_channel: str,
    expected_trigger_func: str,
    expected_action_channel: str,
    expected_action_func: str,
    retriever,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluate a single sample.

    Returns dict with:
        - trigger_correct: bool
        - action_correct: bool
        - recipe_correct: bool (both correct)
        - trigger_rank: int (rank of correct trigger, 0 if not found)
        - action_rank: int (rank of correct action, 0 if not found)
    """
    # Search for triggers
    trigger_results = retriever.search_triggers(query, top_k=top_k)
    action_results = retriever.search_actions(query, top_k=top_k)

    # Normalize expected values
    exp_t_channel = normalize_string(expected_trigger_channel)
    exp_t_func = normalize_string(expected_trigger_func)
    exp_a_channel = normalize_string(expected_action_channel)
    exp_a_func = normalize_string(expected_action_func)

    # Check trigger results
    trigger_correct = False
    trigger_rank = 0
    for i, result in enumerate(trigger_results, 1):
        result_channel = normalize_string(extract_channel_from_trigger_result(result))
        result_func = normalize_string(result.get("service_name", ""))

        # Check if channel and function match
        if result_channel == exp_t_channel and result_func == exp_t_func:
            trigger_correct = True
            trigger_rank = i
            break

    # Check action results
    action_correct = False
    action_rank = 0
    for i, result in enumerate(action_results, 1):
        result_channel = normalize_string(extract_channel_from_action_result(result))
        result_func = normalize_string(result.get("service_name", ""))

        if result_channel == exp_a_channel and result_func == exp_a_func:
            action_correct = True
            action_rank = i
            break

    return {
        "trigger_correct": trigger_correct,
        "action_correct": action_correct,
        "recipe_correct": trigger_correct and action_correct,
        "trigger_rank": trigger_rank,
        "action_rank": action_rank,
    }


def evaluate_dataset(
    dataset_path: Path,
    retriever,
    top_k: int = 5,
    max_samples: int = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate FARM on a dataset in TARGE format.

    Args:
        dataset_path: Path to test_*_recipe.json file
        retriever: FARM retriever instance
        top_k: Number of candidates to retrieve
        max_samples: Maximum samples to evaluate (None for all)
        verbose: Print progress

    Returns:
        Dictionary with metrics
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    # Results tracking
    trigger_correct = 0
    action_correct = 0
    recipe_correct = 0
    total = len(data)

    trigger_ranks = []
    action_ranks = []

    print(f"Evaluating {total} samples...")

    for i, sample in enumerate(data):
        query = sample["input"]
        output = sample["output"]

        # Parse expected output
        exp_t_channel, exp_t_func, exp_a_channel, exp_a_func = parse_targe_output(output)

        # Evaluate
        result = evaluate_sample(
            query=query,
            expected_trigger_channel=exp_t_channel,
            expected_trigger_func=exp_t_func,
            expected_action_channel=exp_a_channel,
            expected_action_func=exp_a_func,
            retriever=retriever,
            top_k=top_k
        )

        if result["trigger_correct"]:
            trigger_correct += 1
        if result["action_correct"]:
            action_correct += 1
        if result["recipe_correct"]:
            recipe_correct += 1

        if result["trigger_rank"] > 0:
            trigger_ranks.append(result["trigger_rank"])
        if result["action_rank"] > 0:
            action_ranks.append(result["action_rank"])

        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total} | Recipe EM: {recipe_correct/(i+1):.3f}")

    # Calculate metrics
    metrics = {
        "total_samples": total,
        "trigger_em": trigger_correct / total if total > 0 else 0,
        "action_em": action_correct / total if total > 0 else 0,
        "recipe_em": recipe_correct / total if total > 0 else 0,
        "trigger_correct": trigger_correct,
        "action_correct": action_correct,
        "recipe_correct": recipe_correct,
        "avg_trigger_rank": sum(trigger_ranks) / len(trigger_ranks) if trigger_ranks else 0,
        "avg_action_rank": sum(action_ranks) / len(action_ranks) if action_ranks else 0,
    }

    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str):
    """Print evaluation metrics in a nice format."""
    print("\n" + "=" * 60)
    print(f"FARM Service-Level Evaluation Results - {dataset_name.upper()}")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print()
    print("Exact Match (EM) Metrics:")
    print(f"  Trigger EM:  {metrics['trigger_em']:.4f} ({metrics['trigger_correct']}/{metrics['total_samples']})")
    print(f"  Action EM:   {metrics['action_em']:.4f} ({metrics['action_correct']}/{metrics['total_samples']})")
    print(f"  Recipe EM:   {metrics['recipe_em']:.4f} ({metrics['recipe_correct']}/{metrics['total_samples']})")
    print()
    print("Average Rank (when found):")
    print(f"  Trigger: {metrics['avg_trigger_rank']:.2f}")
    print(f"  Action:  {metrics['avg_action_rank']:.2f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FARM at service level for TARGE comparison"
    )
    parser.add_argument(
        "--dataset",
        choices=["gold", "noisy", "oneshot", "all"],
        default="gold",
        help="Dataset to evaluate (default: gold)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates to retrieve (default: 5)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress during evaluation"
    )

    args = parser.parse_args()

    # Import FARM retriever
    print("Loading FARM retriever...")
    try:
        from rag.retriever import FARMRetriever
        retriever = FARMRetriever()
        print("Retriever loaded successfully.")
    except Exception as e:
        print(f"Error loading retriever: {e}")
        print("Make sure RAG system is properly set up.")
        sys.exit(1)

    # Dataset paths
    data_dir = Path(__file__).parent / "farm_as_targe_format"

    datasets = {
        "gold": data_dir / "test_gold_recipe.json",
        "noisy": data_dir / "test_noisy_recipe.json",
        "oneshot": data_dir / "test_oneshot_recipe.json",
    }

    if args.dataset == "all":
        datasets_to_eval = list(datasets.items())
    else:
        datasets_to_eval = [(args.dataset, datasets[args.dataset])]

    all_results = {}

    for name, path in datasets_to_eval:
        if not path.exists():
            print(f"Warning: {path} not found, skipping...")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating {name.upper()} dataset")
        print(f"{'=' * 60}")

        metrics = evaluate_dataset(
            dataset_path=path,
            retriever=retriever,
            top_k=args.top_k,
            max_samples=args.max_samples,
            verbose=args.verbose
        )

        print_metrics(metrics, name)
        all_results[name] = metrics

    # Print comparison summary if evaluating multiple datasets
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY - Recipe EM Comparison")
        print("=" * 60)
        for name, metrics in all_results.items():
            print(f"  {name.upper():10} Recipe EM: {metrics['recipe_em']:.4f}")

    # Save results
    results_path = Path(__file__).parent / "farm_service_level_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()

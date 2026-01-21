#!/usr/bin/env python3
"""
FARM Agentic AI - Command Line Interface
========================================

Entry point for running the resolution system.

Usage:
    # Single query
    python -m agents.run "When darkness detected, log to spreadsheet"

    # With verbose output
    python -m agents.run "When darkness detected, log to spreadsheet" --verbose

    # Using mock data (no RAG required)
    python -m agents.run "When darkness detected, log to spreadsheet" --mock

    # Batch evaluation
    python -m agents.run --eval queries.json --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for module imports
AGENTS_DIR = Path(__file__).parent
PROJECT_ROOT = AGENTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.graph import run_resolution
from agents.metrics import get_metrics_tracker, ResolutionMetrics


def run_single_query(
    query: str,
    verbose: bool = False,
    use_mock: bool = False,
    use_simple: bool = False,
) -> dict:
    """
    Run resolution for a single query.

    Args:
        query: User query
        verbose: Print progress
        use_mock: Use mock candidates
        use_simple: Use simple (non-LLM) agents

    Returns:
        Final state dictionary
    """
    tracker = get_metrics_tracker()
    metrics = tracker.start_session(query)

    try:
        result = run_resolution(
            query=query,
            use_mock=use_mock,
            use_simple=use_simple,
            verbose=verbose,
        )

        tracker.end_session(metrics, result)
        return result

    except Exception as e:
        metrics.errors.append(str(e))
        metrics.final_status = "error"
        raise


def run_batch_evaluation(
    queries_file: str,
    output_file: Optional[str] = None,
    use_mock: bool = False,
    use_simple: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run batch evaluation on multiple queries.

    Args:
        queries_file: JSON file with list of queries
        output_file: Optional output file for results
        use_mock: Use mock candidates
        use_simple: Use simple agents
        verbose: Print progress

    Returns:
        Aggregate results dictionary
    """
    # Load queries
    with open(queries_file, "r") as f:
        data = json.load(f)

    # Handle different input formats
    if isinstance(data, list):
        queries = data
    elif isinstance(data, dict) and "queries" in data:
        queries = data["queries"]
    else:
        raise ValueError("Invalid queries file format")

    tracker = get_metrics_tracker()
    results = []

    print(f"Running evaluation on {len(queries)} queries...")
    print("-" * 60)

    for i, query in enumerate(queries, 1):
        # Handle query as string or dict
        if isinstance(query, dict):
            query_text = query.get("query", query.get("text", str(query)))
        else:
            query_text = str(query)

        if verbose:
            print(f"\n[{i}/{len(queries)}] {query_text[:50]}...")

        metrics = tracker.start_session(query_text, session_id=f"batch_{i}")

        try:
            result = run_resolution(
                query=query_text,
                use_mock=use_mock,
                use_simple=use_simple,
                verbose=False,  # Don't print each step in batch mode
            )

            tracker.end_session(metrics, result)

            results.append({
                "query": query_text,
                "success": result.get("final_applet") is not None,
                "score": result.get("verifier_score", 0),
                "pairs_tried": result.get("pair_attempt", 0) + 1,
                "applet": result.get("final_applet"),
            })

            if verbose:
                status = "OK" if result.get("final_applet") else "FAIL"
                score = result.get("verifier_score", 0)
                print(f"    [{status}] Score: {score:.2f}")

        except Exception as e:
            metrics.errors.append(str(e))
            metrics.final_status = "error"

            results.append({
                "query": query_text,
                "success": False,
                "error": str(e),
            })

            if verbose:
                print(f"    [ERROR] {e}")

    # Print summary
    tracker.print_summary()

    # Save results
    if output_file:
        output_data = {
            "aggregate": tracker.get_aggregate_metrics(),
            "results": results,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return tracker.get_aggregate_metrics()


def print_applet(applet: dict):
    """Pretty print a full applet with all details."""
    print("\n" + "=" * 70)
    print("GENERATED EXECUTABLE APPLET")
    print("=" * 70)

    print(f"\nQuery: {applet.get('query', 'N/A')}")

    # Trigger Section
    trigger = applet.get("trigger", {})
    print(f"\n{'-' * 70}")
    print("TRIGGER")
    print(f"{'-' * 70}")
    print(f"  Service:     {trigger.get('service_name', 'N/A')}")
    print(f"  Category:    {trigger.get('category', 'N/A')}")
    print(f"  Description: {trigger.get('description', 'N/A')[:80]}...")

    # Trigger Ingredients (OFFER)
    ingredients = trigger.get("ingredients", [])
    api_info = trigger.get("api_info", {})
    raw_ingredients = api_info.get("Ingredients", {})

    if raw_ingredients:
        print(f"\n  INGREDIENTS (OFFER) - {len(raw_ingredients)} available:")
        for name, info in raw_ingredients.items():
            if isinstance(info, dict):
                ing_type = info.get("Type", "String")
                example = info.get("Example", "N/A")
                slug = info.get("Slug", name)
                filter_code = info.get("Filter code", "")
                print(f"    - {name}")
                print(f"        Slug:        {slug}")
                print(f"        Type:        {ing_type}")
                example_str = str(example)
                print(f"        Example:     {example_str[:50]}..." if len(example_str) > 50 else f"        Example:     {example_str}")
                if filter_code:
                    print(f"        Filter code: {filter_code}")
    elif ingredients:
        print(f"\n  INGREDIENTS (OFFER) - {len(ingredients)} available:")
        for ing in ingredients:
            print(f"    - {ing}")

    # Action Section
    action = applet.get("action", {})
    print(f"\n{'-' * 70}")
    print("ACTION")
    print(f"{'-' * 70}")
    print(f"  Service:     {action.get('service_name', 'N/A')}")
    print(f"  Category:    {action.get('category', 'N/A')}")
    print(f"  Description: {action.get('description', 'N/A')[:80]}...")

    # Action Fields (REQUIREMENTS)
    action_api = action.get("api_info", {})
    raw_fields = action_api.get("Action fields", {})

    if raw_fields:
        print(f"\n  ACTION FIELDS (REQUIREMENTS) - {len(raw_fields)} fields:")
        for name, info in raw_fields.items():
            if isinstance(info, dict):
                required = info.get("Required", "false")
                label = info.get("Label", name)
                slug = info.get("Slug", name)
                helper = info.get("Helper text", "")
                filter_method = info.get("Filter code method", "")
                req_marker = "[REQUIRED]" if required.lower() == "true" else "[optional]"
                print(f"    - {name} {req_marker}")
                print(f"        Label:  {label}")
                print(f"        Slug:   {slug}")
                if helper:
                    helper_str = str(helper)
                    print(f"        Help:   {helper_str[:50]}..." if len(helper_str) > 50 else f"        Help:   {helper_str}")
                if filter_method:
                    print(f"        Method: {filter_method}")

    # Bindings
    bindings = applet.get("bindings", [])
    print(f"\n{'-' * 70}")
    print("BINDINGS (Ingredient -> Field Mapping)")
    print(f"{'-' * 70}")
    if bindings:
        for b in bindings:
            field = b.get("action_field", "?")
            source = b.get("source_type", "?")
            if source == "ingredient":
                ing_name = b.get("ingredient_name", "?")
                print(f"  {field} <- {{{{{{ing_name}}}}}}")
                print(f"      Source: Trigger ingredient '{ing_name}'")
            else:
                static_val = b.get("static_value", "?")
                print(f"  {field} <- \"{static_val}\"")
                print(f"      Source: Static value")
            if b.get("reasoning"):
                print(f"      Reason: {b['reasoning'][:60]}...")
    else:
        # Show field_values if no bindings
        field_values = action.get("field_values", {})
        if field_values:
            for field, value in field_values.items():
                print(f"  {field} <- {value}")

    # Metadata
    metadata = applet.get("metadata", {})
    print(f"\n{'-' * 70}")
    print("METADATA")
    print(f"{'-' * 70}")
    print(f"  Verifier Score:      {metadata.get('verifier_score', 'N/A')}")
    print(f"  Resolution Rounds:  {metadata.get('resolution_rounds', 'N/A')}")
    print(f"  Is Executable:       {metadata.get('is_executable', 'N/A')}")
    if metadata.get("verifier_critique"):
        print(f"  Critique:            {metadata['verifier_critique'][:60]}...")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FARM Agentic AI - Contract-based resolution for TAP synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m agents.run "When darkness detected, log to spreadsheet"
  python -m agents.run "Send notification when new email" --verbose
  python -m agents.run --mock --simple "Test query"
  python -m agents.run --eval queries.json --output results.json
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="User query describing the desired applet"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress information"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock candidates instead of RAG"
    )

    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple (non-LLM) agent implementations"
    )

    parser.add_argument(
        "--eval",
        metavar="FILE",
        help="Run batch evaluation on queries from JSON file"
    )

    parser.add_argument(
        "--output", "-o",
        metavar="FILE",
        help="Output file for batch results (JSON)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON"
    )

    args = parser.parse_args()

    # Batch evaluation mode
    if args.eval:
        run_batch_evaluation(
            queries_file=args.eval,
            output_file=args.output,
            use_mock=args.mock,
            use_simple=args.simple,
            verbose=args.verbose,
        )
        return

    # Single query mode
    if not args.query:
        parser.print_help()
        print("\nError: Please provide a query or use --eval for batch mode")
        sys.exit(1)

    result = run_single_query(
        query=args.query,
        verbose=args.verbose,
        use_mock=args.mock,
        use_simple=args.simple,
    )

    # Output result
    if args.json:
        # JSON output
        output = {
            "query": args.query,
            "success": result.get("final_applet") is not None,
            "applet": result.get("final_applet"),
            "score": result.get("verifier_score"),
            "status": result.get("resolution_status"),
        }
        print(json.dumps(output, indent=2))
    else:
        # Pretty print
        applet = result.get("final_applet")
        if applet:
            print_applet(applet)
        else:
            print("\n" + "=" * 60)
            print("NEGOTIATION FAILED")
            print("=" * 60)
            print(f"Status: {result.get('resolution_status', 'Unknown')}")
            if result.get("error"):
                print(f"Error: {result['error']}")
            if result.get("rejection_history"):
                print(f"Rejections: {len(result['rejection_history'])}")
                for r in result["rejection_history"]:
                    print(f"  - T{r['trigger_idx']}-A{r['action_idx']}: {r['reason'][:50]}...")


if __name__ == "__main__":
    main()

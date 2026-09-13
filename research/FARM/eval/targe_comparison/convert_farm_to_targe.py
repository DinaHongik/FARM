"""
Convert FARM Dataset to TARGE Format

This script converts FARM's rich test data format to TARGE's service-level format
for fair comparison between the two systems.

FARM format:
{
    "query": "...",
    "trigger": {"service_name": "...", "api_info": {...}},
    "action": {"service_name": "...", "api_info": {...}}
}

TARGE format (test_recipe.json):
{
    "input": "...",
    "output": "IF {channel} {trigger_func} THEN {channel} {action_func}",
    "score": 0.8,
    "instruction": "..."
}

TARGE format (test_trigger.json):
{
    "input": "...",
    "output": "TRIGGER SERVICE: {channel}, TRIGGER EVENT: {trigger_func}",
    "score": 0.8,
    "instruction": "..."
}

TARGE format (test_action.json):
{
    "input": "...",
    "output": "ACTION SERVICE: {channel}, ACTION EVENT: {action_func}",
    "score": 0.8,
    "instruction": "..."
}

Usage:
    python convert_farm_to_targe.py
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def extract_channel_from_trigger(api_info: Dict[str, Any]) -> str:
    """
    Extract channel name from trigger's api_info.

    Looks in Ingredients -> Filter code -> first part before '.'
    Example: "Nytimes.newArticleMatchingSearch.Title" -> "Nytimes"
    """
    ingredients = api_info.get("Ingredients", {})
    for name, info in ingredients.items():
        if isinstance(info, dict) and "Filter code" in info:
            filter_code = info["Filter code"]
            # Extract channel: first part before '.'
            channel = filter_code.split('.')[0]
            # Remove leading underscore if present (e.g., "_5MinuteCrafts" -> "5MinuteCrafts")
            if channel.startswith('_'):
                channel = channel[1:]
            return channel
    return ""


def extract_channel_from_action(api_info: Dict[str, Any]) -> str:
    """
    Extract channel name from action's api_info.

    Looks in Action fields -> Filter code method -> first part before '.'
    Example: "Evernote.appendToNote.setTitle(string: title)" -> "Evernote"
    """
    action_fields = api_info.get("Action fields", {})
    for name, info in action_fields.items():
        if isinstance(info, dict) and "Filter code method" in info:
            filter_code = info["Filter code method"]
            # Extract channel: first part before '.'
            channel = filter_code.split('.')[0]
            if channel.startswith('_'):
                channel = channel[1:]
            return channel
    return ""


def convert_sample_to_targe(sample: Dict[str, Any]) -> Tuple[Dict, Dict, Dict]:
    """
    Convert a single FARM sample to TARGE format.

    Returns:
        Tuple of (recipe_format, trigger_format, action_format)
    """
    query = sample.get("query", "")
    trigger = sample.get("trigger", {})
    action = sample.get("action", {})

    # Extract service names
    trigger_service = trigger.get("service_name", "")
    action_service = action.get("service_name", "")

    # Extract channels from api_info
    trigger_api_info = trigger.get("api_info", {})
    action_api_info = action.get("api_info", {})

    trigger_channel = extract_channel_from_trigger(trigger_api_info)
    action_channel = extract_channel_from_action(action_api_info)

    # Use service_name as fallback if channel extraction fails
    if not trigger_channel:
        trigger_channel = trigger_service.split()[0] if trigger_service else "Unknown"
    if not action_channel:
        action_channel = action_service.split()[0] if action_service else "Unknown"

    # Default score (TARGE uses similarity score, we use 0.8 as placeholder)
    score = 0.8

    # Recipe format: "IF {channel} {func} THEN {channel} {func}"
    recipe_output = f"IF {trigger_channel} {trigger_service} THEN {action_channel} {action_service}"
    recipe_format = {
        "input": query,
        "output": recipe_output,
        "score": score,
        "instruction": "From the description of a rule: identify the 'trigger', identify the 'action', write a IF 'trigger' THEN 'action' rule."
    }

    # Trigger format: "TRIGGER SERVICE: {channel}, TRIGGER EVENT: {func}"
    trigger_output = f"TRIGGER SERVICE: {trigger_channel}, TRIGGER EVENT: {trigger_service}"
    trigger_format = {
        "input": query,
        "output": trigger_output,
        "score": score,
        "instruction": "From the description of a rule: identify the 'trigger'"
    }

    # Action format: "ACTION SERVICE: {channel}, ACTION EVENT: {func}"
    action_output = f"ACTION SERVICE: {action_channel}, ACTION EVENT: {action_service}"
    action_format = {
        "input": query,
        "output": action_output,
        "score": score,
        "instruction": "From the description of a rule: identify the 'action'"
    }

    return recipe_format, trigger_format, action_format


def convert_dataset(input_path: Path, output_dir: Path, dataset_name: str) -> Dict[str, int]:
    """
    Convert a FARM dataset file to TARGE format files.

    Creates three files:
    - test_{name}_recipe.json
    - test_{name}_trigger.json
    - test_{name}_action.json

    Returns:
        Statistics about the conversion
    """
    # Load FARM data
    with open(input_path, 'r', encoding='utf-8') as f:
        farm_data = json.load(f)

    recipes = []
    triggers = []
    actions = []

    stats = {
        "total": len(farm_data),
        "converted": 0,
        "trigger_channel_extracted": 0,
        "action_channel_extracted": 0,
        "errors": 0
    }

    for sample in farm_data:
        try:
            recipe, trigger, action = convert_sample_to_targe(sample)
            recipes.append(recipe)
            triggers.append(trigger)
            actions.append(action)
            stats["converted"] += 1

            # Check if channels were extracted
            trigger_api = sample.get("trigger", {}).get("api_info", {})
            action_api = sample.get("action", {}).get("api_info", {})
            if extract_channel_from_trigger(trigger_api):
                stats["trigger_channel_extracted"] += 1
            if extract_channel_from_action(action_api):
                stats["action_channel_extracted"] += 1

        except Exception as e:
            stats["errors"] += 1
            print(f"Error converting sample: {e}")
            continue

    # Save converted files
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe_path = output_dir / f"test_{dataset_name}_recipe.json"
    trigger_path = output_dir / f"test_{dataset_name}_trigger.json"
    action_path = output_dir / f"test_{dataset_name}_action.json"

    with open(recipe_path, 'w', encoding='utf-8') as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)

    with open(trigger_path, 'w', encoding='utf-8') as f:
        json.dump(triggers, f, indent=2, ensure_ascii=False)

    with open(action_path, 'w', encoding='utf-8') as f:
        json.dump(actions, f, indent=2, ensure_ascii=False)

    print(f"\n=== {dataset_name.upper()} Conversion Stats ===")
    print(f"Total samples: {stats['total']}")
    print(f"Converted: {stats['converted']}")
    print(f"Trigger channels extracted: {stats['trigger_channel_extracted']}")
    print(f"Action channels extracted: {stats['action_channel_extracted']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output files:")
    print(f"  - {recipe_path}")
    print(f"  - {trigger_path}")
    print(f"  - {action_path}")

    return stats


def main():
    """Convert all FARM test datasets to TARGE format."""
    # Paths
    project_root = Path(__file__).parent.parent.parent
    farm_test_dir = project_root / "data" / "test"
    output_dir = Path(__file__).parent / "farm_as_targe_format"

    print("=" * 60)
    print("FARM to TARGE Format Converter")
    print("=" * 60)
    print(f"Input directory: {farm_test_dir}")
    print(f"Output directory: {output_dir}")

    # Convert each dataset
    datasets = [
        ("gold.json", "gold"),
        ("noisy.json", "noisy"),
        ("oneshot.json", "oneshot"),
    ]

    all_stats = {}
    for filename, name in datasets:
        input_path = farm_test_dir / filename
        if input_path.exists():
            stats = convert_dataset(input_path, output_dir, name)
            all_stats[name] = stats
        else:
            print(f"\nWarning: {input_path} not found, skipping...")

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    total_converted = sum(s.get("converted", 0) for s in all_stats.values())
    print(f"Total samples converted: {total_converted}")
    print(f"\nOutput files are in: {output_dir}")
    print("\nThese files can be used with TARGE's evaluation scripts.")

    # Show sample output
    sample_path = output_dir / "test_gold_recipe.json"
    if sample_path.exists():
        with open(sample_path, 'r') as f:
            samples = json.load(f)
        if samples:
            print("\n=== Sample Converted Entry ===")
            print(json.dumps(samples[0], indent=2))


if __name__ == "__main__":
    main()

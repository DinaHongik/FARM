#!/usr/bin/env python3
"""
Generate Reference Bindings for Test Data
==========================================

Creates semantic bindings between trigger ingredients and action fields
for use in binding F1 evaluation metrics.

Usage:
    python scripts/generate_reference_bindings.py
    python scripts/generate_reference_bindings.py --input data/test/gold.json --output data/test/gold_with_bindings.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional


# =============================================================================
# SEMANTIC MATCHING RULES
# =============================================================================

# Maps action field patterns to ingredient patterns (case-insensitive)
SEMANTIC_MAPPINGS = {
    # Title/Name fields
    "title": ["title", "name", "subject", "entrytitle", "headline"],
    "name": ["name", "title", "filename"],
    "subject": ["subject", "title", "name", "entrytitle"],

    # Content/Body fields
    "body": ["content", "body", "text", "blurb", "description", "entrycontent", "message"],
    "content": ["content", "body", "text", "blurb", "entrycontent"],
    "message": ["message", "content", "body", "text", "blurb"],
    "text": ["text", "content", "body", "message"],
    "description": ["description", "blurb", "content", "summary"],

    # URL fields
    "url": ["url", "link", "articleurl", "entryurl", "imageurl", "sourceurl"],
    "link": ["url", "link", "articleurl", "entryurl"],
    "image": ["imageurl", "image", "photo", "picture", "entryimageurl"],
    "imageurl": ["imageurl", "image", "entryimageurl", "photo"],
    "photo": ["photo", "image", "imageurl", "picture"],

    # Author/User fields
    "author": ["author", "creator", "user", "by", "from"],
    "from": ["from", "author", "sender", "user"],
    "user": ["user", "author", "username", "creator"],

    # Date/Time fields
    "date": ["date", "time", "published", "created", "timestamp", "publisheddate", "entrypublished"],
    "time": ["time", "date", "timestamp", "published"],
    "published": ["published", "date", "publisheddate", "entrypublished", "created"],

    # Location fields
    "location": ["location", "address", "place", "city"],
    "address": ["address", "location", "street"],

    # Category/Tags
    "tags": ["tags", "keywords", "categories", "labels"],
    "keywords": ["keywords", "tags", "categories"],
    "category": ["category", "section", "type", "tags"],

    # Source
    "source": ["source", "origin", "from", "via"],

    # ID fields
    "id": ["id", "identifier", "code"],

    # Row/formatted for spreadsheets - special handling
    "row": ["*"],  # Use all ingredients
    "formatted": ["*"],  # Use all ingredients
}


def normalize(s: str) -> str:
    """Normalize string for matching."""
    return s.lower().replace("_", "").replace(" ", "").replace("-", "")


def extract_ingredients(trigger_api: Dict) -> List[Dict[str, str]]:
    """Extract ingredients from trigger API info."""
    ingredients = []
    raw_ingredients = trigger_api.get("Ingredients", {})

    if not isinstance(raw_ingredients, dict):
        return ingredients

    for name, info in raw_ingredients.items():
        if isinstance(info, dict):
            ingredients.append({
                "name": name,
                "slug": info.get("Slug", name),
                "type": info.get("Type", "String"),
            })
        else:
            # Simple string value
            ingredients.append({
                "name": name,
                "slug": name,
                "type": "String",
            })

    return ingredients


def extract_action_fields(action_api: Dict) -> List[Dict[str, Any]]:
    """Extract required fields from action API info."""
    fields = []
    raw_fields = action_api.get("Action fields", {})

    if not isinstance(raw_fields, dict):
        return fields

    for name, info in raw_fields.items():
        if isinstance(info, dict):
            required = str(info.get("Required", "false")).lower() == "true"
            fields.append({
                "name": name,
                "slug": info.get("Slug", name),
                "required": required,
            })
        else:
            # Simple string value - treat as required field
            fields.append({
                "name": name,
                "slug": name.lower().replace(" ", "_"),
                "required": True,
            })

    return fields


def find_best_ingredient(
    field_name: str,
    ingredients: List[Dict[str, str]],
    used_ingredients: set,
) -> Optional[str]:
    """
    Find the best matching ingredient for an action field.

    Args:
        field_name: Action field name
        ingredients: Available trigger ingredients
        used_ingredients: Set of already used ingredient names

    Returns:
        Best matching ingredient name or None
    """
    field_norm = normalize(field_name)

    # Check semantic mappings
    for pattern, matches in SEMANTIC_MAPPINGS.items():
        if pattern in field_norm or field_norm in pattern:
            # Found a relevant pattern, look for matching ingredients
            for match in matches:
                if match == "*":
                    # Return first unused ingredient
                    for ing in ingredients:
                        if ing["name"] not in used_ingredients:
                            return ing["name"]
                    continue

                for ing in ingredients:
                    ing_norm = normalize(ing["name"])
                    if match in ing_norm or ing_norm in match:
                        if ing["name"] not in used_ingredients:
                            return ing["name"]

    # Direct matching as fallback
    for ing in ingredients:
        ing_norm = normalize(ing["name"])
        if field_norm in ing_norm or ing_norm in field_norm:
            if ing["name"] not in used_ingredients:
                return ing["name"]

    return None


def generate_bindings(
    trigger_api: Dict,
    action_api: Dict,
    action_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Generate reference bindings between trigger and action.

    Args:
        trigger_api: Trigger API info
        action_api: Action API info
        action_name: Action service name for smart defaults

    Returns:
        List of binding dictionaries
    """
    ingredients = extract_ingredients(trigger_api)
    action_fields = extract_action_fields(action_api)

    bindings = []
    used_ingredients = set()
    action_lower = action_name.lower()

    for field in action_fields:
        field_name = field["name"]
        field_slug = field["slug"]
        field_norm = normalize(field_name)

        # Special handling for spreadsheet row fields
        if "spreadsheet" in action_lower or "row" in action_lower:
            if "formatted" in field_norm or "row" in field_norm:
                # Combine all ingredients
                ing_names = [ing["name"] for ing in ingredients]
                formatted = "|||".join([f"{{{{{name}}}}}" for name in ing_names])
                bindings.append({
                    "action_field": field_name,
                    "source_type": "static",
                    "ingredient_name": None,
                    "static_value": formatted,
                    "reasoning": f"Combined all {len(ing_names)} trigger ingredients"
                })
                continue

        # Try to find matching ingredient
        matched_ing = find_best_ingredient(field_name, ingredients, used_ingredients)

        if matched_ing:
            bindings.append({
                "action_field": field_name,
                "source_type": "ingredient",
                "ingredient_name": matched_ing,
                "static_value": None,
                "reasoning": f"Semantic match: {matched_ing} -> {field_name}"
            })
            used_ingredients.add(matched_ing)
        else:
            # No match - use placeholder or smart default
            if not field["required"]:
                # Optional field - skip or use empty
                bindings.append({
                    "action_field": field_name,
                    "source_type": "static",
                    "ingredient_name": None,
                    "static_value": "",
                    "reasoning": f"Optional field, no matching ingredient"
                })
            else:
                # Required field - use placeholder
                bindings.append({
                    "action_field": field_name,
                    "source_type": "static",
                    "ingredient_name": None,
                    "static_value": f"[{field_name}]",
                    "reasoning": f"Required field, no matching ingredient"
                })

    return bindings


def process_test_file(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    Process a test file and add reference bindings.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file

    Returns:
        Statistics about the processing
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    stats = {
        "total_samples": len(data),
        "samples_with_bindings": 0,
        "total_bindings": 0,
        "ingredient_bindings": 0,
        "static_bindings": 0,
    }

    for sample in data:
        trigger_api = sample.get("trigger", {}).get("api_info", {})
        action_api = sample.get("action", {}).get("api_info", {})
        action_name = sample.get("action", {}).get("service_name", "")

        if trigger_api and action_api:
            bindings = generate_bindings(trigger_api, action_api, action_name)
            sample["reference_bindings"] = bindings

            stats["samples_with_bindings"] += 1
            stats["total_bindings"] += len(bindings)
            stats["ingredient_bindings"] += sum(
                1 for b in bindings if b["source_type"] == "ingredient"
            )
            stats["static_bindings"] += sum(
                1 for b in bindings if b["source_type"] == "static"
            )

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate reference bindings for test data"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input JSON file (or process all test files)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify files in place"
    )

    args = parser.parse_args()

    # Find test data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    test_dir = project_root / "data" / "test"

    if args.input:
        # Process single file
        input_path = Path(args.input)
        if args.inplace:
            output_path = input_path
        elif args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_with_bindings{input_path.suffix}"

        print(f"Processing: {input_path}")
        stats = process_test_file(str(input_path), str(output_path))
        print(f"  Saved to: {output_path}")
        print(f"  Samples: {stats['total_samples']}")
        print(f"  Total bindings: {stats['total_bindings']}")
        print(f"  Ingredient bindings: {stats['ingredient_bindings']}")
        print(f"  Static bindings: {stats['static_bindings']}")

    else:
        # Process all test files
        test_files = ["gold.json", "noisy.json", "oneshot.json"]

        print("Processing all test files...")
        print("=" * 50)

        for filename in test_files:
            input_path = test_dir / filename
            if not input_path.exists():
                print(f"  Skipping {filename} (not found)")
                continue

            if args.inplace:
                output_path = input_path
            else:
                output_path = test_dir / f"{input_path.stem}_with_bindings{input_path.suffix}"

            print(f"\n{filename}:")
            stats = process_test_file(str(input_path), str(output_path))
            print(f"  Saved to: {output_path.name}")
            print(f"  Samples: {stats['total_samples']}")
            print(f"  Total bindings: {stats['total_bindings']}")
            print(f"  Ingredient bindings: {stats['ingredient_bindings']} ({stats['ingredient_bindings']/max(1,stats['total_bindings'])*100:.1f}%)")
            print(f"  Static bindings: {stats['static_bindings']} ({stats['static_bindings']/max(1,stats['total_bindings'])*100:.1f}%)")

        print("\n" + "=" * 50)
        print("Done! Reference bindings added to test files.")
        if not args.inplace:
            print("Use --inplace to modify original files.")


if __name__ == "__main__":
    main()

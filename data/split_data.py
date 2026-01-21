"""
Split IFTTT applets into train/test sets.

Creates:
- data/train_applets.json - For training (90%)
- data/test/gold.json - Clear descriptions from test set
- data/test/noisy.json - Vague descriptions from test set
- data/test/oneshot.json - Rare APIs from test set
"""

import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple


def load_applets(filepath: Path) -> List[Dict]:
    """Load and flatten applets from the nested structure."""
    with open(filepath, 'r', encoding='utf-8') as f:
        services = json.load(f)

    applets = []
    skipped = 0

    for service in services:
        if not isinstance(service, dict):
            skipped += 1
            continue

        for applet in service.get("applets", []):
            # Skip malformed entries
            if not isinstance(applet, dict):
                skipped += 1
                continue

            # Ensure applet has required components
            components = applet.get("components", [])
            if not isinstance(components, list):
                skipped += 1
                continue

            trigger = None
            action = None

            for comp in components:
                if not isinstance(comp, dict):
                    continue
                if comp.get("label") == "If":
                    trigger = comp
                elif comp.get("label") == "Then":
                    action = comp

            if trigger and action and applet.get("description"):
                applets.append({
                    "query": applet["description"],
                    "trigger": trigger,
                    "action": action,
                    "user_count": applet.get("user_count", "0"),
                    "applet_url": applet.get("applet_url", "")
                })

    if skipped > 0:
        print(f"Skipped {skipped} malformed entries")

    return applets


def split_train_test(applets: List[Dict], test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split applets into train and test sets."""
    random.seed(seed)
    shuffled = applets.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    train = shuffled[:split_idx]
    test = shuffled[split_idx:]

    return train, test


def create_gold_test(applets: List[Dict], n: int = 100) -> List[Dict]:
    """Create gold test set with clear descriptions (>=50 chars)."""
    candidates = [a for a in applets if len(a["query"]) >= 50]
    random.shuffle(candidates)
    return candidates[:n]


def create_noisy_test(applets: List[Dict], n: int = 100) -> List[Dict]:
    """Create noisy test set with vague descriptions (<40 chars)."""
    candidates = [a for a in applets if len(a["query"]) < 40 and len(a["query"]) >= 10]
    random.shuffle(candidates)
    return candidates[:n]


def create_oneshot_test(applets: List[Dict], n: int = 100) -> List[Dict]:
    """Create oneshot test set with rare APIs (<20 occurrences)."""
    # Count API occurrences
    trigger_counts = Counter(a["trigger"]["service_name"] for a in applets)
    action_counts = Counter(a["action"]["service_name"] for a in applets)

    # Find applets with rare APIs
    candidates = []
    for a in applets:
        trigger_name = a["trigger"]["service_name"]
        action_name = a["action"]["service_name"]
        if trigger_counts[trigger_name] < 20 or action_counts[action_name] < 20:
            candidates.append(a)

    random.shuffle(candidates)
    return candidates[:n]


def save_json(data: List[Dict], filepath: Path):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} items to {filepath}")


def main():
    data_dir = Path(__file__).parent

    # Load all applets
    print("Loading applets...")
    applets = load_applets(data_dir / "iftttt_dataset_full_trigger_action.json")
    print(f"Total applets: {len(applets)}")

    # Split into train/test (90/10)
    print("\nSplitting into train/test (90/10)...")
    train_applets, test_applets = split_train_test(applets, test_ratio=0.1, seed=42)
    print(f"Train: {len(train_applets)}")
    print(f"Test:  {len(test_applets)}")

    # Save train set
    save_json(train_applets, data_dir / "train_applets.json")

    # Create test sets from held-out test portion
    print("\nCreating test sets from held-out data...")

    gold = create_gold_test(test_applets, n=100)
    noisy = create_noisy_test(test_applets, n=100)
    oneshot = create_oneshot_test(test_applets, n=100)

    # Save test sets
    save_json(gold, data_dir / "test" / "gold.json")
    save_json(noisy, data_dir / "test" / "noisy.json")
    save_json(oneshot, data_dir / "test" / "oneshot.json")

    # Print summary
    print("\n" + "=" * 60)
    print("DATA SPLIT SUMMARY")
    print("=" * 60)
    print(f"Train applets:     {len(train_applets):,} (for NCE training)")
    print(f"Test gold:         {len(gold)} (clear descriptions)")
    print(f"Test noisy:        {len(noisy)} (vague descriptions)")
    print(f"Test oneshot:      {len(oneshot)} (rare APIs)")
    print(f"Total test:        {len(gold) + len(noisy) + len(oneshot)}")
    print("=" * 60)
    print("\nFiles created:")
    print("  - data/train_applets.json")
    print("  - data/test/gold.json")
    print("  - data/test/noisy.json")
    print("  - data/test/oneshot.json")


if __name__ == "__main__":
    main()

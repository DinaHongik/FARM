"""
Dataset Builder for FARM Contrastive Training

Builds (query, positive_API) pairs for InfoNCE training using applet data.
- Query: applet description (user's natural language query)
- Positive: schema-enriched embedding text of the trigger/action API
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datasets import Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexer import extract_trigger_text, extract_action_text


def load_json(filepath: Path) -> List[Dict[str, Any]]:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_trigger_pairs(
    applets_path: Path,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Build (query, positive) pairs for trigger encoder training.

    Uses applet descriptions as queries and trigger APIs as positives.

    Args:
        applets_path: Path to train_applets.json (flat list format)
        val_split: Fraction for validation set
        seed: Random seed for splitting

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    applets = load_json(applets_path)

    anchors = []
    positives = []

    for applet in applets:
        # Query = applet description (user's natural language)
        query = applet.get("query", "")
        if not query or len(query) < 10:
            continue

        # Get trigger (already extracted in split_data.py)
        trigger = applet.get("trigger")
        if not trigger:
            continue

        # Positive = schema-enriched text (same as indexer)
        positive = extract_trigger_text(trigger)
        if not positive:
            continue

        anchors.append(query)
        positives.append(positive)

    print(f"Built {len(anchors)} trigger training pairs from applets")

    # Create dataset
    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })

    # Split into train/val
    split = dataset.train_test_split(test_size=val_split, seed=seed)

    print(f"  Train: {len(split['train'])} pairs")
    print(f"  Val:   {len(split['test'])} pairs")

    return split['train'], split['test']


def build_action_pairs(
    applets_path: Path,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Build (query, positive) pairs for action encoder training.

    Uses applet descriptions as queries and action APIs as positives.

    Args:
        applets_path: Path to train_applets.json (flat list format)
        val_split: Fraction for validation set
        seed: Random seed for splitting

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    applets = load_json(applets_path)

    anchors = []
    positives = []

    for applet in applets:
        # Query = applet description (user's natural language)
        query = applet.get("query", "")
        if not query or len(query) < 10:
            continue

        # Get action (already extracted in split_data.py)
        action = applet.get("action")
        if not action:
            continue

        # Positive = schema-enriched text (same as indexer)
        positive = extract_action_text(action)
        if not positive:
            continue

        anchors.append(query)
        positives.append(positive)

    print(f"Built {len(anchors)} action training pairs from applets")

    # Create dataset
    dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives
    })

    # Split into train/val
    split = dataset.train_test_split(test_size=val_split, seed=seed)

    print(f"  Train: {len(split['train'])} pairs")
    print(f"  Val:   {len(split['test'])} pairs")

    return split['train'], split['test']


def preview_pairs(dataset: Dataset, n: int = 3, title: str = "Preview"):
    """Preview some training pairs."""
    print(f"\n{title}")
    print("=" * 60)
    for i in range(min(n, len(dataset))):
        print(f"\n[Pair {i+1}]")
        print(f"Anchor (user query):\n  {dataset[i]['anchor'][:100]}...")
        print(f"Positive (API):\n  {dataset[i]['positive'][:100]}...")


if __name__ == "__main__":
    from config import config

    print("Building Trigger Pairs from Applets...")
    train_t, val_t = build_trigger_pairs(config.applets_path, config.val_split, config.seed)
    preview_pairs(train_t, n=2, title="Trigger Training Pairs")

    print("\n" + "=" * 60)
    print("Building Action Pairs from Applets...")
    train_a, val_a = build_action_pairs(config.applets_path, config.val_split, config.seed)
    preview_pairs(train_a, n=2, title="Action Training Pairs")

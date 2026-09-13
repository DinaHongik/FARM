#!/usr/bin/env python3
"""Leakage-free re-implementation of data/split_data.py.

THE BUG IN THE ORIGINAL
    split_data.py:69-79 shuffles applet INSTANCES and slices 90/10. The source
    dataset contains duplicate applets (same query + trigger + action appearing
    more than once), so a single logical applet can be dealt into BOTH sides.
    Measured on the shipped splits: 41/100 gold, 32/100 noisy, 39/100 oneshot
    test records also occur in data/train_applets.json.

THE FIX
    Group records by their identity key first, then assign whole GROUPS to train
    or test. A key can then never span the boundary, by construction.

NON-DESTRUCTIVE
    Writes data/train_applets_dedup.json and data/test_dedup/*.json.
    The originals are left untouched so the contaminated-vs-clean comparison
    (Run A vs Run B) stays possible.

Everything else - the loader, the >=50 / <40 / rare-API filters, n=100, seed=42 -
is kept identical to the original so the only changed variable is the leakage.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
N_PER_SPLIT = 100
TEST_RATIO = 0.1


def load_applets(filepath):
    """Identical to split_data.load_applets - flatten the nested service structure."""
    with open(filepath, encoding="utf-8") as f:
        services = json.load(f)

    applets, skipped = [], 0
    for service in services:
        if not isinstance(service, dict):
            skipped += 1
            continue
        for applet in service.get("applets", []):
            if not isinstance(applet, dict):
                skipped += 1
                continue
            components = applet.get("components", [])
            if not isinstance(components, list):
                skipped += 1
                continue
            trigger = action = None
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
                    "applet_url": applet.get("applet_url", ""),
                })
    if skipped:
        print(f"Skipped {skipped} malformed entries")
    return applets


def key(rec):
    return (
        rec["query"].strip(),
        rec["trigger"].get("service_name", ""),
        rec["action"].get("service_name", ""),
    )


def group_split(applets, test_ratio=TEST_RATIO, seed=SEED):
    """Split by GROUP, not by instance. This is the whole fix."""
    groups = defaultdict(list)
    for a in applets:
        groups[key(a)].append(a)

    keys = sorted(groups.keys())          # sorted first => deterministic under seed
    random.seed(seed)
    random.shuffle(keys)

    split_idx = int(len(keys) * (1 - test_ratio))
    train_keys, test_keys = keys[:split_idx], keys[split_idx:]

    train = [r for k in train_keys for r in groups[k]]
    # one representative per test group - duplicates in test inflate nothing
    test = [groups[k][0] for k in test_keys]
    return train, test, len(applets) - len(groups)


def create_gold(applets, n=N_PER_SPLIT):
    c = [a for a in applets if len(a["query"]) >= 50]
    random.shuffle(c)
    return c[:n]


def create_noisy(applets, n=N_PER_SPLIT):
    c = [a for a in applets if 10 <= len(a["query"]) < 40]
    random.shuffle(c)
    return c[:n]


def create_oneshot(applets, n=N_PER_SPLIT):
    tc = Counter(a["trigger"]["service_name"] for a in applets)
    ac = Counter(a["action"]["service_name"] for a in applets)
    c = [a for a in applets
         if tc[a["trigger"]["service_name"]] < 20 or ac[a["action"]["service_name"]] < 20]
    random.shuffle(c)
    return c[:n]


def save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  saved {len(data):>6} -> {path.relative_to(ROOT)}")


def main():
    src = DATA / "iftttt_dataset_full_trigger_action.json"
    if not src.exists():
        sys.exit(f"missing source dataset: {src}")

    print("Loading applets...")
    applets = load_applets(src)
    print(f"Total applet instances: {len(applets):,}")

    train, test_pool, dup_count = group_split(applets)
    print(f"Duplicate instances collapsed: {dup_count:,}")
    print(f"Unique applets: {len(applets) - dup_count:,}")
    print(f"\nGroup-wise 90/10 split:")
    print(f"  train instances: {len(train):,}")
    print(f"  test  groups:    {len(test_pool):,}")

    random.seed(SEED)
    gold = create_gold(test_pool)
    noisy = create_noisy(test_pool)
    oneshot = create_oneshot(test_pool)

    print("\nWriting (originals untouched):")
    save(train, DATA / "train_applets_dedup.json")
    save(gold, DATA / "test_dedup" / "gold.json")
    save(noisy, DATA / "test_dedup" / "noisy.json")
    save(oneshot, DATA / "test_dedup" / "oneshot.json")

    # ---- self-verify: the whole point is that this prints zero ----
    tk = {key(r) for r in train}
    print("\n" + "=" * 60)
    print("VERIFICATION - leakage into train")
    print("=" * 60)
    bad = 0
    for name, split in (("gold", gold), ("noisy", noisy), ("oneshot", oneshot)):
        leaked = sum(1 for r in split if key(r) in tk)
        bad += leaked
        print(f"  {name:<9} {len(split):>4} records   leaked: {leaked}")
    print("=" * 60)
    if bad:
        print(f"FAILED - {bad} leaked records. Do not train on this.")
        return 1
    print("CLEAN - no test record occurs in training.")
    print("\nNext: point training at data/train_applets_dedup.json and eval at data/test_dedup/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

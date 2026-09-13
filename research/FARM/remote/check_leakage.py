#!/usr/bin/env python3
"""Report train/test leakage for a FARM split.

Why this exists: data/split_data.py shuffles applet INSTANCES and slices 90/10
without deduplicating. The source dataset contains duplicate applets, so the same
logical applet can land in both train and test. This quantifies that.

Usage:
    python check_leakage.py                       # current splits
    python check_leakage.py --train data/train_applets_dedup.json --testdir data/test_dedup
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key(rec):
    """Identity of an applet: what makes two records the same experiment item."""
    return (
        rec.get("query", "").strip(),
        rec.get("trigger", {}).get("service_name", ""),
        rec.get("action", {}).get("service_name", ""),
    )


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_applets.json")
    ap.add_argument("--testdir", default="data/test")
    ap.add_argument("--splits", nargs="+", default=["gold", "noisy", "oneshot"])
    args = ap.parse_args()

    train_path = ROOT / args.train
    test_dir = ROOT / args.testdir
    if not train_path.exists():
        sys.exit(f"missing train file: {train_path}")

    train = load(train_path)
    train_keys = {key(r) for r in train}
    train_queries = {r.get("query", "").strip() for r in train}

    print("=" * 72)
    print(f"LEAKAGE REPORT   train={args.train}   tests={args.testdir}")
    print("=" * 72)
    print(f"train: {len(train):,} records, {len(train_keys):,} unique (query,trigger,action)\n")

    print(f"{'split':<10}{'n':>6}{'leaked':>9}{'pct':>7}   {'query-only':>10}")
    print("-" * 50)

    all_keys = set()
    total_n = total_leak = 0
    for name in args.splits:
        p = test_dir / f"{name}.json"
        if not p.exists():
            print(f"{name:<10}  (missing: {p})")
            continue
        recs = load(p)
        ks = [key(r) for r in recs]
        leaked = [k for k in ks if k in train_keys]
        qleak = sum(1 for r in recs if r.get("query", "").strip() in train_queries)
        all_keys |= set(ks)
        total_n += len(recs)
        total_leak += len(leaked)
        pct = (100.0 * len(leaked) / len(recs)) if recs else 0.0
        print(f"{name:<10}{len(recs):>6}{len(leaked):>9}{pct:>6.0f}%   {qleak:>7}/{len(recs)}")

    print("-" * 50)
    overall = (100.0 * total_leak / total_n) if total_n else 0.0
    print(f"{'TOTAL':<10}{total_n:>6}{total_leak:>9}{overall:>6.0f}%")
    print(f"\nunion of unique test keys: {len(all_keys)}   also in train: {len(all_keys & train_keys)}")

    if total_leak == 0:
        print("\nCLEAN - no test record appears in training.")
        return 0
    print(f"\nLEAKAGE PRESENT: {total_leak}/{total_n} test records were seen during training.")
    print("Any accuracy computed on these splits is inflated. Regenerate with split_data_dedup.py.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

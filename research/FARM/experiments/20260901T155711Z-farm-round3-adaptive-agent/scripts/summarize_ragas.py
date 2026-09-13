#!/usr/bin/env python3
"""Aggregate a completed package-executed Ragas ID parity artifact."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from experiment_core import read_json, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite Ragas summary")
    source = read_json(args.input)
    rows = source["row_scores"]
    if source.get("ragas_version") != "0.4.3" or source.get("rows") != len(rows) or not rows:
        raise RuntimeError("invalid Ragas parity input")
    cutoffs = sorted(rows[0]["cutoffs"], key=int)
    aggregates = {}
    for cutoff in cutoffs:
        keys = sorted(rows[0]["cutoffs"][cutoff])
        aggregates[cutoff] = {
            key: sum(float(row["cutoffs"][cutoff][key]) for row in rows) / len(rows)
            for key in keys
        }
        if not all(math.isfinite(value) for value in aggregates[cutoff].values()):
            raise RuntimeError("non-finite Ragas aggregate")
    write_json(args.output, {
        "status": "completed",
        "dataset_id": source["dataset_id"],
        "split": "dev",
        "rows": len(rows),
        "ragas_version": source["ragas_version"],
        "source_sha256": sha256_file(args.input),
        "aggregates_by_cutoff": aggregates,
        "interpretation": (
            "Supplementary per-side ID-set parity only; exact URL R@k/MRR and exact pair metrics remain primary."
        ),
    })


if __name__ == "__main__":
    main()

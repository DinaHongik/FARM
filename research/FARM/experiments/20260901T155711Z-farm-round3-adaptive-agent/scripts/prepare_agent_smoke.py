#!/usr/bin/env python3
"""Freeze the first deterministic routed dev case for end-to-end agent smoke tests."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from experiment_core import DATASET_ID, load_candidate_artifact, read_json, sha256_file, write_json
from run_agent_experiment import router_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output_manifest.exists():
        raise RuntimeError("refusing to overwrite agent smoke candidates")
    rows, source_manifest = load_candidate_artifact(args.candidates, args.manifest)
    router = read_json(args.router)
    ordered = sorted(rows, key=lambda row: hashlib.sha256(row["group_id"].encode()).hexdigest())
    selected = next(
        (row for row in ordered if router_score(row, router) >= float(router["threshold"])),
        None,
    )
    if selected is None:
        raise RuntimeError("no routed dev row is available for agent smoke")
    write_json(args.output, [selected])
    write_json(args.output_manifest, {
        "status": "passed",
        "dataset_id": DATASET_ID,
        "split": "dev",
        "rows": 1,
        "selection": "lowest sha256(group_id) among recoverability-routed rows",
        "group_id": selected["group_id"],
        "source_candidates_sha256": source_manifest["output_sha256"],
        "router_sha256": sha256_file(args.router),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
    })


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Select a deterministic agent-evaluation subset from a frozen candidate file."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import DATASET_ID, read_json, sha256_file, utc_now, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.input.resolve()
    source_manifest = read_json(args.manifest.resolve())
    if source_manifest.get("dataset_id") != DATASET_ID or source_manifest.get("split") != "dev":
        raise RuntimeError("source candidates are not Dataset v2 dev")
    if source_manifest.get("output_sha256") != sha256_file(source_path):
        raise RuntimeError("source candidate manifest hash mismatch")
    source = read_json(source_path)
    if not 0 < args.rows <= len(source):
        raise ValueError("invalid subset row count")
    selected = sorted(
        source,
        key=lambda row: (hashlib.sha256(row["group_id"].encode()).hexdigest(), row["group_id"]),
    )[: args.rows]
    output = args.output.resolve()
    write_json(output, selected)
    top1 = []
    ceiling = []
    for row in selected:
        valid = {(p["trigger_url"], p["action_url"]) for p in row["valid_pairs"]}
        top1.append(int((row["trigger_candidates"][0]["url"], row["action_candidates"][0]["url"]) in valid))
        trigger_ids = {x["url"] for x in row["trigger_candidates"]}
        action_ids = {x["url"] for x in row["action_candidates"]}
        ceiling.append(int(any(t in trigger_ids and a in action_ids for t, a in valid)))
    manifest = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": DATASET_ID,
        "split": "dev",
        "selection": "lowest SHA-256(group_id), deterministic",
        "rows": len(selected),
        "top_k": source_manifest["top_k"],
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "metrics": {
            "retrieval_joint_R@1": sum(top1) / len(top1),
            f"candidate_joint_R@{source_manifest['top_k']}": sum(ceiling) / len(ceiling),
        },
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

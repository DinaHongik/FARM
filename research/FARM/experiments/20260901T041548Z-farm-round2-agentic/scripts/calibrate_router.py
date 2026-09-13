#!/usr/bin/env python3
"""Freeze an unsupervised low-margin routing threshold on reranker_train."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import DATASET_ID, read_json, sha256_file, utc_now, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--route-fraction", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.route_fraction < 1:
        raise ValueError("route-fraction must be between zero and one")
    path = args.candidates.resolve()
    manifest = read_json(args.manifest.resolve())
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("split") != "reranker_train":
        raise RuntimeError("router calibration must use Dataset v2 reranker_train")
    if manifest.get("output_sha256") != sha256_file(path):
        raise RuntimeError("calibration candidate hash mismatch")
    rows = read_json(path)
    confidences = sorted(float(row["routing_confidence"]) for row in rows)
    if not confidences or not all(math.isfinite(value) for value in confidences):
        raise RuntimeError("invalid calibration confidences")
    target = max(1, math.ceil(len(confidences) * args.route_fraction))
    threshold = confidences[target - 1]
    routed = sum(value <= threshold for value in confidences)
    result = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": DATASET_ID,
        "calibration_split": "reranker_train",
        "supervised_labels_used": False,
        "confidence": "min(trigger top1-top2 score margin, action top1-top2 score margin)",
        "decision": "invoke agent when confidence <= threshold",
        "target_route_fraction": args.route_fraction,
        "threshold": threshold,
        "calibration_rows": len(rows),
        "calibration_routed_rows": routed,
        "calibration_route_fraction": routed / len(rows),
        "candidate_sha256": sha256_file(path),
    }
    write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit a compact recoverability router on reranker_train only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiment_core import DATASET_ID, load_candidate_artifact, sha256_file, write_json


FEATURES = (
    "trigger_top1", "action_top1", "trigger_gap12", "action_gap12",
    "trigger_gap15", "action_gap15", "trigger_entropy5", "action_entropy5",
    "trigger_services5", "action_services5", "trigger_services10", "action_services10",
)


def entropy(values: list[float]) -> float:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    return -sum(value * math.log(max(value, 1e-12)) for value in probabilities)


def feature_row(row: dict) -> list[float]:
    sides = {}
    for side in ("trigger", "action"):
        candidates = row[f"{side}_candidates"]
        values = [float(item["retrieval_score"]) for item in candidates]
        sides[side] = {
            "top1": values[0], "gap12": values[0] - values[1],
            "gap15": values[0] - values[4], "entropy5": entropy(values[:5]),
            "services5": float(len({item["channel"] for item in candidates[:5]})),
            "services10": float(len({item["channel"] for item in candidates[:10]})),
        }
    return [
        sides["trigger"]["top1"], sides["action"]["top1"],
        sides["trigger"]["gap12"], sides["action"]["gap12"],
        sides["trigger"]["gap15"], sides["action"]["gap15"],
        sides["trigger"]["entropy5"], sides["action"]["entropy5"],
        sides["trigger"]["services5"], sides["action"]["services5"],
        sides["trigger"]["services10"], sides["action"]["services10"],
    ]


def targets(row: dict) -> tuple[int, int]:
    valid = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
    top1 = (row["trigger_candidates"][0]["url"], row["action_candidates"][0]["url"])
    trigger_ids = {item["url"] for item in row["trigger_candidates"][:10]}
    action_ids = {item["url"] for item in row["action_candidates"][:10]}
    return int(top1 not in valid), int(any(t in trigger_ids and a in action_ids for t, a in valid))


def stable_holdout(group_id: str) -> bool:
    return int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 5 == 0


def serialize_pipeline(pipeline) -> dict:
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    return {
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if not 0.05 <= args.budget_fraction <= 0.8:
        raise ValueError("budget fraction outside declared range")
    rows, manifest = load_candidate_artifact(args.candidates, args.manifest)
    if manifest["split"] != "reranker_train":
        raise ValueError("router may only fit reranker_train")

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray([feature_row(row) for row in rows], dtype=np.float64)
    wrong, covered = zip(*(targets(row) for row in rows))
    wrong = np.asarray(wrong, dtype=np.int64)
    covered = np.asarray(covered, dtype=np.int64)
    holdout = np.asarray([stable_holdout(row["group_id"]) for row in rows], dtype=bool)
    train = ~holdout

    def create() -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=2000, random_state=42)),
        ])

    calibration = {}
    validation_probabilities = {}
    final_models = {}
    for label, target in (("wrong", wrong), ("covered", covered)):
        model = create().fit(x[train], target[train])
        probabilities = model.predict_proba(x[holdout])[:, 1]
        validation_probabilities[label] = probabilities
        calibration[label] = {
            "rows": int(holdout.sum()), "positive_rate": float(target[holdout].mean()),
            "roc_auc": float(roc_auc_score(target[holdout], probabilities)),
            "brier": float(brier_score_loss(target[holdout], probabilities)),
        }
        final_models[label] = serialize_pipeline(create().fit(x, target))
    recoverability = validation_probabilities["wrong"] * validation_probabilities["covered"]
    order = sorted(float(value) for value in recoverability)
    threshold_index = max(0, math.ceil((1.0 - args.budget_fraction) * len(order)) - 1)
    threshold = order[threshold_index]
    artifact = {
        "status": "passed", "dataset_id": DATASET_ID, "calibration_split": "reranker_train",
        "candidate_sha256": manifest["output_sha256"], "script_sha256": sha256_file(Path(__file__)),
        "feature_names": list(FEATURES), "models": final_models,
        "routing_score": "P(non-agentic top1 wrong) * P(valid pair in top10)",
        "threshold": threshold, "budget_fraction": args.budget_fraction,
        "holdout_policy": "sha256(group_id) mod 5 == 0", "calibration": calibration,
        "holdout_routed_fraction": float((recoverability >= threshold).mean()),
    }
    write_json(args.output, artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

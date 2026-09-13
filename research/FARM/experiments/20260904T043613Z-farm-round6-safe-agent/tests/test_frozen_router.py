from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from frozen_router import FEATURE_NAMES, routing_score, validate_router  # noqa: E402


def artifact():
    model = {
        "mean": [0.0] * len(FEATURE_NAMES), "scale": [1.0] * len(FEATURE_NAMES),
        "coef": [0.0] * len(FEATURE_NAMES), "intercept": 0.0,
    }
    return {
        "feature_names": list(FEATURE_NAMES), "models": {"wrong": model, "covered": model},
        "threshold": .3, "dataset_id": "x", "calibration_split": "reranker_train",
        "candidate_sha256": "abc",
    }


def row():
    return {
        side + "_candidates": [
            {"retrieval_score": 10 - i, "channel": f"s{i % 2}"} for i in range(10)
        ]
        for side in ("trigger", "action")
    }


def test_router_binding_and_score():
    value = validate_router(artifact(), candidate_sha256="abc")
    assert routing_score(row(), value) == .25


def test_router_rejects_feature_reordering():
    value = artifact()
    value["feature_names"] = list(reversed(FEATURE_NAMES))
    try:
        validate_router(value, candidate_sha256="abc")
    except ValueError as error:
        assert "feature order" in str(error)
    else:
        raise AssertionError("reordered router was accepted")


def test_router_rejects_wrong_candidate_hash():
    try:
        validate_router(artifact(), candidate_sha256="different")
    except ValueError as error:
        assert "candidate binding" in str(error)
    else:
        raise AssertionError("wrong candidate hash was accepted")

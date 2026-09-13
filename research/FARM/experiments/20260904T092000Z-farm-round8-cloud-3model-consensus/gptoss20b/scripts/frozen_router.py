#!/usr/bin/env python3
"""Pure inference for the hash-pinned Round3/Round6 recoverability router."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


FEATURE_NAMES = (
    "trigger_top1",
    "action_top1",
    "trigger_gap12",
    "action_gap12",
    "trigger_gap15",
    "action_gap15",
    "trigger_entropy5",
    "action_entropy5",
    "trigger_services5",
    "action_services5",
    "trigger_services10",
    "action_services10",
)


def entropy(values: Sequence[float]) -> float:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return -sum(
        (weight / total) * math.log(max(weight / total, 1e-12))
        for weight in weights
    )


def feature_row(row: Mapping[str, Any]) -> list[float]:
    sides: dict[str, dict[str, float]] = {}
    for side in ("trigger", "action"):
        candidates = row[f"{side}_candidates"]
        if len(candidates) < 10:
            raise ValueError("router requires ten candidates per endpoint")
        values = [float(item["retrieval_score"]) for item in candidates]
        sides[side] = {
            "top1": values[0],
            "gap12": values[0] - values[1],
            "gap15": values[0] - values[4],
            "entropy5": entropy(values[:5]),
            "services5": float(
                len({item["channel"] for item in candidates[:5]})
            ),
            "services10": float(
                len({item["channel"] for item in candidates[:10]})
            ),
        }
    return [
        sides["trigger"]["top1"],
        sides["action"]["top1"],
        sides["trigger"]["gap12"],
        sides["action"]["gap12"],
        sides["trigger"]["gap15"],
        sides["action"]["gap15"],
        sides["trigger"]["entropy5"],
        sides["action"]["entropy5"],
        sides["trigger"]["services5"],
        sides["action"]["services5"],
        sides["trigger"]["services10"],
        sides["action"]["services10"],
    ]


def validate_router(
    value: Mapping[str, Any], *, candidate_sha256: str
) -> dict[str, Any]:
    required = {
        "feature_names",
        "models",
        "threshold",
        "dataset_id",
        "calibration_split",
        "candidate_sha256",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"router artifact lacks {sorted(missing)}")
    if tuple(value["feature_names"]) != FEATURE_NAMES:
        raise ValueError("router feature order changed")
    if value["calibration_split"] != "reranker_train":
        raise ValueError("router was not fit on reranker_train")
    if value["candidate_sha256"] != candidate_sha256:
        raise ValueError("router candidate binding changed")
    if not 0.0 < float(value["threshold"]) < 1.0:
        raise ValueError("router threshold is invalid")
    for label in ("wrong", "covered"):
        model = value["models"].get(label)
        if not isinstance(model, Mapping) or not all(
            len(model[key]) == len(FEATURE_NAMES)
            for key in ("mean", "scale", "coef")
        ):
            raise ValueError(f"invalid router model {label}")
    return dict(value)


def probability(features: Sequence[float], model: Mapping[str, Any]) -> float:
    standardized = [
        (value - float(mean)) / float(scale) if float(scale) else 0.0
        for value, mean, scale in zip(
            features, model["mean"], model["scale"]
        )
    ]
    logit = float(model["intercept"]) + sum(
        value * float(weight)
        for value, weight in zip(standardized, model["coef"])
    )
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


def routing_score(row: Mapping[str, Any], router: Mapping[str, Any]) -> float:
    features = feature_row(row)
    return probability(features, router["models"]["wrong"]) * probability(
        features, router["models"]["covered"]
    )


__all__ = [
    "FEATURE_NAMES",
    "feature_row",
    "routing_score",
    "validate_router",
]

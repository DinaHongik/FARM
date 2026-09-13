#!/usr/bin/env python3
"""Contracts for the deterministic selective-router replay."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))

from analyze_selective_router import (  # noqa: E402
    FEATURE_NAMES,
    extract_router,
    feature_row,
    probability,
)


def _row() -> dict:
    def candidates(prefix: str) -> list[dict]:
        return [
            {
                "url": f"{prefix}-{rank}",
                "channel": f"service-{(rank - 1) // 2}",
                "retrieval_rank": rank,
                "retrieval_score": 1.0 / rank,
            }
            for rank in range(1, 11)
        ]

    return {
        "trigger_candidates": candidates("trigger"),
        "action_candidates": candidates("action"),
    }


class SelectiveRouterAnalysisTests(unittest.TestCase):
    def test_feature_contract_preserves_frozen_round3_order(self) -> None:
        features = feature_row(_row())

        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertEqual(features[:6], [1.0, 1.0, 0.5, 0.5, 0.8, 0.8])
        self.assertEqual(features[8:], [3.0, 3.0, 5.0, 5.0])
        self.assertTrue(all(math.isfinite(value) for value in features))

    def test_probability_uses_serialized_standardization_and_logit(self) -> None:
        model = {
            "mean": [1.0, 2.0],
            "scale": [2.0, 4.0],
            "coef": [2.0, -4.0],
            "intercept": 0.0,
        }

        # Both standardized values are 1, so the logit is -2.
        self.assertAlmostEqual(probability([3.0, 6.0], model), 1.0 / (1.0 + math.exp(2.0)))
        with self.assertRaisesRegex(ValueError, "width"):
            probability([3.0], model)

    def test_only_a_reranker_train_router_with_exact_features_is_accepted(self) -> None:
        router = {
            "feature_names": list(FEATURE_NAMES),
            "models": {"wrong": {}, "covered": {}},
            "threshold": 0.3,
            "dataset_id": "dataset-v2",
            "calibration_split": "reranker_train",
        }

        self.assertIs(extract_router({"router": router}), router)
        with self.assertRaisesRegex(ValueError, "feature contract"):
            extract_router({"router": router | {"feature_names": list(reversed(FEATURE_NAMES))}})
        with self.assertRaisesRegex(ValueError, "reranker_train"):
            extract_router({"router": router | {"calibration_split": "dev"}})


if __name__ == "__main__":
    unittest.main()

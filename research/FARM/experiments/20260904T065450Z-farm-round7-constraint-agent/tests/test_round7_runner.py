#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
FARM_ROOT = RUN_ROOT.parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))

import run_round7_agent as runner  # noqa: E402


class RunnerTests(unittest.TestCase):
    def test_frozen_discovery_and_confirmation_identities(self) -> None:
        dev_split = json.loads((FARM_ROOT / "data/v2/splits/dev.json").read_text())
        candidates = json.loads((
            FARM_ROOT
            / "experiments/20260901T155711Z-farm-round3-adaptive-agent"
            / "derived_data/function_rrf/dev.json"
        ).read_text())
        selected, metadata = runner.select_discovery_rows(dev_split, candidates)
        self.assertEqual(231, len(selected))
        self.assertEqual(runner.SCREEN_HASH, metadata["selected_group_ids_sha256"])
        self.assertEqual(runner.CONFIRMATION_HASH, metadata["reserved_confirmation_group_ids_sha256"])
        self.assertFalse(metadata["reserved_payload_sent_to_model"])

    def test_exact_mcnemar_uses_only_discordant_pairs(self) -> None:
        result = runner.exact_mcnemar([0, 0, 1, 1], [1, 1, 0, 1])
        self.assertEqual(2, result["recoveries"])
        self.assertEqual(1, result["regressions"])
        self.assertEqual(3, result["discordant"])
        self.assertEqual(1, result["net"])

    def test_dataset_manifest_is_bound_to_preregistered_hash(self) -> None:
        data_root = FARM_ROOT / "data/v2"
        expected = runner.sha256_file(data_root / "manifest.json")
        self.assertEqual(
            data_root / "manifest.json",
            runner.verify_pinned_data_manifest(data_root, expected),
        )
        with self.assertRaisesRegex(ValueError, "data_manifest"):
            runner.verify_pinned_data_manifest(data_root, "0" * 64)

    def test_family_bootstrap_is_deterministic(self) -> None:
        records = [
            {
                "semantic_family_id": f"f{i}",
                "baseline_score": {"function_joint": before},
                "final_score": {"function_joint": after},
            }
            for i, (before, after) in enumerate(((0, 1), (1, 1), (1, 0), (0, 1)))
        ]
        first = runner.family_bootstrap(records, iterations=100, seed=7)
        second = runner.family_bootstrap(records, iterations=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(0.25, first["delta"])

    def test_oracle_reporting_separates_system_and_routed_agent(self) -> None:
        records = [
            {
                "candidate_oracle": {"function_joint": True},
                "routed": True,
                "protocol_valid": True,
                "proposal_score": {"function_joint": True},
                "final_score": {"function_joint": True},
            },
            {
                "candidate_oracle": {"function_joint": True},
                "routed": True,
                "protocol_valid": False,
                "proposal_score": None,
                "final_score": {"function_joint": False},
            },
            {
                "candidate_oracle": {"function_joint": True},
                "routed": False,
                "protocol_valid": True,
                "proposal_score": None,
                "final_score": {"function_joint": True},
            },
            {
                "candidate_oracle": {"function_joint": False},
                "routed": True,
                "protocol_valid": True,
                "proposal_score": {"function_joint": False},
                "final_score": {"function_joint": False},
            },
        ]
        system, agent = runner.conditional_oracle_metrics(
            records, ("function_joint",)
        )
        self.assertEqual(
            {"eligible": 3, "final_hits": 2, "final_accuracy": 2 / 3},
            system["function_joint"],
        )
        self.assertEqual(2, agent["function_joint"]["eligible_routed"])
        self.assertEqual(1, agent["function_joint"]["protocol_valid"])
        self.assertEqual(1, agent["function_joint"]["proposal_available"])
        self.assertEqual(1.0, agent["function_joint"]["proposal_accuracy_when_available"])
        self.assertEqual(1, agent["function_joint"]["final_hits"])
        self.assertEqual(0.5, agent["function_joint"]["final_accuracy"])

    def test_override_metrics_separate_discordant_precision_and_all_changes(self) -> None:
        def record(retained: bool, before: bool, after: bool) -> dict:
            return {
                "retained_baseline": retained,
                "baseline_score": {"function_joint": before},
                "final_score": {"function_joint": after},
            }

        metrics = runner.accepted_override_metrics([
            record(True, False, False),
            record(False, False, True),
            record(False, True, False),
            record(False, False, False),
            record(False, True, True),
        ])
        self.assertEqual(4, metrics["changed_cases"])
        self.assertEqual(0.5, metrics["accepted_override_precision"])
        self.assertEqual("recoveries_plus_regressions", metrics["accepted_override_precision_denominator"])
        self.assertEqual(2, metrics["changed_final_correct"])
        self.assertEqual(0.5, metrics["changed_final_correct_rate"])
        self.assertEqual(2, metrics["neutral_changed_cases"])

    def test_protocol_cost_is_broken_down_by_baseline_service(self) -> None:
        def record(service: str, routed: bool, calls: int, tokens: int) -> dict:
            return {
                "baseline_pair": {
                    "trigger_service": service,
                    "action_service": "shared-action",
                },
                "routed": routed,
                "accounting": {
                    "logical_model_calls": calls,
                    "usage": {"total_tokens": tokens},
                },
            }

        metrics = runner.protocol_by_baseline_service([
            record("zeta", True, 2, 120),
            record("alpha", False, 0, 0),
            record("zeta", False, 0, 0),
        ], "trigger")
        self.assertEqual(["alpha", "zeta"], list(metrics))
        self.assertEqual(2, metrics["zeta"]["cases"])
        self.assertEqual(1, metrics["zeta"]["routed_cases"])
        self.assertEqual(2, metrics["zeta"]["logical_model_calls"])
        self.assertEqual(1.0, metrics["zeta"]["logical_model_calls_per_case"])
        self.assertEqual(60.0, metrics["zeta"]["tokens_per_case"])
        self.assertIsNone(metrics["alpha"]["logical_model_calls_per_routed_case"])


if __name__ == "__main__":
    unittest.main()

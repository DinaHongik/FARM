from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_round7 import aggregate, holm_adjust  # noqa: E402


RUN_ID = "20260904T065450Z-farm-round7-constraint-agent"
DATASET_ID = "synthetic-dataset-v2"
ARMS = ["plain", "schema", "single", "dual"]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def matrix() -> dict:
    return {
        "run_id": RUN_ID,
        "dataset_id": DATASET_ID,
        "agent_discovery_arms": [{"id": name, "scope": 10} for name in ARMS],
        "gpu_router_folds": [
            {"id": f"router{fold}", "holdout_fold": fold} for fold in range(4)
        ],
        "discovery_gates": {
            "minimum_absolute_delta": 0.03,
            "minimum_baseline_correct_retention": 0.97,
            "minimum_override_precision": 0.70,
            "maximum_logical_model_calls_per_case": 0.75,
            "maximum_tokens_per_case": 1200,
            "minimum_protocol_valid_rate": 0.99,
        },
    }


def agent_result(name: str, p_value: float) -> dict:
    return {
        "status": "completed",
        "binding": {"arm": name, "run_id": RUN_ID, "dataset_id": DATASET_ID},
        "metrics": {
            "rows": 10,
            "function_joint": {
                "baseline_hits": 5,
                "baseline_rate": 0.5,
                "agent_hits": 6,
                "agent_rate": 0.6,
                "absolute_delta": 0.1,
            },
            "primary_comparison": {
                "recoveries": 2,
                "regressions": 1,
                "net": 1,
                "exact_two_sided_p": p_value,
                "accepted_override_precision": 0.8,
                "baseline_correct_retention": 0.98,
                "family_cluster_paired_bootstrap": {
                    "delta": 0.1,
                    "ci95": [0.01, 0.2],
                },
            },
            "protocol": {
                "logical_calls_per_case": 0.7,
                "tokens_per_case": 1000,
                "valid_rows": 10,
            },
        },
    }


def router_result(name: str, fold: int, rows: int) -> dict:
    binary = {"precision": 0.7, "recall": 0.6, "f1": 0.64}
    return {
        "status": "completed",
        "binding": {
            "experiment_id": name,
            "run_id": RUN_ID,
            "dataset_id": DATASET_ID,
            "config": {"holdout_fold": fold},
        },
        "metrics": {
            "rows": rows,
            "operational_four_class": {
                "accuracy": 0.7,
                "balanced_accuracy": 0.6,
                "macro_f1": 0.65,
            },
            "routing": binary,
            "side_routing": {"trigger": binary, "action": binary},
        },
    }


class AggregateRound7Tests(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_step_down(self) -> None:
        adjusted = holm_adjust([("a", 0.01), ("b", 0.04), ("c", 0.03), ("d", 0.2)])
        self.assertAlmostEqual(adjusted["a"]["holm_adjusted_p"], 0.04)
        self.assertAlmostEqual(adjusted["c"]["holm_adjusted_p"], 0.09)
        self.assertAlmostEqual(adjusted["b"]["holm_adjusted_p"], 0.09)
        self.assertAlmostEqual(adjusted["d"]["holm_adjusted_p"], 0.2)
        self.assertTrue(adjusted["a"]["reject_at_0_05"])
        self.assertFalse(adjusted["c"]["reject_at_0_05"])
        self.assertFalse(adjusted["b"]["reject_at_0_05"])

    def test_complete_matrix_uses_only_expected_result_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / RUN_ID
            write_json(root / "EXPERIMENT_MATRIX.json", matrix())
            for name, p_value in zip(ARMS, (0.01, 0.02, 0.03, 0.04)):
                write_json(root / "results" / f"{name}.json", agent_result(name, p_value))
            for fold, rows in enumerate((260, 306, 271, 308)):
                name = f"router{fold}"
                write_json(root / "results" / f"{name}.json", router_result(name, fold, rows))
            # These files must not be discovered or opened by the aggregator.
            (root / "records").mkdir()
            (root / "records" / "reserved.jsonl").write_text("not-json", encoding="utf-8")
            (root / "predictions").mkdir()
            (root / "predictions" / "hidden.jsonl").write_text("not-json", encoding="utf-8")

            summary = aggregate(root)

            self.assertEqual(summary["agent_discovery"]["completed_arms"], 4)
            self.assertEqual(summary["agent_discovery"]["multiplicity"]["status"], "complete")
            self.assertTrue(summary["side_router_cross_validation"]["covers_all_1145_rows"])
            self.assertEqual(summary["side_router_cross_validation"]["heldout_rows_observed"], 1145)
            self.assertFalse(summary["input_policy"]["records_read"])
            self.assertFalse(summary["input_policy"]["predictions_read"])

    def test_partial_matrix_does_not_shrink_holm_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / RUN_ID
            write_json(root / "EXPERIMENT_MATRIX.json", matrix())
            write_json(root / "results" / "plain.json", agent_result("plain", 0.001))

            summary = aggregate(root)

            multiplicity = summary["agent_discovery"]["multiplicity"]
            self.assertEqual(multiplicity["status"], "pending_all_four_preregistered_arms")
            self.assertEqual(multiplicity["family_size"], 4)
            self.assertIsNone(multiplicity["by_arm"])
            self.assertIsNone(
                summary["agent_discovery"]["arms"]["plain"]["all_discovery_gates_pass"]
            )

    def test_undefined_override_precision_is_reported_as_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / RUN_ID
            write_json(root / "EXPERIMENT_MATRIX.json", matrix())
            value = agent_result("plain", 1.0)
            value["metrics"]["primary_comparison"]["accepted_override_precision"] = None
            write_json(root / "results" / "plain.json", value)

            summary = aggregate(root)

            arm = summary["agent_discovery"]["arms"]["plain"]
            self.assertIsNone(arm["observed"]["accepted_override_precision"])
            self.assertFalse(arm["gate_checks"]["accepted_override_precision"]["pass"])

    def test_undefined_retention_is_reported_as_failed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / RUN_ID
            write_json(root / "EXPERIMENT_MATRIX.json", matrix())
            value = agent_result("plain", 1.0)
            value["metrics"]["primary_comparison"]["baseline_correct_retention"] = None
            write_json(root / "results" / "plain.json", value)

            summary = aggregate(root)

            arm = summary["agent_discovery"]["arms"]["plain"]
            self.assertIsNone(arm["observed"]["baseline_correct_retention"])
            self.assertFalse(
                arm["gate_checks"]["baseline_correct_retention"]["pass"]
            )

    def test_binding_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / RUN_ID
            write_json(root / "EXPERIMENT_MATRIX.json", matrix())
            value = agent_result("wrong-arm", 0.01)
            write_json(root / "results" / "plain.json", value)
            with self.assertRaisesRegex(ValueError, "binding arm mismatch"):
                aggregate(root)


if __name__ == "__main__":
    unittest.main()

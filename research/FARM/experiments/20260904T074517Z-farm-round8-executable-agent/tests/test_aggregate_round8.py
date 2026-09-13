from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate_round8 as aggregate  # noqa: E402


OUTCOMES = (
    "function_trigger",
    "function_action",
    "function_joint",
    "service_trigger",
    "service_action",
    "service_joint",
)


def _scores(joint: bool, *, trigger: bool | None = None, action: bool | None = None) -> dict[str, bool]:
    trigger = joint if trigger is None else trigger
    action = joint if action is None else action
    return {
        "function_trigger": trigger,
        "function_action": action,
        "function_joint": joint,
        "service_trigger": trigger,
        "service_action": action,
        "service_joint": joint,
    }


def _record(
    arm: str,
    group: str,
    family: str,
    *,
    baseline: bool,
    proposal: bool | None,
    final: bool,
    oracle: bool,
    provider_requests: int = 1,
    repair_attempted: bool = False,
    repair_success: bool = False,
) -> dict:
    return {
        "schema_version": "farm_round8_executable_record_v1",
        "group_id": group,
        "semantic_family_id": family,
        "arm_id": arm,
        "baseline_score": _scores(baseline),
        "proposal_score": None if proposal is None else _scores(proposal),
        "final_score": _scores(final),
        "candidate_oracle": _scores(oracle),
        "execution": {
            "strict_parse": True,
            "compiled": True,
            "sandbox_run": True,
            "repair_attempted": repair_attempted,
            "repair_success": repair_success,
        },
        "calls": [
            {
                "role": "planner",
                "ok": True,
                "schema_valid": True,
                "provider_requests": provider_requests,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "latency_seconds": 0.1,
            }
        ],
        "accounting": {
            "semantic_calls": 1,
            "transport_attempts": provider_requests,
            "model_tool_calls": 1,
            "provider_requests": provider_requests,
            "deterministic_tool_calls": 2,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "latency_seconds": 0.1,
            "failures": 0,
        },
    }


def _write_completed_arm(root: Path, arm: str, records: list[dict], journal: list[dict]) -> None:
    for directory in ("records", "attempts", "results", "progress"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    records_path = root / "records" / f"{arm}.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    attempts_path = root / "attempts" / f"{arm}.jsonl"
    attempts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in journal),
        encoding="utf-8",
    )
    binding = {"arm": arm, "selection": {"executed_rows": len(records)}}
    result_path = root / "results" / f"{arm}.json"
    result_path.write_text(
        json.dumps({"status": "completed", "binding": binding, "metrics": {"rows": len(records)}}),
        encoding="utf-8",
    )
    result_digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    (root / "progress" / f"{arm}.json").write_text(
        json.dumps(
            {
                "phase": "complete",
                "binding": binding,
                "completed_rows": len(records),
                "target_rows": len(records),
                "output": str(result_path),
                "output_sha256": result_digest,
            }
        ),
        encoding="utf-8",
    )


def _event(event: str, key: str, **extra: object) -> dict:
    return {
        "event": event,
        "semantic_id": key,
        "role": "planner",
        "request_hash": f"hash-{key}",
        "attempt": 1,
        **extra,
    }


class Round8AggregateTests(unittest.TestCase):
    def test_cli_writes_reviewer_metrics_and_separates_two_cost_sources(self) -> None:
        arms = ("arm_a", "arm_b")
        layouts = {
            "arm_a": [
                ("g1", "f1", True, True, True, True),
                ("g2", "f1", False, None, True, True),
                ("g3", "f2", False, True, False, True),
                ("g4", "f2", True, False, True, False),
            ],
            "arm_b": [
                ("g1", "f1", True, True, False, True),
                ("g2", "f1", False, False, False, True),
                ("g3", "f2", False, True, True, True),
                ("g4", "f2", True, None, True, False),
            ],
        }
        journal = [
            _event("request_started", "j1"),
            _event("request_finished", "j1", outcome="valid_tool_call", prompt_tokens=10, completion_tokens=2, model_tool_calls=1, latency_seconds=0.1),
            _event("request_started", "j2"),
            _event("request_started", "j2"),
            _event("request_finished", "j2", outcome="transport_error", prompt_tokens=0, completion_tokens=0, model_tool_calls=0, latency_seconds=0.2),
            _event("request_finished", "j3", outcome="valid_tool_call", prompt_tokens=5, completion_tokens=1, model_tool_calls=1, latency_seconds=0.3),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for arm in arms:
                records = [
                    _record(
                        arm,
                        group,
                        family,
                        baseline=baseline,
                        proposal=proposal,
                        final=final,
                        oracle=oracle,
                    )
                    for group, family, baseline, proposal, final, oracle in layouts[arm]
                ]
                _write_completed_arm(root, arm, records, journal)

            self.assertEqual(
                0,
                aggregate.main(
                    [
                        "--artifact-root", str(root),
                        "--arms", *arms,
                        "--bootstrap-iterations", "200",
                        "--seed", "7",
                    ]
                ),
            )
            payload = json.loads((root / "aggregate" / "round8_aggregate.json").read_text())
            markdown = (root / "aggregate" / "round8_aggregate.md").read_text()

        arm_a = payload["arms"]["arm_a"]
        proposal = arm_a["stages"]["proposal"]["function_joint"]
        self.assertEqual(3, proposal["available"])
        self.assertEqual(4, proposal["intent_to_treat_denominator"])
        self.assertEqual(0.5, proposal["intent_to_treat_accuracy"])
        self.assertAlmostEqual(2 / 3, proposal["available_case_accuracy"])
        rescue = arm_a["baseline_wrong_joint_oracle"]
        self.assertEqual(2, rescue["denominator"])
        self.assertEqual(0.5, rescue["final_rescue_accuracy"])
        self.assertEqual({"numerator": 0, "denominator": 0, "rate": None}, arm_a["execution"]["repair_success"])
        comparison = arm_a["paired_comparisons"]["function_joint"]
        self.assertEqual(1, comparison["mcnemar"]["rescues"])
        self.assertEqual(0, comparison["mcnemar"]["regressions"])
        self.assertEqual(0.25, comparison["delta"])
        self.assertEqual(200, comparison["family_cluster_bootstrap_95_ci"]["iterations"])
        self.assertIn("holm_adjusted_p", payload["multiplicity"]["function_joint"]["arm_a"])
        operational = arm_a["cost"]["journal_observed"]
        self.assertEqual(3, operational["provider_attempts_observed"])
        self.assertEqual(1, operational["duplicate_start_events"])
        self.assertEqual(1, operational["unfinished_start_events"])
        self.assertEqual(1, operational["orphan_finish_events"])
        self.assertEqual(4, arm_a["cost"]["record_attributed"]["provider_requests"])
        self.assertIn("trigger", arm_a["sides"])
        self.assertIn("planner", arm_a["model_stages"])
        self.assertIn("Intent-to-treat", markdown)
        self.assertIn("not interchangeable", markdown)

    def test_completed_arm_binding_must_identify_the_requested_arm(self) -> None:
        arm = "arm_a"
        records = [
            _record(
                arm, "g1", "f1", baseline=False, proposal=True, final=True, oracle=True
            )
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_completed_arm(root, arm, records, [])
            result_path = root / "results" / f"{arm}.json"
            progress_path = root / "progress" / f"{arm}.json"
            result = json.loads(result_path.read_text())
            progress = json.loads(progress_path.read_text())
            result["binding"]["arm"] = "different_arm"
            progress["binding"]["arm"] = "different_arm"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            progress["output_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            progress_path.write_text(json.dumps(progress), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "binding arm"):
                aggregate.load_completed_arm(root, arm)

    def test_aggregate_refuses_group_family_misalignment(self) -> None:
        first = _record(
            "arm_a", "g1", "family-one", baseline=False, proposal=True, final=True, oracle=True
        )
        second = _record(
            "arm_b", "g1", "family-two", baseline=False, proposal=True, final=True, oracle=True
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_completed_arm(root, "arm_a", [first], [])
            _write_completed_arm(root, "arm_b", [second], [])
            with self.assertRaisesRegex(RuntimeError, "group/family alignment"):
                aggregate.aggregate_artifacts(
                    root, ["arm_a", "arm_b"], bootstrap_iterations=10, seed=7
                )

    def test_exact_statistics_and_fixed_family_bootstrap_are_reproducible(self) -> None:
        comparison = aggregate.exact_mcnemar(
            [True, True, False, False, False, False, False, False],
            [True, False, True, True, True, True, True, True],
        )
        self.assertEqual(6, comparison["rescues"])
        self.assertEqual(1, comparison["regressions"])
        self.assertEqual(0.125, comparison["exact_two_sided_p"])
        adjusted = aggregate.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(0.03, adjusted["a"]["holm_adjusted_p"])
        self.assertEqual(0.06, adjusted["b"]["holm_adjusted_p"])
        self.assertEqual(0.06, adjusted["c"]["holm_adjusted_p"])

        records = [
            _record("arm", "g1", "f1", baseline=False, proposal=True, final=True, oracle=True),
            _record("arm", "g2", "f1", baseline=False, proposal=True, final=True, oracle=True),
            _record("arm", "g3", "f2", baseline=True, proposal=False, final=False, oracle=True),
        ]
        first = aggregate.family_cluster_bootstrap(
            records, "function_joint", iterations=100, seed=11
        )
        second = aggregate.family_cluster_bootstrap(
            records, "function_joint", iterations=100, seed=11
        )
        self.assertEqual(first, second)
        self.assertEqual(2, first["clusters"])


if __name__ == "__main__":
    unittest.main()

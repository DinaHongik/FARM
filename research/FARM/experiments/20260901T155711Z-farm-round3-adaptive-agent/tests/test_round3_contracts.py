#!/usr/bin/env python3
"""Behavioral contract tests for the isolated FARM round-three experiment."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = RUN_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))


class FakeChatClient:
    """External Ollama boundary fake; application modules remain real."""

    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def select_pair(self, request: dict) -> dict:
        self.requests.append(request)
        return self.replies.pop(0)


class ToolOmittingOllamaClient:
    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs) -> dict:
        self.calls += 1
        return {
            "message": {"content": "I cannot decide."},
            "prompt_eval_count": 10,
            "eval_count": 3,
            "total_duration": 5,
        }


def candidate(side: str, rank: int) -> dict:
    return {
        "url": f"https://ifttt.com/{side}{rank}/{side}s/f{rank}",
        "channel": f"{side}{rank}",
        "function_name": f"{side.title()} {rank}",
        "text_plain": f"{side} function {rank}",
        "text_schema": f"{side} function {rank}\nfield: value{rank}",
        "retrieval_rank": rank,
        "retrieval_score": 1.0 - rank / 100.0,
    }


def case_without_gold() -> dict:
    return {
        "group_id": "q_known",
        "query": "When trigger two happens, perform action seven",
        "trigger_candidates": [candidate("trigger", i) for i in range(1, 11)],
        "action_candidates": [candidate("action", i) for i in range(1, 11)],
        "routing_confidence": 0.01,
    }


class DatasetContractTests(unittest.TestCase):
    def test_candidate_loader_refuses_locked_test_without_opening_payload(self) -> None:
        from experiment_core import load_candidate_artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "must_not_open.json"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"dataset_id": "toy", "split": "test"}))
            with self.assertRaisesRegex(ValueError, "locked test"):
                load_candidate_artifact(payload, manifest, expected_dataset_id="toy")

    def test_candidate_loader_accepts_hash_bound_dev_rows(self) -> None:
        from experiment_core import load_candidate_artifact, sha256_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "dev.json"
            manifest = root / "manifest.json"
            payload.write_text(json.dumps([case_without_gold()]))
            manifest.write_text(json.dumps({
                "dataset_id": "toy", "split": "dev",
                "output_sha256": sha256_file(payload),
            }))
            rows, _ = load_candidate_artifact(payload, manifest, expected_dataset_id="toy")
            self.assertEqual(rows[0]["group_id"], "q_known")


class RerankerContractTests(unittest.TestCase):
    def test_pair_serialization_preserves_both_sides_under_long_input(self) -> None:
        from experiment_core import serialize_pair_document

        value = serialize_pair_document("trigger " * 1000, "ACTION_SENTINEL " + "action " * 1000, per_side_chars=120)
        self.assertEqual((value.startswith("TRIGGER:"), "ACTION_SENTINEL" in value), (True, True))

    def test_rrf_fusion_is_deterministic_and_url_tie_broken(self) -> None:
        from build_rrf_lattice import rrf_fuse

        rankings = [["b", "a", "c"], ["a", "b", "d"]]
        first = rrf_fuse(rankings, constant=60)
        second = rrf_fuse(list(reversed(rankings)), constant=60)
        self.assertEqual(first, second)

    def test_pair_training_uses_observed_multigold_pairs(self) -> None:
        from experiment_core import make_pair_training_group

        row = case_without_gold() | {
            "valid_pairs": [
                {"trigger_url": candidate("trigger", 2)["url"],
                 "action_url": candidate("action", 7)["url"]},
                {"trigger_url": candidate("trigger", 3)["url"],
                 "action_url": candidate("action", 8)["url"]},
            ]
        }
        group = make_pair_training_group(row, view="schema", depth=10)
        positives = {doc["pair_id"] for doc in group["documents"] if doc["label"] == 1.0}
        self.assertEqual(positives, {
            candidate("trigger", 2)["url"] + " || " + candidate("action", 7)["url"],
            candidate("trigger", 3)["url"] + " || " + candidate("action", 8)["url"],
        })

    def test_independent_metrics_keep_trigger_action_and_joint_units_separate(self) -> None:
        from experiment_core import independent_rank_metrics

        rows = [
            {"trigger_rank": 1, "action_rank": 2},
            {"trigger_rank": 3, "action_rank": None},
        ]
        metrics = independent_rank_metrics(rows, cutoffs=(1, 5))
        self.assertEqual(
            {k: metrics[k] for k in ("trigger_R@1", "action_R@1", "joint_independent_R@5")},
            {"trigger_R@1": 0.5, "action_R@1": 0.0, "joint_independent_R@5": 0.5},
        )

    def test_pair_metrics_are_ranked_pair_metrics_not_cartesian_recall(self) -> None:
        from experiment_core import pair_rank_metrics

        metrics = pair_rank_metrics([{"pair_rank": 2}, {"pair_rank": None}], cutoffs=(1, 5))
        self.assertEqual(metrics, {"pair_R@1": 0.0, "pair_R@5": 0.5, "pair_MRR@5": 0.25})

    def test_resume_checkpoint_uses_highest_complete_trainer_checkpoint(self) -> None:
        from train_reranker import find_resume_checkpoint, validated_final_checkpoint_metadata

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (25, 100, 50):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                for name in ("trainer_state.json", "model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
                    (checkpoint / name).write_text("{}")
            incomplete = root / "checkpoint-200"
            incomplete.mkdir()
            for name in ("trainer_state.json", "model.safetensors", "scheduler.pt", "rng_state.pth"):
                (incomplete / name).write_text("{}")
            self.assertEqual(find_resume_checkpoint(root), root / "checkpoint-100")

            final = root / "final"
            final.mkdir()
            (final / "model.safetensors").write_text("weights")
            (final / "round3_checkpoint.json").write_text(json.dumps({
                "binding": {"candidate_hash": "abc"},
                "training_seconds": 12.5,
            }))
            self.assertEqual(
                validated_final_checkpoint_metadata(final, {"candidate_hash": "abc"})["training_seconds"],
                12.5,
            )
            with self.assertRaisesRegex(RuntimeError, "binding"):
                validated_final_checkpoint_metadata(final, {"candidate_hash": "changed"})


class AdaptiveAgentContractTests(unittest.TestCase):
    def test_progressive_agent_restarts_and_exposes_five_then_next_five(self) -> None:
        from adaptive_agent import AdaptivePolicy, run_adaptive_case

        first = {
            "trigger_url": candidate("trigger", 2)["url"],
            "action_url": candidate("action", 3)["url"],
            "confidence": 0.2,
            "request_more": True,
        }
        second = {
            "trigger_url": candidate("trigger", 2)["url"],
            "action_url": candidate("action", 7)["url"],
            "confidence": 0.9,
            "request_more": False,
        }
        client = FakeChatClient([first, second])
        run_adaptive_case(case_without_gold(), AdaptivePolicy(view="schema"), client)
        observed = [
            (request["iteration"], request["candidate_ranks"], "conversation_history" in request)
            for request in client.requests
        ]
        self.assertEqual(observed, [(1, [1, 2, 3, 4, 5], False), (2, [6, 7, 8, 9, 10], False)])

    def test_second_iteration_rehydrates_prior_and_baseline_in_schema_view(self) -> None:
        from adaptive_agent import AdaptivePolicy, run_adaptive_case

        first = {
            "trigger_url": candidate("trigger", 2)["url"],
            "action_url": candidate("action", 3)["url"],
            "confidence": 0.2,
            "request_more": True,
        }
        second = first | {"confidence": 0.9, "request_more": False}
        client = FakeChatClient([first, second])
        run_adaptive_case(
            case_without_gold(),
            AdaptivePolicy(view="names", second_view="schema"),
            client,
        )
        second_request = client.requests[1]
        self.assertIn("evidence", second_request["prior_proposal"]["trigger_candidate"])
        self.assertIn("evidence", second_request["prior_proposal"]["action_candidate"])
        self.assertIn("evidence", second_request["baseline_proposal"]["trigger_candidate"])
        self.assertEqual(second_request["view"], "schema")

    def test_missing_tool_call_falls_back_to_frozen_top1_and_is_counted(self) -> None:
        from adaptive_agent import run_one_shot_case
        from run_agent_experiment import OllamaPairClient

        raw = ToolOmittingOllamaClient()
        trace = run_one_shot_case(
            case_without_gold(),
            view="plain",
            client=OllamaPairClient(raw, "test-model"),
        )
        self.assertEqual(trace["final_pair"], {
            "trigger_url": candidate("trigger", 1)["url"],
            "action_url": candidate("action", 1)["url"],
        })
        self.assertEqual(trace["api_attempts"], 2)
        self.assertEqual(
            trace["iterations"][0]["selection"]["protocol_fallback"],
            "tool_call_missing",
        )

    def test_agent_inference_is_independent_of_gold_references(self) -> None:
        from adaptive_agent import AdaptivePolicy, run_adaptive_case, score_agent_trace

        reply = {
            "trigger_url": candidate("trigger", 2)["url"],
            "action_url": candidate("action", 4)["url"],
            "confidence": 0.9,
            "request_more": False,
        }
        trace = run_adaptive_case(case_without_gold(), AdaptivePolicy(view="plain"), FakeChatClient([reply]))
        correct = score_agent_trace(trace, [{"trigger_url": reply["trigger_url"], "action_url": reply["action_url"]}])
        wrong = score_agent_trace(trace, [{"trigger_url": candidate("trigger", 9)["url"], "action_url": candidate("action", 9)["url"]}])
        self.assertEqual(
            (trace["final_pair"], correct["exact_pair"], wrong["exact_pair"]),
            ({"trigger_url": reply["trigger_url"], "action_url": reply["action_url"]}, True, False),
        )

    def test_call_ledger_reports_total_denominators_and_service_groups(self) -> None:
        from adaptive_agent import summarize_agent_rows

        rows = [
            {"trigger_service": "weather", "action_service": "dropbox", "calls": 0, "exact_pair": True},
            {"trigger_service": "weather", "action_service": "facebook", "calls": 2, "exact_pair": False},
        ]
        summary = summarize_agent_rows(rows)
        self.assertEqual(
            (summary["cases"], summary["total_calls"], summary["calls_per_case"], summary["by_trigger_service"]["weather"]["calls"]),
            (2, 2, 1.0, 2),
        )

    def test_iteration_ledger_reports_second_call_recovery(self) -> None:
        from adaptive_agent import score_agent_trace, summarize_agent_rows

        gold = {
            "trigger_url": candidate("trigger", 2)["url"],
            "action_url": candidate("action", 7)["url"],
        }
        trace = {
            "final_pair": gold,
            "calls": 2,
            "iterations": [
                {"selection": {
                    "trigger_url": candidate("trigger", 1)["url"],
                    "action_url": candidate("action", 1)["url"],
                }},
                {"selection": gold},
            ],
        }
        scored = score_agent_trace(trace, [gold])
        row = trace | scored | {
            "calls": 2,
            "trigger_service": "trigger2",
            "action_service": "action7",
        }
        summary = summarize_agent_rows([row])
        self.assertEqual(scored["iteration_exact_pair"], [False, True])
        self.assertEqual(summary["iterations"]["second_call_transition"]["recoveries"], 1)
        self.assertEqual(summary["iterations"]["iteration_2"]["accuracy"], 1.0)


class RagasParityContractTests(unittest.TestCase):
    def test_native_id_recall_known_answer(self) -> None:
        from ragas_adapter import native_id_recall

        self.assertEqual(native_id_recall(["b", "a", "c"], ["a", "z"]), 0.5)

    def test_id_precision_is_not_mislabeled_as_rank_metric(self) -> None:
        from ragas_adapter import native_id_precision

        self.assertEqual(native_id_precision(["wrong", "gold", "other"], ["gold"]), 1 / 3)

    def test_id_metrics_reject_empty_or_non_string_identifiers(self) -> None:
        from ragas_adapter import native_id_precision, native_id_recall

        with self.assertRaisesRegex(ValueError, "non-empty"):
            native_id_recall(["candidate"], [])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            native_id_precision([], ["gold"])
        with self.assertRaisesRegex(TypeError, "strings"):
            native_id_recall([1], ["1"])


class LaunchContractTests(unittest.TestCase):
    def test_launch_matrix_assigns_unique_declared_gpu_to_each_job(self) -> None:
        from launch_status import validate_launch_matrix

        matrix = [
            {"id": "independent_plain", "gpu": 0},
            {"id": "independent_schema", "gpu": 1},
            {"id": "pair_plain", "gpu": 2},
            {"id": "pair_schema", "gpu": 3},
            {"id": "service_pair", "gpu": 4},
        ]
        self.assertEqual(validate_launch_matrix(matrix), {0, 1, 2, 3, 4})

    def test_gpu_supervisor_backoff_is_bounded(self) -> None:
        from gpu_supervisor import restart_delay

        self.assertEqual([restart_delay(i) for i in (0, 1, 2, 10)], [30, 60, 120, 300])


if __name__ == "__main__":
    unittest.main()

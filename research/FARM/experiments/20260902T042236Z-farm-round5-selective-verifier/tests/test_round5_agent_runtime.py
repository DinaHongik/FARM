#!/usr/bin/env python3
"""Focused controller tests for the round-five pair-card runtime."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))


def _candidate(side: str, rank: int) -> dict:
    service = f"{side}-service-{rank}"
    return {
        "url": f"https://ifttt.com/{service}/{side}s/function-{rank}",
        "channel": service,
        "service_name": service,
        "function_name": f"{side} function {rank}",
        "text_plain": f"plain {side} evidence {rank}",
        "text_schema": f"schema {side} evidence {rank}",
        "retrieval_rank": rank,
    }


def _row(group_id: str) -> dict:
    return {
        "group_id": group_id,
        "query": f"query {group_id}",
        "valid_pairs": [
            {
                "trigger_url": _candidate("trigger", 2)["url"],
                "action_url": _candidate("action", 2)["url"],
            }
        ],
        "gold_trigger_urls": [_candidate("trigger", 2)["url"]],
        "trigger_candidates": [_candidate("trigger", rank) for rank in range(1, 11)],
        "action_candidates": [_candidate("action", rank) for rank in range(1, 11)],
    }


class FakeModelClient:
    def __init__(self, models):
        self.models = models

    def list(self):
        return {"models": self.models}


class Round5AgentRuntimeTests(unittest.TestCase):
    def test_consumed_slice_is_hash_ordered_and_cannot_open_600_or_later(self) -> None:
        from run_round5_agent import select_consumed_rows

        rows = [_row(f"case-{index:04d}") for index in range(700)]
        expected = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(row["group_id"].encode()).hexdigest(), row["group_id"]
            ),
        )[100:120]

        selected = select_consumed_rows(rows, start=100, stop=120)

        self.assertEqual(
            [row["group_id"] for row in selected],
            [row["group_id"] for row in expected],
        )
        with self.assertRaisesRegex(ValueError, r"dev\[0:600\]"):
            select_consumed_rows(rows, start=600, stop=601)

    def test_inference_case_strips_all_references(self) -> None:
        from run_round5_agent import inference_case

        row = _row("case-no-leak")
        maps = {
            side: {
                _candidate(side, rank)["url"]: _candidate(side, rank)
                for rank in range(1, 11)
            }
            for side in ("trigger", "action")
        }

        case = inference_case(row, maps)

        self.assertEqual(
            set(case), {"group_id", "query", "trigger_candidates", "action_candidates"}
        )
        self.assertNotIn("valid_pairs", repr(case))
        self.assertNotIn("gold_trigger_urls", repr(case))

    def test_pair_loader_discards_gold_derived_rank_and_rrf_is_deterministic(self) -> None:
        from run_round5_agent import load_pair_rankings, rrf_pair_ranking

        pair_a = "trigger-a || action-a"
        pair_b = "trigger-b || action-b"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair.json"
            path.write_text(
                json.dumps(
                    {
                        "dataset_id": "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d",
                        "split": "dev",
                        "rows_detail": [
                            {
                                "group_id": "g1",
                                "pair_rank": 1,
                                "pair_scores": [99.0, -2.0],
                                "pair_ranking": [pair_a, pair_b],
                            }
                        ],
                    }
                )
            )

            loaded = load_pair_rankings(path)

        self.assertEqual(loaded, {"g1": [pair_a, pair_b]})
        fused = rrf_pair_ranking([pair_a, pair_b], [pair_b, pair_a])
        self.assertEqual(
            fused,
            [
                {"trigger_url": "trigger-a", "action_url": "action-a"},
                {"trigger_url": "trigger-b", "action_url": "action-b"},
            ],
        )

    def test_model_digest_is_exactly_gated_without_persisting_credentials(self) -> None:
        from run_round5_agent import verify_model_digest

        verify_model_digest(
            "https://ollama.invalid",
            "secret",
            "gemma4:31b",
            "221b330d11a8",
            client=FakeModelClient(
                [{"model": "gemma4:31b", "digest": "221b330d11a8full"}]
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "digest"):
            verify_model_digest(
                "https://ollama.invalid",
                "secret",
                "gemma4:31b",
                "221b330d11a8",
                client=FakeModelClient(
                    [{"model": "gemma4:31b", "digest": "changed"}]
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "digest"):
            verify_model_digest(
                "https://ollama.invalid",
                "secret",
                "gemma4:31b",
                "221b330d11a8",
                client=FakeModelClient(
                    [{"model": "gemma4:31b", "digest": "221b"}]
                ),
            )

    def test_jsonl_resume_and_secret_gates(self) -> None:
        from run_round5_agent import append_jsonl, load_jsonl, write_json_atomic

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            append_jsonl(path, {"group_id": "one", "value": 1}, secret="secret")
            self.assertEqual(load_jsonl(path)[0]["group_id"], "one")
            append_jsonl(path, {"group_id": "one", "value": 2}, secret="secret")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_jsonl(path)
            with self.assertRaisesRegex(RuntimeError, "secret"):
                append_jsonl(
                    path, {"group_id": "two", "value": "secret"}, secret="secret"
                )
            with self.assertRaisesRegex(RuntimeError, "secret"):
                write_json_atomic(
                    Path(directory) / "result.json", {"value": "secret"}, secret="secret"
                )

    def test_evaluation_reports_trigger_action_service_and_joint_separately(self) -> None:
        from run_round5_agent import evaluation_record, summarize_records

        row = _row("case-metrics")
        maps = {
            side: {
                _candidate(side, rank)["url"]: _candidate(side, rank)
                for rank in range(1, 11)
            }
            for side in ("trigger", "action")
        }
        trace = {
            "baseline_pair": {
                "trigger_url": _candidate("trigger", 1)["url"],
                "action_url": _candidate("action", 1)["url"],
            },
            "final_pair": row["valid_pairs"][0],
            "proposer_pair": row["valid_pairs"][0],
            "alternative_pair": None,
            "policy": {"card_count": 5},
            "card_ids": ["pc_1", "pc_2"],
            "calls": [
                {
                    "ok": True,
                    "accounting_complete": True,
                    "api_attempts": 1,
                    "tool_calls": 1,
                }
                for _ in range(3)
            ],
            "accounting": {
                "logical_calls": 3,
                "api_attempts": 3,
                "tool_calls": 3,
                "catalog_reads": 4,
                "complete": True,
                "evidence_presentations": {
                    "proposer_pair_cards": 5,
                    "proposer_endpoints": 10,
                    "verifier_pair_cards": 6,
                    "verifier_endpoints": 12,
                    "verifier_unique_schema_catalog_rows": 4,
                },
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "latency_seconds": 1.5,
                },
            },
            "verifier_decisions": ["ACCEPT_PROPOSAL", "ACCEPT_PROPOSAL"],
            "stable_decision": "ACCEPT_PROPOSAL",
            "fallback_reason": None,
            "retained_baseline": False,
        }

        ranking = [
            row["valid_pairs"][0],
            {
                "trigger_url": _candidate("trigger", 2)["url"],
                "action_url": _candidate("action", 3)["url"],
            },
            {
                "trigger_url": _candidate("trigger", 3)["url"],
                "action_url": _candidate("action", 2)["url"],
            },
            {
                "trigger_url": _candidate("trigger", 3)["url"],
                "action_url": _candidate("action", 3)["url"],
            },
        ]
        record = evaluation_record(
            row, trace, maps, arm="test", pair_ranking=ranking
        )
        metrics = summarize_records([record])

        self.assertFalse(record["baseline_score"]["function_joint"])
        self.assertTrue(record["final_score"]["function_trigger"])
        self.assertTrue(record["final_score"]["function_action"])
        self.assertTrue(record["final_score"]["function_joint"])
        self.assertTrue(record["final_score"]["service_joint"])
        self.assertEqual(metrics["comparison"]["recovered"], 1)
        self.assertEqual(metrics["comparison"]["regressed"], 0)
        self.assertEqual(metrics["proposer"]["function_joint_accuracy"], 1.0)
        self.assertEqual(metrics["proposer"]["recoveries"], 1)
        self.assertEqual(metrics["verifier"]["agreement_rate"], 1.0)
        self.assertEqual(metrics["verifier"]["agreed_choice_conditional_accuracy"], 1.0)
        self.assertEqual(metrics["pair_card_oracle"]["function_joint_coverage"], 1.0)
        self.assertEqual(metrics["pair_card_oracle"]["final_accuracy_when_function_gold_in_deck"], 1.0)
        self.assertEqual(metrics["per_case_averages"]["logical_calls"], 3.0)
        self.assertEqual(metrics["per_case_averages"]["total_tokens"], 15.0)
        self.assertEqual(metrics["per_case_averages"]["latency_seconds"], 1.5)
        self.assertEqual(metrics["per_case_averages"]["catalog_reads"], 4.0)
        self.assertEqual(
            metrics["per_case_averages"]["evidence_presentations"][
                "proposer_pair_cards"
            ],
            5.0,
        )


if __name__ == "__main__":
    unittest.main()

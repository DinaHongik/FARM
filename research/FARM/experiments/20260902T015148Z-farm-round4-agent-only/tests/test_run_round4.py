#!/usr/bin/env python3
"""Controller contracts for the frozen, resumable round-four run."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))


def _candidate(side: str, rank: int, service: str | None = None) -> dict:
    channel = service or f"{side}_service_{rank}"
    return {
        "url": f"https://ifttt.com/{channel}/{side}s/f{rank}",
        "channel": channel,
        "function_name": f"{side} function {rank}",
        "text_plain": f"plain {side} evidence {rank}",
        "text_schema": f"schema {side} evidence {rank}",
        "retrieval_rank": rank,
    }


def _row(group_id: str) -> dict:
    return {
        "group_id": group_id,
        "query": f"query for {group_id}",
        "valid_pairs": [
            {
                "trigger_url": _candidate("trigger", 2)["url"],
                "action_url": _candidate("action", 2)["url"],
            }
        ],
        "gold_trigger_urls": [_candidate("trigger", 2)["url"]],
        "trigger_candidates": [_candidate("trigger", rank) for rank in range(1, 11)],
        "action_candidates": [_candidate("action", rank) for rank in range(1, 11)],
        "trigger_union_candidates": [],
        "action_union_candidates": [],
    }


class Round4ControllerTests(unittest.TestCase):
    def test_frozen_slice_uses_hash_order_and_validates_selected_id_hash(self) -> None:
        from run_round4 import select_frozen_rows

        rows = [_row(f"case-{index}") for index in range(12)]
        ordered = sorted(rows, key=lambda row: hashlib.sha256(row["group_id"].encode()).hexdigest())
        expected = ordered[3:8]
        expected_hash = hashlib.sha256(
            json.dumps([row["group_id"] for row in expected]).encode()
        ).hexdigest()
        split = {
            "ordering": "sha256(group_id) ascending",
            "slice_start": 3,
            "slice_stop": 8,
            "selected_group_ids_sha256": expected_hash,
        }

        selected = select_frozen_rows(rows, split)

        self.assertEqual([row["group_id"] for row in selected], [row["group_id"] for row in expected])
        split["selected_group_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "selected group identity"):
            select_frozen_rows(rows, split)

    def test_inference_case_strips_references_and_only_hierarchy_gets_full_catalog(self) -> None:
        from run_round4 import inference_case

        row = _row("case-one")
        corpora = {
            "trigger": [_candidate("trigger", rank) for rank in range(1, 11)],
            "action": [_candidate("action", rank) for rank in range(1, 11)],
        }

        plain = inference_case(row, corpora, hierarchy=False)
        hierarchy = inference_case(row, corpora, hierarchy=True)

        self.assertEqual(set(plain), {"group_id", "query", "trigger_candidates", "action_candidates"})
        self.assertNotIn("valid_pairs", repr(plain))
        self.assertNotIn("gold_trigger_urls", repr(plain))
        self.assertEqual(hierarchy["trigger_catalog"], corpora["trigger"])
        self.assertEqual(hierarchy["action_catalog"], corpora["action"])

    def test_evaluation_record_enriches_every_gold_and_prediction_with_services(self) -> None:
        from agent_metrics import score_prediction
        from run_round4 import evaluation_record

        row = _row("case-two")
        trigger_gold = row["valid_pairs"][0]["trigger_url"]
        action_gold = row["valid_pairs"][0]["action_url"]
        maps = {
            "trigger": {
                item["url"]: item | {"channel_display": item["channel"].title()}
                for item in row["trigger_candidates"]
            },
            "action": {
                item["url"]: item | {"channel_display": item["channel"].title()}
                for item in row["action_candidates"]
            },
        }
        trace = {
            "case_id": row["group_id"],
            "policy": {"mode": "function_topk", "top_k": 5, "view": "plain", "order": "ranked"},
            "baseline_pair": {
                "trigger_url": row["trigger_candidates"][0]["url"],
                "action_url": row["action_candidates"][0]["url"],
            },
            "final_pair": {"trigger_url": trigger_gold, "action_url": action_gold},
            "retained_baseline": False,
            "selected_services": None,
            "selected_service_pair": None,
            "calls": [{"ok": True, "accounting_complete": True}],
            "accounting": {
                "logical_calls": 1,
                "api_attempts": 1,
                "tool_calls": 1,
                "catalog_reads": 0,
                "complete": True,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        }

        record = evaluation_record(row, trace, maps, arm_id="test_arm")

        self.assertEqual(record["valid_pairs"][0]["trigger_service"], "trigger_service_2")
        self.assertEqual(record["valid_pairs"][0]["action_service"], "action_service_2")
        self.assertTrue(record["protocol_valid"])
        self.assertFalse(record["fallback_used"])
        self.assertTrue(score_prediction(record, record["final_pair"])["function_joint"])
        self.assertTrue(score_prediction(record, record["final_pair"])["service_joint"])

    def test_jsonl_resume_rejects_duplicate_or_secret_bearing_records(self) -> None:
        from run_round4 import append_jsonl, load_jsonl

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            append_jsonl(path, {"group_id": "a", "value": 1}, secret="secret-value")
            self.assertEqual(load_jsonl(path), [{"group_id": "a", "value": 1}])
            append_jsonl(path, {"group_id": "a", "value": 2}, secret="secret-value")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_jsonl(path)
            with self.assertRaisesRegex(RuntimeError, "secret"):
                append_jsonl(path, {"group_id": "b", "value": "secret-value"}, secret="secret-value")

    def test_cross_arm_merge_requires_identical_gold_and_candidate_alignment(self) -> None:
        from summarize_round4 import merge_aligned

        row = _row("aligned-case")
        base = {
            "group_id": row["group_id"],
            "query": row["query"],
            "valid_pairs": [
                row["valid_pairs"][0]
                | {"trigger_service": "trigger_service_2", "action_service": "action_service_2"}
            ],
            "gold_service_pairs": [
                {"trigger_service": "trigger_service_2", "action_service": "action_service_2"}
            ],
            "trigger_candidates": row["trigger_candidates"],
            "action_candidates": row["action_candidates"],
            "baseline_pair": {
                "trigger_url": row["trigger_candidates"][0]["url"],
                "action_url": row["action_candidates"][0]["url"],
            },
            "final_pair": row["valid_pairs"][0],
            "policy": {"mode": "function_topk", "top_k": 5},
            "calls": [{"ok": True}],
            "accounting": {"logical_calls": 1},
            "protocol_valid": True,
            "fallback_used": False,
            "retained_baseline": False,
        }

        merged = merge_aligned({"arm_a": [base], "arm_b": [dict(base)]})

        self.assertEqual(set(merged[0]["arms"]), {"retrieval_top1", "arm_a", "arm_b"})
        changed = dict(base) | {"query": "changed after freezing"}
        with self.assertRaisesRegex(ValueError, "payload changed"):
            merge_aligned({"arm_a": [base], "arm_b": [changed]})


if __name__ == "__main__":
    unittest.main()

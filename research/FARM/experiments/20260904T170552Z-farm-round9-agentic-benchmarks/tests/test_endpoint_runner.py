from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.artifact_io import ordered_ids_sha256, read_json, read_jsonl, sha256_text
from farm_r9.endpoint_runner import (
    EndpointRun,
    _rescore_terminal_records,
    exact_mcnemar,
    run_endpoint_experiment,
    wilson_95,
)
from farm_r9.privacy import DataClassification, DataSource, export_public_aggregate
from farm_r9.ollama_client import OllamaTransportError


def _case(*, n: int = 10) -> dict:
    triggers = [{"alias": f"T{i:02d}", "service": "svc", "service_id": "svc", "function": "good" if i == 2 else f"bad-{i}", "field_names": []} for i in range(1, n + 1)]
    actions = [{"alias": f"A{i:02d}", "service": "act", "service_id": "act", "function": "send" if i == 4 else f"bad-{i}", "field_names": []} for i in range(1, n + 1)]
    return {
        "benchmark": "recipegen_gold", "data_classification": "public", "case_id": "case-1",
        "input": {"query": "private-looking request must not reach aggregate"},
        "public_evidence": {"trigger_candidates": triggers, "action_candidates": actions},
        "private": {
            "trigger": {"ranking": ["trigger-good"], "alias_map": {row["alias"]: "trigger-good" if row["alias"] == "T02" else f"trigger-{row['alias']}" for row in triggers}},
            "action": {"ranking": ["action-good"], "alias_map": {row["alias"]: "action-good" if row["alias"] == "A04" else f"action-{row['alias']}" for row in actions}},
            "gold": {
                "trigger_channel_norm": "svc", "action_channel_norm": "act",
                "trigger_function_norm": "good", "action_function_norm": "send",
                "trigger_fields_norm": [], "action_fields_norm": [],
            },
        },
    }


def _run(directory: Path, arm: str = "retrieval_top1") -> EndpointRun:
    cases = [_case()]
    return EndpointRun(
        benchmark="recipegen_gold", arm=arm, classification=DataClassification.PUBLIC,
        data_source=DataSource.RECIPEGEN, output_directory=directory,
        candidate_artifact_sha256=sha256_text("candidate"), ordered_case_ids_sha256=ordered_ids_sha256(cases),
        model_metadata={}, protocol_metadata={"arm": arm, "semantic_calls_max": 0 if arm == "retrieval_top1" else 1},
    )


class EndpointRunnerTests(unittest.TestCase):
    def test_transport_pending_case_does_not_crash_or_publish_partial_table(self) -> None:
        cases = [deepcopy(_case()) for _ in range(3)]
        for index, case in enumerate(cases):
            case["case_id"] = f"case-{index}"

        class Client:
            cloud = False
            calls = 0

            def chat(self, **kwargs):
                self.calls += 1
                if self.calls == 2:
                    raise OllamaTransportError("timeout")
                return SimpleNamespace(
                    content="not json", request_sha256="a" * 64,
                    response_sha256="b" * 64, prompt_tokens=1,
                    completion_tokens=2, provider_latency_ms=3.0,
                    queue_wait_ms=0.0, physical_attempts=1, cache_hit=False,
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run = replace(_run(output, "same_model_one_shot"),
                          ordered_case_ids_sha256=ordered_ids_sha256(cases))
            result = run_endpoint_experiment(cases=cases, run=run, client=Client())
            self.assertEqual(result["terminal_n"], 2)
            self.assertEqual(result["pending_n"], 1)
            self.assertTrue((output / "aggregate_private.json").exists())
            self.assertFalse((output / "aggregate_public.json").exists())

    def test_retrieval_maps_private_rank_one_to_shuffled_alias_and_resumes(self) -> None:
        cases = [_case()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private-run"
            result = run_endpoint_experiment(cases=cases, run=_run(output), client=None)
            self.assertEqual(result["metrics"]["function_joint"]["correct"], 1)
            records = read_jsonl(output / "records.jsonl")
            self.assertEqual(records[0]["prediction"]["trigger_alias"], "T02")
            self.assertEqual(records[0]["prediction"]["action_alias"], "A04")
            self.assertEqual(os.stat(output / "records.jsonl").st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o700)
            public = read_json(output / "aggregate_public.json")
            def keys(value):
                if isinstance(value, dict):
                    return set(value) | set().union(*(keys(child) for child in value.values()))
                if isinstance(value, list):
                    return set().union(*(keys(child) for child in value)) if value else set()
                return set()
            self.assertTrue({"query", "gold", "case_id", "schemas", "candidates"}.isdisjoint(keys(public)))
            self.assertEqual(export_public_aggregate(public, source_classification=DataClassification.PUBLIC)["n"], 1)
            self.assertEqual(public["operational_metrics"]["semantic_calls_total"], 0)
            run_endpoint_experiment(cases=cases, run=_run(output), client=None)
            self.assertEqual(len(read_jsonl(output / "records.jsonl")), 1)

    def test_model_gets_all_ten_candidates_and_protocol_failure_is_not_retrieval(self) -> None:
        cases = [_case()]
        received = []

        class Client:
            cloud = False

            def chat(self, **kwargs):
                received.append(json.loads(kwargs["messages"][-1]["content"]))
                return SimpleNamespace(content="not json", request_sha256="a" * 64, response_sha256="b" * 64,
                                       prompt_tokens=1, completion_tokens=2, provider_latency_ms=3.0,
                                       queue_wait_ms=0.0, physical_attempts=1, cache_hit=False)

        with tempfile.TemporaryDirectory() as directory:
            result = run_endpoint_experiment(cases=cases, run=_run(Path(directory) / "run", "same_model_one_shot"), client=Client())
            self.assertEqual(len(received[0]["trigger_candidates"]), 10)
            self.assertEqual(len(received[0]["action_candidates"]), 10)
            self.assertEqual(result["metrics"]["function_joint"]["correct"], 0)
            self.assertEqual(result["metrics"]["field_joint_exact"]["n"], 1)
            self.assertEqual(result["metrics"]["field_joint_exact"]["correct"], 0)
            record = read_jsonl(Path(directory) / "run" / "records.jsonl")[0]
            self.assertEqual(record["outcome"], "protocol_failure")
            self.assertIsNone(record["prediction"])

    def test_exact_statistics(self) -> None:
        interval = wilson_95(0, 10)
        self.assertAlmostEqual(interval["low"], 0.0)
        self.assertAlmostEqual(interval["high"], 27.753, places=2)
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(1, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar(5, 0), 0.0625)

    def test_field_name_mismatch_has_privacy_safe_public_failure_label(self) -> None:
        cases = [_case()]

        class Client:
            cloud = False

            def chat(self, **kwargs):
                payload = {
                    "trigger_alias": "T02",
                    "action_alias": "A04",
                    "trigger_field_names": ["invented"],
                    "action_field_names": [],
                    "preview": "preview",
                    "evidence_aliases": ["T02", "A04"],
                }
                return SimpleNamespace(
                    content=json.dumps(payload), request_sha256="a" * 64,
                    response_sha256="b" * 64, prompt_tokens=1,
                    completion_tokens=2, provider_latency_ms=3.0,
                    queue_wait_ms=0.0, physical_attempts=1, cache_hit=False,
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            run_endpoint_experiment(
                cases=cases,
                run=_run(output, "same_model_one_shot"),
                client=Client(),
            )
            public = read_json(output / "aggregate_public.json")
            self.assertEqual(public["failure_counts"], {"trigger_field_name_mismatch": 1})

    def test_aggregate_rescores_stale_null_field_scores_as_failures(self) -> None:
        case = _case()
        stale = {
            "case-1": {
                "case_id": "case-1",
                "terminal": True,
                "prediction": None,
                "scores": {"field_joint_exact": None},
            }
        }
        refreshed = _rescore_terminal_records(stale, {"case-1": case})
        self.assertFalse(refreshed["case-1"]["scores"]["field_trigger_exact"])
        self.assertFalse(refreshed["case-1"]["scores"]["field_action_exact"])
        self.assertFalse(refreshed["case-1"]["scores"]["field_joint_exact"])


if __name__ == "__main__":
    unittest.main()

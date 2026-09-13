#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ollama_constraint_adapter import OllamaConstraintChooser  # noqa: E402


class FakeResponse:
    def __init__(self, arguments: dict[str, str], name: str, *, status: int = 200) -> None:
        self.status_code = status
        self._value = {
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {"tool_calls": [{
                    "function": {"name": name, "arguments": json.dumps(arguments)}
                }]},
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
        self.content = json.dumps(self._value).encode()

    def json(self) -> dict:
        return self._value


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def post(self, endpoint: str, **kwargs):
        self.requests.append({"endpoint": endpoint, **kwargs})
        return self.responses.pop(0)


def selection_request() -> dict:
    triggers = [
        {
            "candidate_id": f"T{i:02d}", "service_name": f"trigger-service-{i}",
            "function_name": f"trigger-{i}", "evidence": f"trigger evidence {i}",
        }
        for i in range(1, 11)
    ]
    actions = [
        {
            "candidate_id": f"A{i:02d}", "service_name": f"action-service-{i}",
            "function_name": f"action-{i}", "evidence": f"action evidence {i}",
        }
        for i in range(1, 11)
    ]
    return {
        "schema_version": "farm_r7_factorized_request_v1",
        "request_id": "R1234567890abcdef",
        "phase": "factorized_selection",
        "query": "When this happens, do that",
        "instruction": "Choose independently.",
        "current_pair": {"trigger_choice": "T01", "action_choice": "A01"},
        "candidates": {"trigger": triggers, "action": actions},
        "output_contract": {
            "decision": ["KEEP", "CHANGE_TRIGGER", "CHANGE_ACTION", "CHANGE_BOTH", "ABSTAIN"],
            "trigger_choice": ["KEEP", *[f"T{i:02d}" for i in range(1, 11)]],
            "action_choice": ["KEEP", *[f"A{i:02d}" for i in range(1, 11)]],
            "additional_properties": False,
        },
    }


def verifier_request() -> dict:
    side = {
        "candidate_id": "T01", "service_name": "service",
        "function_name": "function", "evidence": "evidence",
        "schema": {"kind": "trigger"},
    }
    action = dict(side) | {"candidate_id": "A01", "schema": {"kind": "action"}}
    validation = {"status": "compatible", "issues": []}
    return {
        "schema_version": "farm_r7_pair_verifier_request_v1",
        "request_id": "Rabcdef1234567890",
        "phase": "pair_verification",
        "query": "When this happens, do that",
        "instruction": "Choose the better pair.",
        "pairs": [
            {"pair_id": "P01", "trigger": side, "action": action, "schema_validation": validation},
            {"pair_id": "P02", "trigger": side, "action": action, "schema_validation": validation},
        ],
        "output_contract": {
            "choice_id": ["P01", "P02", "ABSTAIN"],
            "additional_properties": False,
        },
    }


class AdapterTests(unittest.TestCase):
    def chooser(self, client: FakeClient, journal: Path) -> OllamaConstraintChooser:
        return OllamaConstraintChooser(
            "https://example.invalid", "secret-value", "model:tag", journal,
            client=client, retry_delay=0, monotonic=lambda: 1.0,
        )

    def test_selection_tool_call_is_normalized(self) -> None:
        response = FakeResponse(
            {"decision": "CHANGE_TRIGGER", "trigger_choice": "T02", "action_choice": "KEEP"},
            "select_endpoints",
        )
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient([response])
            result = self.chooser(client, Path(temp) / "journal.jsonl").select(selection_request())
        self.assertTrue(result["ok"])
        self.assertEqual("T02", result["trigger_choice"])
        self.assertEqual("KEEP", result["action_choice"])
        self.assertEqual(1, result["tool_calls"])
        payload = client.requests[0]["json"]
        self.assertEqual("select_endpoints", payload["tools"][0]["function"]["name"])
        self.assertEqual("select_endpoints", payload["tool_choice"]["function"]["name"])
        self.assertEqual(2, len(payload["messages"]))

    def test_verifier_tool_call_is_normalized(self) -> None:
        response = FakeResponse({"choice_id": "P02"}, "choose_pair")
        with tempfile.TemporaryDirectory() as temp:
            result = self.chooser(
                FakeClient([response]), Path(temp) / "journal.jsonl"
            ).select(verifier_request())
        self.assertEqual("P02", result["choice_id"])
        self.assertTrue(result["ok"])

    def test_invalid_enum_retries_from_original_request(self) -> None:
        invalid = FakeResponse(
            {"decision": "CHANGE_TRIGGER", "trigger_choice": "T99", "action_choice": "KEEP"},
            "select_endpoints",
        )
        valid = FakeResponse(
            {"decision": "KEEP", "trigger_choice": "KEEP", "action_choice": "KEEP"},
            "select_endpoints",
        )
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient([invalid, valid])
            result = self.chooser(client, Path(temp) / "journal.jsonl").select(selection_request())
        self.assertTrue(result["ok"])
        self.assertEqual(2, result["api_attempts"])
        self.assertEqual(2, result["tool_calls"])
        self.assertEqual(3, len(client.requests[1]["json"]["messages"]))
        self.assertEqual(
            client.requests[0]["json"]["messages"][1],
            client.requests[1]["json"]["messages"][1],
        )

    def test_reference_fields_fail_before_transport(self) -> None:
        request = selection_request() | {"valid_pairs": [{"trigger_url": "leak"}]}
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient([])
            result = self.chooser(client, Path(temp) / "journal.jsonl").select(request)
        self.assertFalse(result["ok"])
        self.assertEqual([], client.requests)
        self.assertIn("reference", result["error"])

    def test_journal_is_content_free_and_secret_free(self) -> None:
        response = FakeResponse(
            {"decision": "KEEP", "trigger_choice": "KEEP", "action_choice": "KEEP"},
            "select_endpoints",
        )
        with tempfile.TemporaryDirectory() as temp:
            journal = Path(temp) / "journal.jsonl"
            self.chooser(FakeClient([response]), journal).select(selection_request())
            content = journal.read_text()
        self.assertNotIn("secret-value", content)
        self.assertNotIn("When this happens", content)
        self.assertEqual(2, len(content.splitlines()))


if __name__ == "__main__":
    unittest.main()

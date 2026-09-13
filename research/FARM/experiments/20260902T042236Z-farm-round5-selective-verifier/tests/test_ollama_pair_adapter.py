#!/usr/bin/env python3
"""Focused transport-boundary tests for round five."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))


class FakeResponse:
    def __init__(self, status: int, body: dict):
        self.status_code = status
        self._body = body
        self.content = json.dumps(body, sort_keys=True).encode()

    def json(self) -> dict:
        return self._body


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def post(self, url: str, **kwargs):
        self.requests.append({"url": url, **kwargs})
        value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _card(identifier: str, number: int) -> dict:
    return {
        "card_id": identifier,
        "trigger": {
            "service_name": f"trigger service {number}",
            "function_name": f"trigger function {number}",
            "evidence": f"trigger evidence {number}",
        },
        "action": {
            "service_name": f"action service {number}",
            "function_name": f"action function {number}",
            "evidence": f"action evidence {number}",
        },
    }


def _request(*, verification: bool = False) -> dict:
    return {
        "case_id": "internal-case-id",
        "query": "When this happens, perform that operation.",
        "phase": "pair_verification" if verification else "pair_proposal",
        "instruction": "Choose the best complete pair.",
        "evidence_view": "schema" if verification else "plain",
        "allow_abstain": verification,
        "cards": [_card("pc_aaa", 1), _card("pc_bbb", 2), _card("pc_ccc", 3)],
    }


def _tool_response(choice: str, *, usage=(7, 2, 9)) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "choose_card",
                                "arguments": json.dumps({"choice_id": choice}),
                            },
                        }
                    ]
                },
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[2],
        },
    }


class OllamaPairAdapterTests(unittest.TestCase):
    def test_success_is_blinded_validated_and_write_ahead_journaled(self) -> None:
        from ollama_pair_adapter import OllamaPairChooser

        client = FakeClient([FakeResponse(200, _tool_response("pc_bbb"))])
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            chooser = OllamaPairChooser(
                "https://ollama.invalid",
                "top-secret-api-key",
                "gemma4:31b",
                journal,
                client=client,
                monotonic=lambda: 0.0,
            )

            result = chooser.select(_request())

            self.assertTrue(result["ok"])
            self.assertEqual(result["choice_id"], "pc_bbb")
            payload_text = json.dumps(client.requests[0]["json"], sort_keys=True)
            self.assertNotIn("internal-case-id", payload_text)
            self.assertNotIn("retrieval_rank", payload_text)
            self.assertNotIn("ifttt.com", payload_text)
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(
                [record["event"] for record in records],
                ["request_started", "request_finished"],
            )
            self.assertEqual(records[1]["protocol_outcome"], "valid_choose_card")
            self.assertNotIn("top-secret-api-key", journal.read_text())

    def test_nested_reference_is_rejected_before_http(self) -> None:
        from ollama_pair_adapter import OllamaPairChooser

        client = FakeClient([])
        with tempfile.TemporaryDirectory() as directory:
            chooser = OllamaPairChooser(
                "https://ollama.invalid",
                "secret",
                "model",
                Path(directory) / "attempts.jsonl",
                client=client,
            )
            request = _request() | {"metadata": {"audit": {"pair_rank": 1}}}

            result = chooser.select(request)

            self.assertFalse(result["ok"])
            self.assertIn("reference_fields_forbidden", result["error"])
            self.assertEqual(client.requests, [])

    def test_abstain_is_valid_only_for_verification(self) -> None:
        from ollama_pair_adapter import OllamaPairChooser

        with tempfile.TemporaryDirectory() as directory:
            verifier = OllamaPairChooser(
                "https://ollama.invalid",
                "secret",
                "model",
                Path(directory) / "verify.jsonl",
                client=FakeClient([FakeResponse(200, _tool_response("ABSTAIN"))]),
                monotonic=lambda: 0.0,
            )
            self.assertEqual(verifier.select(_request(verification=True))["choice_id"], "ABSTAIN")

            proposer_client = FakeClient(
                [
                    FakeResponse(200, _tool_response("ABSTAIN")),
                    FakeResponse(200, _tool_response("ABSTAIN")),
                ]
            )
            proposer = OllamaPairChooser(
                "https://ollama.invalid",
                "secret",
                "model",
                Path(directory) / "propose.jsonl",
                client=proposer_client,
                retry_delay=0,
                monotonic=lambda: 0.0,
            )
            result = proposer.select(_request())
            self.assertFalse(result["ok"])
            self.assertEqual(result["error"], "choice_outside_allowed_enum")
            self.assertEqual(result["api_attempts"], 2)

    def test_transient_failure_uses_one_bounded_retry_and_accumulates_usage(self) -> None:
        from ollama_pair_adapter import OllamaPairChooser

        client = FakeClient(
            [
                FakeResponse(503, {"error": {"type": "capacity"}}),
                FakeResponse(200, _tool_response("pc_aaa", usage=(5, 1, 6))),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            chooser = OllamaPairChooser(
                "https://ollama.invalid",
                "secret",
                "model",
                Path(directory) / "attempts.jsonl",
                client=client,
                retry_delay=0,
                monotonic=lambda: 0.0,
            )

            result = chooser.select(_request())

            self.assertTrue(result["ok"])
            self.assertEqual(result["api_attempts"], 2)
            self.assertEqual(result["usage"]["total_tokens"], 6)
            self.assertEqual(len(client.requests), 2)


if __name__ == "__main__":
    unittest.main()

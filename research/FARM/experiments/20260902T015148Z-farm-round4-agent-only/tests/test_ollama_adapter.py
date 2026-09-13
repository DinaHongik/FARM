#!/usr/bin/env python3
"""Contract tests for the round-four Ollama selector boundary."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))


class FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.content = json.dumps(body, sort_keys=True).encode()
        self._body = body

    def json(self) -> dict:
        return self._body


class FakeHTTPClient:
    """Fake only the external HTTP transport seam."""

    def __init__(self, replies: list[FakeResponse | Exception]):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def post(self, url: str, **kwargs) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class SequenceClock:
    def __init__(self, values: list[float]):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


def selection_request() -> dict:
    return {
        "case_id": "case-secret-internal-id",
        "query": "When a new photo is posted, archive it.",
        "phase": "function",
        "instruction": "Select the semantically coherent trigger/action pair.",
        "evidence_scope": {"top_k": 5, "cumulative": True},
        "trigger_candidates": [
            {
                "candidate_id": "T_01",
                "service_name": "photos",
                "function_name": "new photo",
                "evidence": "fires when a photo is created",
            },
            {
                "candidate_id": "T_02",
                "service_name": "photos",
                "function_name": "new album",
                "evidence": "fires when an album is created",
            },
        ],
        "action_candidates": [
            {
                "candidate_id": "A_01",
                "service_name": "storage",
                "function_name": "archive file",
                "evidence": "stores a file in an archive",
            },
            {
                "candidate_id": "A_02",
                "service_name": "storage",
                "function_name": "delete file",
                "evidence": "deletes a file",
            },
        ],
    }


def tool_response(
    arguments: dict | None,
    *,
    usage: tuple[int, int, int] = (0, 0, 0),
    include_tool: bool = True,
) -> dict:
    message: dict = {"content": None}
    if include_tool:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "submit_pair",
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    return {
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[2],
        },
    }


class OllamaSelectorTests(unittest.TestCase):
    def test_attempt_latency_is_journaled_and_accumulated_deterministically(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                FakeResponse(408, {"error": {"type": "timeout"}}),
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_01", "action_id": "A_02"}, usage=(5, 1, 6)),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                "https://ollama.invalid",
                "test-key",
                "test-model",
                journal,
                client=client,
                retry_delay=0,
                monotonic=SequenceClock([1.0, 1.25, 2.0, 2.75]),
            )

            result = selector.select(selection_request())

            self.assertEqual(
                result["usage"],
                {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                    "latency_seconds": 1.0,
                },
            )
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([record["latency_seconds"] for record in records], [0.25, 0.75])

    def test_transient_transport_failure_uses_the_single_remaining_attempt(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                OSError("connection reset"),
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_02", "action_id": "A_01"}, usage=(6, 2, 8)),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                "https://ollama.invalid",
                "test-key",
                "test-model",
                journal,
                client=client,
                retry_delay=0,
                monotonic=lambda: 0.0,
            )

            result = selector.select(selection_request())

            self.assertEqual(
                result,
                {
                    "ok": True,
                    "trigger_id": "T_02",
                    "action_id": "A_01",
                    "api_attempts": 2,
                    "tool_calls": 1,
                    "usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 2,
                        "total_tokens": 8,
                        "latency_seconds": 0.0,
                    },
                    "error": None,
                },
            )
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(
                [record["protocol_outcome"] for record in records],
                ["transport_error", "valid_submit_pair"],
            )

    def test_transient_http_statuses_use_the_single_remaining_attempt(self) -> None:
        from ollama_adapter import OllamaSelector

        for status in (408, 429, 500, 599):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                client = FakeHTTPClient(
                    [
                        FakeResponse(status, {"error": {"type": "transient"}}),
                        FakeResponse(
                            200,
                            tool_response(
                                {"trigger_id": "T_01", "action_id": "A_01"},
                                usage=(8, 2, 10),
                            ),
                        ),
                    ]
                )
                journal = Path(directory) / "attempts.jsonl"
                selector = OllamaSelector(
                    "https://ollama.invalid",
                    "test-key",
                    "test-model",
                    journal,
                    client=client,
                    retry_delay=0,
                    monotonic=lambda: 0.0,
                )

                result = selector.select(selection_request())

                self.assertEqual(
                    result,
                    {
                        "ok": True,
                        "trigger_id": "T_01",
                        "action_id": "A_01",
                        "api_attempts": 2,
                        "tool_calls": 1,
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 2,
                            "total_tokens": 10,
                            "latency_seconds": 0.0,
                        },
                        "error": None,
                    },
                )
                records = [json.loads(line) for line in journal.read_text().splitlines()]
                self.assertEqual(
                    [(record["attempt"], record["http_status"]) for record in records],
                    [(1, status), (2, 200)],
                )

    def test_second_transient_failure_returns_nondecision_without_a_third_attempt(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                FakeResponse(503, {"error": {"type": "capacity"}}),
                OSError("connection reset again"),
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_01", "action_id": "A_01"}),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                "https://ollama.invalid",
                "test-key",
                "test-model",
                journal,
                client=client,
                retry_delay=0,
                monotonic=lambda: 0.0,
            )

            result = selector.select(selection_request())

            self.assertEqual(
                result,
                {
                    "ok": False,
                    "trigger_id": None,
                    "action_id": None,
                    "api_attempts": 2,
                    "tool_calls": 0,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "latency_seconds": 0.0,
                    },
                    "error": "transport_error",
                },
            )
            self.assertEqual(len(client.requests), 2)
            self.assertEqual(len(journal.read_text().splitlines()), 2)

    def test_nontransient_client_error_is_not_retried(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                FakeResponse(400, {"error": {"type": "invalid_request"}}),
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_01", "action_id": "A_01"}),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                "https://ollama.invalid",
                "test-key",
                "test-model",
                journal,
                client=client,
                retry_delay=0,
                monotonic=lambda: 0.0,
            )

            result = selector.select(selection_request())

            self.assertEqual(
                result,
                {
                    "ok": False,
                    "trigger_id": None,
                    "action_id": None,
                    "api_attempts": 1,
                    "tool_calls": 0,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "latency_seconds": 0.0,
                    },
                    "error": "http_status_400",
                },
            )
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(len(journal.read_text().splitlines()), 1)

    def test_invalid_enum_is_corrected_once_using_explicit_allowed_ids(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                FakeResponse(
                    200,
                    tool_response(
                        {"trigger_id": "T_NOT_ALLOWED", "action_id": "A_01"},
                        usage=(11, 2, 13),
                    ),
                ),
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_01", "action_id": "A_01"}, usage=(17, 3, 20)),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                base_url="https://ollama.invalid",
                api_key="not-a-real-key",
                model="test-model",
                journal_path=journal,
                client=client,
                monotonic=lambda: 0.0,
            )
            result = selector.select(selection_request())

            self.assertEqual(
                result,
                {
                    "ok": True,
                    "trigger_id": "T_01",
                    "action_id": "A_01",
                    "api_attempts": 2,
                    "tool_calls": 2,
                    "usage": {
                        "prompt_tokens": 28,
                        "completion_tokens": 5,
                        "total_tokens": 33,
                        "latency_seconds": 0.0,
                    },
                    "error": None,
                },
            )
            correction = client.requests[1]["json"]["messages"][-1]["content"]
            self.assertIn("T_01", correction)
            self.assertIn("T_02", correction)
            self.assertIn("A_01", correction)
            self.assertIn("A_02", correction)
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(
                [record["protocol_outcome"] for record in records],
                ["candidate_id_outside_allowed_enum", "valid_submit_pair"],
            )

    def test_missing_tool_is_corrected_once_and_all_attempts_are_accounted(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                FakeResponse(200, tool_response(None, usage=(9, 4, 13), include_tool=False)),
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_02", "action_id": "A_02"}, usage=(15, 2, 17)),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                "https://ollama.invalid",
                "test-key",
                "test-model",
                journal,
                client=client,
                monotonic=lambda: 0.0,
            )

            result = selector.select(selection_request())

            self.assertEqual(
                result,
                {
                    "ok": True,
                    "trigger_id": "T_02",
                    "action_id": "A_02",
                    "api_attempts": 2,
                    "tool_calls": 1,
                    "usage": {
                        "prompt_tokens": 24,
                        "completion_tokens": 6,
                        "total_tokens": 30,
                        "latency_seconds": 0.0,
                    },
                    "error": None,
                },
            )
            self.assertIn("missing_submit_pair", client.requests[1]["json"]["messages"][-1]["content"])
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual(len(records), 2)

    def test_later_provider_failure_never_turns_invalid_output_into_a_decision(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient(
            [
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_NOT_ALLOWED", "action_id": "A_01"}),
                ),
                FakeResponse(
                    503,
                    {"error": {"type": "capacity", "message": "try again later"}},
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            selector = OllamaSelector(
                "https://ollama.invalid/v1",
                "test-key",
                "test-model",
                Path(directory) / "attempts.jsonl",
                client=client,
                monotonic=lambda: 0.0,
            )

            result = selector.select(selection_request())

            self.assertEqual(
                result,
                {
                    "ok": False,
                    "trigger_id": None,
                    "action_id": None,
                    "api_attempts": 2,
                    "tool_calls": 1,
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "latency_seconds": 0.0,
                    },
                    "error": "http_status_503",
                },
            )

    def test_reasoning_effort_is_an_explicit_frozen_model_setting(self) -> None:
        from ollama_adapter import OllamaSelector

        client = FakeHTTPClient([
            FakeResponse(200, tool_response({"trigger_id": "T_01", "action_id": "A_01"}))
        ])
        with tempfile.TemporaryDirectory() as directory:
            selector = OllamaSelector(
                "https://ollama.invalid",
                "test-key",
                "test-model",
                Path(directory) / "attempts.jsonl",
                client=client,
                reasoning_effort="none",
                monotonic=lambda: 0.0,
            )

            result = selector.select(selection_request())

            self.assertTrue(result["ok"])
            self.assertEqual(client.requests[0]["json"]["reasoning_effort"], "none")
            with self.assertRaisesRegex(ValueError, "reasoning_effort"):
                OllamaSelector(
                    "https://ollama.invalid",
                    "test-key",
                    "test-model",
                    Path(directory) / "invalid.jsonl",
                    client=FakeHTTPClient([]),
                    reasoning_effort="auto",
                )

    def test_journal_excludes_secrets_prompts_references_and_raw_candidates(self) -> None:
        from ollama_adapter import OllamaSelector

        secret = "super-secret-bearer-value"
        request = selection_request()
        request["trigger_candidates"][0].update(
            {
                "url": "https://private.invalid/trigger/1",
                "retrieval_rank": 1,
                "gold": "must-never-leave-the-runner",
            }
        )
        client = FakeHTTPClient(
            [
                FakeResponse(
                    200,
                    tool_response({"trigger_id": "T_01", "action_id": "A_01"}, usage=(7, 2, 9)),
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            selector = OllamaSelector(
                "https://ollama.invalid",
                secret,
                "test-model",
                journal,
                client=client,
                monotonic=lambda: 0.0,
            )

            result = selector.select(request)

            self.assertTrue(result["ok"])
            persisted = journal.read_text()
            self.assertNotIn(secret, persisted)
            self.assertNotIn(request["query"], persisted)
            self.assertNotIn(request["case_id"], persisted)
            self.assertNotIn("must-never-leave-the-runner", persisted)
            outgoing = client.requests[0]["json"]
            self.assertEqual(
                (
                    client.requests[0]["url"],
                    outgoing["reasoning_effort"],
                    outgoing["tool_choice"],
                ),
                (
                    "https://ollama.invalid/v1/chat/completions",
                    "low",
                    {"type": "function", "function": {"name": "submit_pair"}},
                ),
            )
            public_prompt = outgoing["messages"][1]["content"]
            self.assertEqual(json.loads(public_prompt)["instruction"], request["instruction"])
            self.assertNotIn("https://private.invalid", public_prompt)
            self.assertNotIn("retrieval_rank", public_prompt)
            self.assertNotIn("must-never-leave-the-runner", public_prompt)


if __name__ == "__main__":
    unittest.main()

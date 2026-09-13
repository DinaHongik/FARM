from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.ollama_client import ChatResult
from farm_r9.ollama_client import OllamaTransportError
from farm_r9.privacy import DataClassification, DataSource
from farm_r9.yao_agent import (
    COMPONENTS,
    EndpointLabels,
    aggregate_records,
    build_initial_messages,
    execute_case,
    score_prediction,
    validate_final_cases,
    wilson_interval,
)


CATALOG = {
    "trigger_channel": ("Gmail", "RSS Feed"),
    "trigger_function": ("Any new attachment in inbox", "New feed item"),
    "action_channel": ("Google Drive", "Email"),
    "action_function": ("Add row to spreadsheet", "Send me an email"),
}


def case(case_id: str = "yao:1") -> dict:
    gold = {
        "trigger_channel": "Gmail",
        "trigger_function": "Any new attachment in inbox",
        "action_channel": "Google Drive",
        "action_function": "Add row to spreadsheet",
    }
    return {
        "schema_version": "round9-case-v1",
        "benchmark": "interactive_ifttt",
        "case_id": case_id,
        "stratum": "VI-4",
        "input": {"query": "Save new email attachments in my spreadsheet"},
        "simulator": {
            "answer_options": {component: [f"answer for {component}"] for component in COMPONENTS},
            "frozen_answers": {
                component: {"answer_index": 0, "answer": f"answer for {component}"}
                for component in COMPONENTS
            },
            "max_asks_per_component": 1,
        },
        "private_gold": {
            **gold,
            "pseudo_ask_labels": [1, 0, 0, 0],
            "valid_endpoint_constraints": {},
        },
    }


def response(content: dict, turn: int) -> ChatResult:
    text = json.dumps(content)
    return ChatResult(
        semantic_id=f"s{turn}", model="test-model", content=text, tool_calls=(),
        prompt_tokens=10, completion_tokens=5, provider_latency_ms=20,
        queue_wait_ms=2, physical_attempts=1, request_sha256=f"r{turn}",
        response_sha256=f"o{turn}", cache_hit=False, raw_response={},
    )


class FakeClient:
    model = "test-model"

    def __init__(self, outputs: list[dict]):
        self.outputs = outputs
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return response(output, len(self.calls))


class YaoAgentTests(unittest.TestCase):
    def test_one_shot_commits_once_and_never_exposes_gold(self):
        prediction = {
            "trigger_channel": "Gmail",
            "trigger_function": "Any new attachment in inbox",
            "action_channel": "Google Drive",
            "action_function": "Add row to spreadsheet",
        }
        client = FakeClient([{"action": "commit", "prediction": prediction}])
        result = execute_case(
            case=case(), catalog=CATALOG, arm="same_model_one_shot", client=client
        )
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(result["scores"]["joint"])
        serialized = json.dumps(client.calls)
        self.assertNotIn("pseudo_ask_labels", serialized)
        self.assertNotIn("valid_endpoint_constraints", serialized)

    def test_private_scoring_payload_is_never_sent(self):
        poisoned = case()
        poisoned["private_gold"]["private_marker"] = "NEVER_SEND_PRIVATE_MARKER"
        client = FakeClient([{
            "action": "commit",
            "prediction": {
                "trigger_channel": "Gmail",
                "trigger_function": "Any new attachment in inbox",
                "action_channel": "Google Drive",
                "action_function": "Add row to spreadsheet",
            },
        }])
        execute_case(
            case=poisoned, catalog=CATALOG, arm="same_model_one_shot", client=client
        )
        self.assertNotIn("NEVER_SEND_PRIVATE_MARKER", json.dumps(client.calls))

    def test_transport_failure_propagates_for_cache_backed_resume(self):
        client = FakeClient([OllamaTransportError("exhausted")])
        with self.assertRaises(OllamaTransportError):
            execute_case(
                case=case(), catalog=CATALOG, arm="same_model_one_shot", client=client
            )

    def test_agent_observes_only_answer_to_requested_component(self):
        prediction = {
            "trigger_channel": "Gmail",
            "trigger_function": "Any new attachment in inbox",
            "action_channel": "Google Drive",
            "action_function": "Add row to spreadsheet",
        }
        client = FakeClient([
            {"action": "ask", "component": "trigger_channel"},
            {"action": "commit", "prediction": prediction},
        ])
        result = execute_case(
            case=case(), catalog=CATALOG, arm="bounded_clarification_agent", client=client
        )
        self.assertEqual(result["questions"], ["trigger_channel"])
        self.assertEqual(len(client.calls), 2)
        second_messages = json.dumps(client.calls[1]["messages"])
        self.assertIn("answer for trigger_channel", second_messages)
        self.assertNotIn("answer for action_channel", second_messages)
        self.assertTrue(result["scores"]["joint"])

    def test_repeated_question_is_terminal_failure_without_fallback(self):
        client = FakeClient([
            {"action": "ask", "component": "trigger_channel"},
            {"action": "ask", "component": "trigger_channel"},
        ])
        result = execute_case(
            case=case(), catalog=CATALOG, arm="bounded_clarification_agent", client=client
        )
        self.assertEqual(result["terminal_status"], "protocol_failure")
        self.assertIsNone(result["prediction"])
        self.assertFalse(result["scores"]["joint"])
        self.assertEqual(len(client.calls), 2)

    def test_out_of_catalog_commit_is_terminal_failure(self):
        bad = {
            "trigger_channel": "invented", "trigger_function": "New feed item",
            "action_channel": "Email", "action_function": "Send me an email",
        }
        result = execute_case(
            case=case(), catalog=CATALOG, arm="same_model_one_shot",
            client=FakeClient([{"action": "commit", "prediction": bad}]),
        )
        self.assertEqual(result["terminal_status"], "protocol_failure")
        self.assertFalse(any(result["scores"].values()))

    def test_prompt_has_same_evidence_and_no_gold(self):
        one = build_initial_messages(query="q", catalog=CATALOG, arm="same_model_one_shot")
        agent = build_initial_messages(query="q", catalog=CATALOG, arm="bounded_clarification_agent")
        self.assertEqual(one[1], agent[1])
        self.assertNotIn("gold", json.dumps(one))

    def test_final_preflight_requires_150_unique_cases(self):
        cases = [case(f"yao:{index}") for index in range(150)]
        validate_final_cases(cases)
        with self.assertRaises(ValueError):
            validate_final_cases(cases[:-1])

    def test_exact_scoring_and_wilson(self):
        correct = EndpointLabels(**{
            "trigger_channel": "Gmail", "trigger_function": "Any new attachment in inbox",
            "action_channel": "Google Drive", "action_function": "Add row to spreadsheet",
        })
        self.assertTrue(score_prediction(correct, case())["joint"])
        changed = correct.model_copy(update={"trigger_channel": "gmail"})
        self.assertFalse(score_prediction(changed, case())["trigger_channel"])
        low, high = wilson_interval(75, 150)
        self.assertLess(low, 50)
        self.assertGreater(high, 50)

    def test_aggregate_has_raw_counts_strata_usage_and_pairs(self):
        good = {
            "case_id": "yao:1", "stratum": "VI-4", "terminal": True,
            "terminal_status": "committed", "failure_code": None,
            "scores": {**{component: True for component in COMPONENTS}, "joint": True},
            "question_count": 1,
            "ask_policy": {"true_positive": 1, "false_positive": 0, "false_negative": 0, "true_negative": 3},
            "usage": {"semantic_calls": 2, "prompt_tokens": 20, "completion_tokens": 10,
                      "physical_attempts": 2, "provider_latency_ms": 40, "queue_wait_ms": 4,
                      "strict_json_outputs": 2},
        }
        bad = {**good, "scores": {**{component: False for component in COMPONENTS}, "joint": False}}
        aggregate = aggregate_records([good], intended_n=1, paired_reference=[bad])
        self.assertEqual(aggregate["raw_numerators"]["joint"], 1)
        self.assertEqual(aggregate["paired_outcomes"]["joint"]["current_only"], 1)
        self.assertEqual(aggregate["by_stratum"]["VI-4"]["n"], 1)

    def test_client_boundary_is_public_interactive_source(self):
        client = FakeClient([{
            "action": "commit",
            "prediction": {
                "trigger_channel": "Gmail", "trigger_function": "Any new attachment in inbox",
                "action_channel": "Google Drive", "action_function": "Add row to spreadsheet",
            },
        }])
        execute_case(case=case(), catalog=CATALOG, arm="same_model_one_shot", client=client)
        call = client.calls[0]
        self.assertIs(call["data_classification"], DataClassification.PUBLIC)
        self.assertIs(call["data_source"], DataSource.INTERACTIVE_IFTTT)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.configuration_agent import run_configuration_agent
from farm_r9.privacy import DataClassification, DataSource


def _case() -> dict:
    return {
        "benchmark": "recipegen_gold", "case_id": "case-1", "input": {"query": "Save recipe posts"},
        "public_evidence": {
            "trigger_candidates": [{
                "alias": "T01", "side": "trigger", "service": "Feed", "function": "New item",
                "fields": [], "ingredients": [{"slug": "title", "label": "Title", "value_type": "string"}],
            }],
            "action_candidates": [{
                "alias": "A01", "side": "action", "service": "Notes", "function": "Create note",
                "fields": [
                    {"slug": "body", "label": "Body", "required": True, "bindable": True, "value_type": "string", "resource_like": False, "auth_like": False},
                    {"slug": "folder", "label": "Folder", "required": False, "bindable": False, "value_type": "string", "resource_like": True, "auth_like": False},
                ], "ingredients": [],
            }],
        },
        "configuration_evidence": {"clarification_answers": {}},
    }


def _response(content: dict | str, ordinal: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        content=json.dumps(content) if isinstance(content, dict) else content,
        request_sha256=str(ordinal) * 64, response_sha256="b" * 64,
        prompt_tokens=1, completion_tokens=2, provider_latency_ms=3.0,
        queue_wait_ms=0.0, physical_attempts=1, cache_hit=False,
    )


class FakeClient:
    def __init__(self, responses: list[dict | str]) -> None:
        self.responses = list(responses)
        self.messages = []
        self.call_kwargs = []

    def chat(self, **kwargs):
        self.call_kwargs.append(kwargs)
        self.messages.append(kwargs["messages"])
        return _response(self.responses.pop(0), len(self.messages))


class ConfigurationAgentTests(unittest.TestCase):
    def test_credential_shaped_simulator_answer_is_rejected_before_model(self) -> None:
        case = _case()
        case["configuration_evidence"] = {
            "clarification_answers": {"action:folder": "api_key=definitely-not-safe"}
        }
        client = FakeClient([])
        with self.assertRaises(ValueError):
            run_configuration_agent(
                client=client, case=case, trigger_alias="T01", action_alias="A01",
                benchmark="recipegen_gold", classification=DataClassification.PUBLIC,
                data_source=DataSource.RECIPEGEN, arm="bounded_configuration_agent",
            )
        self.assertEqual(client.messages, [])

    def test_grounded_binding_and_explicit_optional_omit_compile(self) -> None:
        client = FakeClient([{
            "action": "commit", "draft": {
                "selection": {"trigger_alias": "T01", "action_alias": "A01"},
                "trigger_fields": [],
                "action_fields": [
                    {"field_slug": "body", "source": {"kind": "trigger_output", "ingredient_slug": "title"}},
                    {"field_slug": "folder", "source": {"kind": "omit", "reason": "not requested"}},
                ],
                "preview": "Save each item title as a note.", "evidence_aliases": ["T01", "A01"],
            },
        }])
        result = run_configuration_agent(
            client=client, case=_case(), trigger_alias="T01", action_alias="A01",
            benchmark="recipegen_gold", classification=DataClassification.PUBLIC,
            data_source=DataSource.RECIPEGEN, arm="single_shot_configurator",
            think=False,
        )
        self.assertIsNone(result.protocol_failure)
        self.assertTrue(result.compilation and result.compilation.executable_ready)
        self.assertEqual(result.metrics()["accepted_fabrication_count"], 0)
        self.assertFalse(result.metrics()["semantic_accuracy_supported"])
        self.assertIs(client.call_kwargs[0]["think"], False)
        system_prompt = client.messages[0][0]["content"]
        self.assertIn("Asking is unavailable in this one-shot arm", system_prompt)
        self.assertIn("Commit now", system_prompt)
        self.assertIn("needs_input", system_prompt)
        self.assertNotIn('{"action":"ask"', system_prompt)

    def test_ask_without_supplied_answer_must_end_needs_input(self) -> None:
        commit = {
            "action": "commit", "draft": {
                "selection": {"trigger_alias": "T01", "action_alias": "A01"}, "trigger_fields": [],
                "action_fields": [
                    {"field_slug": "body", "source": {"kind": "trigger_output", "ingredient_slug": "title"}},
                    {"field_slug": "folder", "source": {"kind": "needs_input", "question": "Which folder should I use?"}},
                ], "preview": "Save each title; a folder is still needed.", "evidence_aliases": ["T01", "A01"],
            },
        }
        client = FakeClient([
            {"action": "ask", "component": "action:folder", "question": "Which folder should I use?"},
            commit,
        ])
        result = run_configuration_agent(
            client=client, case=_case(), trigger_alias="T01", action_alias="A01",
            benchmark="recipegen_gold", classification=DataClassification.PUBLIC,
            data_source=DataSource.RECIPEGEN, arm="bounded_configuration_agent",
            max_questions=1,
        )
        self.assertTrue(result.compilation and result.compilation.valid)
        self.assertFalse(result.compilation.executable_ready)
        self.assertEqual(result.metrics()["answered_question_count"], 0)
        system_prompt = client.messages[0][0]["content"]
        self.assertIn('{"action":"ask"', system_prompt)
        self.assertIn("at most 1 clarification question before committing", system_prompt)
        second_grounding = json.loads(client.messages[1][-1]["content"])["grounding_text"]
        self.assertNotIn("No supplied answer", second_grounding)

    def test_ungrounded_literal_gets_only_one_error_based_repair(self) -> None:
        invalid = {
            "action": "commit", "draft": {
                "selection": {"trigger_alias": "T01", "action_alias": "A01"}, "trigger_fields": [],
                "action_fields": [
                    {"field_slug": "body", "source": {"kind": "query_literal", "start": 0, "end": 4, "text": "Fake"}},
                    {"field_slug": "folder", "source": {"kind": "omit", "reason": "not requested"}},
                ], "preview": "Save it.", "evidence_aliases": ["T01", "A01"],
            },
        }
        valid = json.loads(json.dumps(invalid))
        valid["draft"]["action_fields"][0]["source"] = {"kind": "trigger_output", "ingredient_slug": "title"}
        client = FakeClient([invalid, valid])
        result = run_configuration_agent(
            client=client, case=_case(), trigger_alias="T01", action_alias="A01",
            benchmark="recipegen_gold", classification=DataClassification.PUBLIC,
            data_source=DataSource.RECIPEGEN, arm="bounded_configuration_agent",
        )
        self.assertTrue(result.compilation and result.compilation.valid)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.metrics()["fabrication_attempt_count"], 1)
        self.assertIn("actual_compiler_errors", client.messages[1][-1]["content"])

    def test_fixed_endpoint_change_fails_closed(self) -> None:
        client = FakeClient([{
            "action": "commit", "draft": {
                "selection": {"trigger_alias": "T02", "action_alias": "A01"},
                "trigger_fields": [], "action_fields": [], "preview": "Changed.", "evidence_aliases": [],
            },
        }])
        result = run_configuration_agent(
            client=client, case=_case(), trigger_alias="T01", action_alias="A01",
            benchmark="recipegen_gold", classification=DataClassification.PUBLIC,
            data_source=DataSource.RECIPEGEN, arm="single_shot_configurator",
        )
        self.assertEqual(result.protocol_failure, "compiler_validation_failed")
        self.assertEqual(result.validation_errors[0]["code"], "changed_fixed_trigger")


if __name__ == "__main__":
    unittest.main()

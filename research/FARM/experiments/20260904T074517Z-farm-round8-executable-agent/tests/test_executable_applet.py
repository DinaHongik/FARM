from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from executable_applet import (  # noqa: E402
    ActionEndpointRef,
    AppletCompiler,
    AppletProgram,
    CandidateSet,
    CompilationError,
    ConstantSource,
    ContextSource,
    EndpointField,
    FieldBinding,
    InferenceLeakError,
    RepairState,
    SandboxConnectorRegistry,
    TriggerEndpointRef,
    TriggerOutputSource,
    build_inference_envelope,
    deterministic_canary,
    hydrate_native_endpoint,
    run_applet_payload,
)


def trigger_candidate(*, rank: int = 1, identity: str = "trigger://mail/new-message") -> dict:
    return {
        "url": identity,
        "kind": "trigger",
        "channel": "mail",
        "function_name": "New message",
        "description": "Fires when a new message arrives.",
        "retrieval_rank": rank,
        "input_fields": [],
        "ingredients": [
            {"slug": "event.message", "label": "Message", "type": "String"},
            {"slug": "event.count", "label": "Count", "type": "Integer"},
        ],
    }


def action_candidate(*, rank: int = 1, identity: str = "action://chat/post") -> dict:
    return {
        "url": identity,
        "kind": "action",
        "channel": "chat",
        "function_name": "Post message",
        "retrieval_rank": rank,
        "ingredients": [],
        "input_fields": [
            {"slug": "body", "label": "Body", "type": "String", "required": True},
            {"slug": "room", "label": "Room", "type": "String", "required": True},
            {"slug": "urgent", "label": "Urgent", "type": "Boolean", "required": True},
        ],
    }


def candidate_set() -> CandidateSet:
    return CandidateSet.from_native([trigger_candidate()], [action_candidate()])


def valid_program() -> AppletProgram:
    return AppletProgram(
        trigger=TriggerEndpointRef(candidate_id="T01"),
        action=ActionEndpointRef(candidate_id="A01"),
        bindings=(
            FieldBinding(
                target_path="body",
                source=TriggerOutputSource(kind="trigger_output", path="event.message"),
            ),
            FieldBinding(
                target_path="room",
                source=ContextSource(kind="context", path="user.room"),
            ),
            FieldBinding(
                target_path="urgent",
                source=ConstantSource(kind="constant", value=True),
            ),
        ),
    )


class ExecutableAppletTests(unittest.TestCase):
    def test_valid_program_compiles_and_executes_trigger_binding_action(self) -> None:
        candidates = candidate_set()
        compiler = AppletCompiler(
            candidates,
            context_fields=(EndpointField(path="user.room", value_type="string"),),
        )

        compiled = compiler.compile(valid_program())
        result = SandboxConnectorRegistry(candidates).execute(
            compiled,
            context={"user": {"room": "research-lab"}},
        )

        self.assertTrue(result.success)
        self.assertEqual(result.invocation.endpoint_identity, "action://chat/post")
        arguments = {item.path: item.value for item in result.invocation.arguments}
        self.assertEqual(arguments["room"], "research-lab")
        self.assertIs(arguments["urgent"], True)
        self.assertRegex(arguments["body"], r"^canary_[0-9a-f]{12}$")
        self.assertEqual(
            tuple(event.stage for event in result.audit_trace.events),
            (
                "compile_started",
                "compile_succeeded",
                "trigger_fired",
                "binding_resolved",
                "binding_resolved",
                "binding_resolved",
                "action_invoked",
            ),
        )

    def test_compiler_reports_missing_required_binding_as_structured_issue(self) -> None:
        program = valid_program().model_copy(
            update={
                "bindings": tuple(
                    binding
                    for binding in valid_program().bindings
                    if binding.target_path != "body"
                )
            }
        )

        with self.assertRaises(CompilationError) as caught:
            AppletCompiler(
                candidate_set(),
                context_fields=(EndpointField(path="user.room", value_type="string"),),
            ).compile(program)

        issue = next(item for item in caught.exception.issues if item.code == "missing_required_binding")
        self.assertEqual(issue.location, "action.inputs.body")
        self.assertEqual(caught.exception.as_dict()["issues"][0]["code"], "missing_required_binding")

    def test_compiler_rejects_type_mismatch(self) -> None:
        action = action_candidate()
        action["input_fields"] = [
            {"slug": "count", "label": "Count", "type": "Integer", "required": True}
        ]
        candidates = CandidateSet.from_native([trigger_candidate()], [action])
        program = AppletProgram(
            trigger=TriggerEndpointRef(candidate_id="T01"),
            action=ActionEndpointRef(candidate_id="A01"),
            bindings=(
                FieldBinding(
                    target_path="count",
                    source=TriggerOutputSource(kind="trigger_output", path="event.message"),
                ),
            ),
        )

        with self.assertRaises(CompilationError) as caught:
            AppletCompiler(candidates).compile(program)

        mismatch = next(item for item in caught.exception.issues if item.code == "type_mismatch")
        self.assertEqual((mismatch.expected, mismatch.observed), ("integer", "string"))

    def test_compiler_rejects_endpoint_outside_retrieved_candidates(self) -> None:
        program = valid_program().model_copy(
            update={"action": ActionEndpointRef(candidate_id="A02")}
        )

        with self.assertRaises(CompilationError) as caught:
            AppletCompiler(candidate_set()).compile(program)

        issue = caught.exception.issues[0]
        self.assertEqual(issue.code, "candidate_not_retrieved")
        self.assertEqual(issue.location, "action.candidate_id")

    def test_paths_and_candidate_aliases_block_escape_and_injection(self) -> None:
        unsafe_paths = ("../secret", "event.__class__", "event/secret", "constructor.value")
        for path in unsafe_paths:
            with self.subTest(path=path), self.assertRaises(ValidationError):
                TriggerOutputSource(kind="trigger_output", path=path)
        with self.assertRaises(ValidationError):
            ActionEndpointRef(candidate_id="A01/../../secret")

    def test_schema_hydration_and_canary_generation_are_deterministic(self) -> None:
        first = trigger_candidate()
        second = copy.deepcopy(first)
        second["ingredients"].reverse()

        hydrated_first = hydrate_native_endpoint(first, "trigger")
        hydrated_second = hydrate_native_endpoint(second, "trigger")

        self.assertEqual(hydrated_first, hydrated_second)
        self.assertEqual(deterministic_canary(hydrated_first), deterministic_canary(hydrated_second))
        self.assertEqual(
            deterministic_canary(hydrated_first),
            deterministic_canary(hydrated_first),
        )

    def test_gold_and_credentials_cannot_cross_inference_seam(self) -> None:
        unsafe_case = {
            "group_id": "dev-1",
            "query": "When mail arrives, post it",
            "evaluation": {"gold_pair": ["trigger://gold", "action://gold"]},
            "api_key": "must-not-cross",
        }
        with self.assertRaises(InferenceLeakError) as caught:
            build_inference_envelope(unsafe_case, candidate_set())
        self.assertEqual(
            caught.exception.paths,
            ("case.evaluation.gold_pair", "case.api_key"),
        )

        envelope = build_inference_envelope(
            {"group_id": "dev-1", "query": "When mail arrives, post it"},
            candidate_set(),
        )
        payload = json.dumps(envelope.model_dump(mode="json"), sort_keys=True)
        self.assertNotIn("gold", payload.casefold())
        self.assertNotIn("trigger://", payload)
        self.assertNotIn("action://", payload)
        self.assertNotIn("retrieval_rank", payload)
        self.assertEqual(
            envelope.trigger_candidates[0].description,
            "Fires when a new message arrives.",
        )

    def test_repair_state_permits_exactly_one_repair(self) -> None:
        issue = ()
        initial = RepairState(program=valid_program())
        repaired = initial.apply_repair(valid_program(), issue)

        self.assertEqual(repaired.repairs_used, 1)
        self.assertEqual(initial.repairs_used, 0)
        with self.assertRaisesRegex(ValueError, "exhausted"):
            repaired.apply_repair(valid_program(), issue)

    def test_json_payload_seam_runs_without_live_connectors(self) -> None:
        payload = {
            "trigger_candidates": [trigger_candidate()],
            "action_candidates": [action_candidate()],
            "context_fields": [
                EndpointField(path="user.room", value_type="string").model_dump(mode="json")
            ],
            "program": valid_program().model_dump(mode="json"),
            "context": {"user": {"room": "research-lab"}},
        }

        result = run_applet_payload(payload)

        self.assertTrue(result.success)
        with self.assertRaisesRegex(ValueError, "unexpected execution payload"):
            run_applet_payload({**payload, "live_connector": "forbidden"})

    def test_audit_trace_is_frozen(self) -> None:
        candidates = candidate_set()
        compiled = AppletCompiler(
            candidates,
            context_fields=(EndpointField(path="user.room", value_type="string"),),
        ).compile(valid_program())
        original = compiled.audit_trace

        with self.assertRaises(ValidationError):
            original.events[0].stage = "action_invoked"  # type: ignore[misc]
        self.assertEqual(original.events[0].stage, "compile_started")


if __name__ == "__main__":
    unittest.main()

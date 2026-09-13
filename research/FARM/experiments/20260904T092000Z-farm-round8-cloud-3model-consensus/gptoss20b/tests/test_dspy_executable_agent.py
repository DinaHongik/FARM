from __future__ import annotations

import importlib.metadata
import sys
import threading
import unittest
from pathlib import Path

try:
    import dspy
    from pydantic import ValidationError
except ModuleNotFoundError as error:  # The Round8 image installs the pinned requirement.
    raise unittest.SkipTest(
        "the pinned Round8 DSPy/Pydantic environment is not installed locally"
    ) from error


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from dspy_executable_agent import (  # noqa: E402
    BinderCriticResponse,
    BoundedDSPyApplet,
    DSPY_REQUIRED_VERSION,
    EndpointProposal,
    FactorizedDSPyApplet,
    PlannerResponse,
    PortUsage,
    SpecialistResponse,
    TriggerConsensusDSPyApplet,
    TriggerConsensusResponse,
)
from executable_applet import (  # noqa: E402
    ActionEndpointRef,
    AppletProgram,
    CandidateSet,
    ConstantSource,
    ContextSource,
    EndpointField,
    FieldBinding,
    TriggerEndpointRef,
    TriggerOutputSource,
)


def trigger_candidate(
    *, rank: int = 1, identity: str = "trigger://mail/new-message"
) -> dict:
    return {
        "url": identity,
        "kind": "trigger",
        "channel": "mail",
        "function_name": "New message",
        "retrieval_rank": rank,
        "input_fields": [],
        "ingredients": [
            {"slug": "event.message", "label": "Message", "type": "String"},
            {"slug": "event.count", "label": "Count", "type": "Integer"},
        ],
    }


def action_candidate(
    *, rank: int = 1, identity: str = "action://chat/post"
) -> dict:
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
    return CandidateSet.from_native(
        [
            trigger_candidate(),
            trigger_candidate(rank=2, identity="trigger://calendar/new-event"),
        ],
        [
            action_candidate(),
            action_candidate(rank=2, identity="action://archive/store"),
        ],
    )


def consensus_candidate_set() -> CandidateSet:
    triggers = [
        trigger_candidate(
            rank=index,
            identity=f"trigger://mail/consensus-{index}",
        )
        for index in range(1, 11)
    ]
    return CandidateSet.from_native(
        triggers,
        [action_candidate(), action_candidate(rank=2, identity="action://archive/store")],
    )


def consensus_context_fields() -> tuple[EndpointField, ...]:
    return tuple(
        EndpointField(path=f"config.A01.{path}", value_type=value_type, required=True)
        for path, value_type in (
            ("body", "string"),
            ("room", "string"),
            ("urgent", "boolean"),
        )
    )


def consensus_context() -> dict:
    return {
        "config": {
            "A01": {
                "body": "message",
                "room": "research-lab",
                "urgent": False,
            }
        }
    }


def context_field() -> EndpointField:
    return EndpointField(path="user.room", value_type="string", required=True)


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


def context_free_program() -> AppletProgram:
    return valid_program().model_copy(
        update={
            "bindings": tuple(
                FieldBinding(
                    target_path="room",
                    source=ConstantSource(kind="constant", value="research-lab"),
                )
                if binding.target_path == "room"
                else binding
                for binding in valid_program().bindings
            )
        }
    )


def missing_body_program() -> AppletProgram:
    return valid_program().model_copy(
        update={
            "bindings": tuple(
                binding for binding in valid_program().bindings if binding.target_path != "body"
            )
        }
    )


def offline_usage() -> PortUsage:
    return PortUsage(provider_requests=0)


class DSPyExecutableAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = {
            "request_id": "round8-case-1",
            "query": "When a new email arrives, post its body to the research room",
        }
        self.candidates = candidate_set()

    def test_direct_success_is_one_explicit_call_compile_and_execution(self) -> None:
        observed_requests = []

        def planner(request):
            observed_requests.append(request)
            self.assertNotIn("research-lab", request.model_dump_json())
            return PlannerResponse(program=valid_program(), usage=offline_usage())

        agent = BoundedDSPyApplet(planner, max_repairs=0)
        prediction = agent(
            case=self.case,
            candidates=self.candidates,
            context_fields=(context_field(),),
            context={"user": {"room": "research-lab"}},
        )
        result = prediction.result

        self.assertIsInstance(agent, dspy.Module)
        self.assertIsInstance(prediction, dspy.Prediction)
        self.assertEqual(len(observed_requests), 1)
        self.assertEqual(result.terminal_status, "executed")
        self.assertEqual(result.compilation_attempts, 1)
        self.assertEqual(result.execution_attempts, 1)
        self.assertEqual(result.repairs_attempted, 0)
        self.assertEqual(result.call_totals.logical_calls, 1)
        self.assertEqual(result.call_totals.provider_requests, 0)
        self.assertEqual(result.call_totals.unreported_usage_calls, 0)
        self.assertEqual(result.calls[0].role, "planner")
        self.assertEqual(list(agent.named_predictors()), [])
        self.assertEqual(result.runtime_policy.dspy_lm_calls, 0)
        self.assertEqual(result.runtime_policy.cache_reads, 0)
        self.assertEqual(result.runtime_policy.dspy_retries, 0)
        self.assertEqual(result.runtime_policy.adapter_fallbacks, 0)

    def test_compile_failure_gets_exactly_one_repair(self) -> None:
        repair_requests = []

        def repairer(request):
            repair_requests.append(request)
            return PlannerResponse(program=valid_program(), usage=offline_usage())

        agent = BoundedDSPyApplet(
            lambda request: PlannerResponse(
                program=missing_body_program(), usage=offline_usage()
            ),
            repairer=repairer,
        )
        result = agent.forward(
            case=self.case,
            candidates=self.candidates,
            context_fields=(context_field(),),
            context={"user": {"room": "research-lab"}},
        ).result

        self.assertEqual(result.terminal_status, "executed")
        self.assertEqual(result.compilation_attempts, 2)
        self.assertEqual(result.execution_attempts, 1)
        self.assertEqual(result.repairs_attempted, 1)
        self.assertEqual(tuple(call.role for call in result.calls), ("planner", "repair"))
        self.assertEqual(result.call_totals.logical_calls, 2)
        self.assertEqual(len(repair_requests), 1)
        self.assertEqual(repair_requests[0].failed_stage, "compilation")
        self.assertIn(
            "missing_required_binding",
            {issue.code for issue in repair_requests[0].issues},
        )

    def test_execution_failure_is_restarted_once_with_repaired_program(self) -> None:
        seen = []

        def repairer(request):
            seen.append(request)
            return PlannerResponse(program=context_free_program(), usage=offline_usage())

        result = BoundedDSPyApplet(
            lambda request: PlannerResponse(program=valid_program(), usage=offline_usage()),
            repairer=repairer,
        ).forward(
            case=self.case,
            candidates=self.candidates,
            context_fields=(context_field(),),
            context={},
        ).result

        self.assertEqual(result.terminal_status, "executed")
        self.assertEqual(result.compilation_attempts, 2)
        self.assertEqual(result.execution_attempts, 2)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].failed_stage, "execution")
        self.assertEqual(seen[0].issues[0].code, "missing_runtime_source")
        self.assertEqual(len(result.attempted_programs), 2)

    def test_failed_repair_stops_and_returns_non_executable_top1_fallback(self) -> None:
        repair_calls = 0

        def repairer(request):
            nonlocal repair_calls
            repair_calls += 1
            return PlannerResponse(program=missing_body_program(), usage=offline_usage())

        result = BoundedDSPyApplet(
            lambda request: PlannerResponse(
                program=missing_body_program(), usage=offline_usage()
            ),
            repairer=repairer,
        ).forward(
            case=self.case,
            candidates=self.candidates,
            context_fields=(context_field(),),
            context={"user": {"room": "research-lab"}},
        ).result

        self.assertEqual(repair_calls, 1)
        self.assertEqual(result.terminal_status, "safe_fallback")
        self.assertIsNone(result.program)
        self.assertIsNone(result.execution)
        self.assertFalse(result.fallback.executable)
        self.assertEqual(result.fallback.trigger_candidate_id, "T01")
        self.assertEqual(result.fallback.action_candidate_id, "A01")
        self.assertEqual(result.call_totals.logical_calls, 2)
        self.assertEqual(result.repairs_attempted, 1)
        self.assertEqual(result.compilation_attempts, 2)
        self.assertEqual(result.execution_attempts, 0)

    def test_planner_exception_is_audited_without_fabricated_usage(self) -> None:
        def planner(request):
            raise RuntimeError("provider detail must not leak")

        result = BoundedDSPyApplet(planner, max_repairs=0).forward(
            case=self.case,
            candidates=self.candidates,
        ).result

        self.assertEqual(result.terminal_status, "safe_fallback")
        self.assertEqual(result.calls[0].status, "exception")
        self.assertIsNone(result.calls[0].usage)
        self.assertEqual(result.call_totals.logical_calls, 1)
        self.assertEqual(result.call_totals.unreported_usage_calls, 1)
        self.assertNotIn("provider detail", result.model_dump_json())

    def test_direct_fallback_uses_explicit_baseline_not_presentation_first(self) -> None:
        def planner(request):
            raise RuntimeError("force fallback")

        result = BoundedDSPyApplet(planner, max_repairs=0).forward(
            case=self.case,
            candidates=self.candidates,
            baseline_trigger_id="T02",
            baseline_action_id="A02",
        ).result

        self.assertEqual(result.terminal_status, "safe_fallback")
        self.assertEqual(result.fallback.trigger_candidate_id, "T02")
        self.assertEqual(result.fallback.action_candidate_id, "A02")

    def test_explicit_baseline_is_side_checked_before_direct_planner_call(self) -> None:
        planner_calls = 0

        def planner(request):
            nonlocal planner_calls
            planner_calls += 1
            return PlannerResponse(program=context_free_program(), usage=offline_usage())

        agent = BoundedDSPyApplet(planner, max_repairs=0)
        with self.assertRaisesRegex(ValueError, "retrieved trigger"):
            agent.forward(
                case=self.case,
                candidates=self.candidates,
                baseline_trigger_id="A01",
                baseline_action_id="A02",
            )
        with self.assertRaisesRegex(ValueError, "retrieved trigger"):
            agent.forward(
                case=self.case,
                candidates=self.candidates,
                baseline_trigger_id="T99",
                baseline_action_id="A02",
            )
        self.assertEqual(planner_calls, 0)

    def test_factorized_specialists_are_parallel_and_binder_is_third_call(self) -> None:
        barrier = threading.Barrier(2, timeout=3.0)
        specialist_threads = []
        binder_requests = []

        def trigger_specialist(request):
            self.assertEqual(request.side, "trigger")
            self.assertTrue(all(item.candidate_id.startswith("T") for item in request.candidates))
            specialist_threads.append(threading.current_thread().name)
            barrier.wait()
            return SpecialistResponse(
                proposal=EndpointProposal(
                    side="trigger", candidate_id="T01", confidence=0.9
                ),
                usage=offline_usage(),
            )

        def action_specialist(request):
            self.assertEqual(request.side, "action")
            self.assertTrue(all(item.candidate_id.startswith("A") for item in request.candidates))
            specialist_threads.append(threading.current_thread().name)
            barrier.wait()
            return SpecialistResponse(
                proposal=EndpointProposal(side="action", candidate_id="A01", confidence=0.8),
                usage=offline_usage(),
            )

        def binder_critic(request):
            binder_requests.append(request)
            self.assertEqual(request.trigger.candidate_id, "T01")
            self.assertEqual(request.action.candidate_id, "A01")
            return BinderCriticResponse(
                approved=True,
                program=context_free_program(),
                usage=offline_usage(),
            )

        agent = FactorizedDSPyApplet(
            trigger_specialist,
            action_specialist,
            binder_critic,
            parallel_specialists=True,
        )
        result = agent.forward(case=self.case, candidates=self.candidates).result

        self.assertEqual(result.terminal_status, "executed")
        self.assertEqual(len(set(specialist_threads)), 2)
        self.assertEqual(len(binder_requests), 1)
        self.assertEqual(
            tuple(call.role for call in result.calls),
            ("trigger_specialist", "action_specialist", "binder_critic"),
        )
        self.assertEqual(result.call_totals.logical_calls, 3)
        self.assertEqual(result.call_totals.provider_requests, 0)
        self.assertEqual(result.compilation_attempts, 1)
        self.assertEqual(result.execution_attempts, 1)
        self.assertEqual(result.repairs_attempted, 0)
        self.assertEqual(list(agent.named_predictors()), [])

    def test_invalid_specialist_proposal_blocks_binder_and_falls_back(self) -> None:
        binder_calls = 0

        def binder_critic(request):
            nonlocal binder_calls
            binder_calls += 1
            raise AssertionError("binder must not run")

        result = FactorizedDSPyApplet(
            lambda request: SpecialistResponse(
                proposal=EndpointProposal(side="action", candidate_id="A01", confidence=0.9),
                usage=offline_usage(),
            ),
            lambda request: SpecialistResponse(
                proposal=EndpointProposal(side="action", candidate_id="A01", confidence=0.9),
                usage=offline_usage(),
            ),
            binder_critic,
            parallel_specialists=False,
        ).forward(case=self.case, candidates=self.candidates).result

        self.assertEqual(binder_calls, 0)
        self.assertEqual(result.terminal_status, "safe_fallback")
        self.assertEqual(result.call_totals.logical_calls, 2)
        self.assertIn("invalid_trigger_proposal", {issue.code for issue in result.issues})

    def test_factorized_fallback_uses_explicit_hidden_rank_baseline(self) -> None:
        result = FactorizedDSPyApplet(
            lambda request: SpecialistResponse(
                proposal=EndpointProposal(side="trigger", candidate_id="T01", confidence=0.9),
                usage=offline_usage(),
            ),
            lambda request: SpecialistResponse(
                proposal=EndpointProposal(side="action", candidate_id="A01", confidence=0.9),
                usage=offline_usage(),
            ),
            lambda request: BinderCriticResponse(
                approved=False,
                program=None,
                rejection_code="unsafe_binding",
                usage=offline_usage(),
            ),
            parallel_specialists=False,
        ).forward(
            case=self.case,
            candidates=self.candidates,
            baseline_trigger_id="T02",
            baseline_action_id="A02",
        ).result

        self.assertEqual(result.terminal_status, "safe_fallback")
        self.assertEqual(result.fallback.trigger_candidate_id, "T02")
        self.assertEqual(result.fallback.action_candidate_id, "A02")
        self.assertEqual(result.call_totals.logical_calls, 3)

    def test_explicit_baseline_is_side_checked_before_factorized_calls(self) -> None:
        specialist_calls = 0

        def specialist(request):
            nonlocal specialist_calls
            specialist_calls += 1
            raise AssertionError("must not run")

        agent = FactorizedDSPyApplet(
            specialist,
            specialist,
            lambda request: (_ for _ in ()).throw(AssertionError("must not run")),
        )
        with self.assertRaisesRegex(ValueError, "retrieved action"):
            agent.forward(
                case=self.case,
                candidates=self.candidates,
                baseline_trigger_id="T02",
                baseline_action_id="T01",
            )
        self.assertEqual(specialist_calls, 0)

    def test_binder_cannot_silently_replace_specialist_endpoints(self) -> None:
        changed = context_free_program().model_copy(
            update={"action": ActionEndpointRef(candidate_id="A02")}
        )
        result = FactorizedDSPyApplet(
            lambda request: SpecialistResponse(
                proposal=EndpointProposal(side="trigger", candidate_id="T01", confidence=1.0),
                usage=offline_usage(),
            ),
            lambda request: SpecialistResponse(
                proposal=EndpointProposal(side="action", candidate_id="A01", confidence=1.0),
                usage=offline_usage(),
            ),
            lambda request: BinderCriticResponse(
                approved=True, program=changed, usage=offline_usage()
            ),
            parallel_specialists=False,
        ).forward(case=self.case, candidates=self.candidates).result

        self.assertEqual(result.terminal_status, "safe_fallback")
        self.assertEqual(result.compilation_attempts, 0)
        self.assertIn(
            "binder_changed_specialist_selection",
            {issue.code for issue in result.issues},
        )

    def test_trigger_consensus_executes_only_agreed_trigger_with_baseline_action(self) -> None:
        observed_views = []

        def schema_selector(request):
            observed_views.append(request.view)
            return TriggerConsensusResponse(
                deferred=False,
                candidate_id="T02",
                usage=offline_usage(),
            )

        def fused_selector(request):
            observed_views.append(request.view)
            return TriggerConsensusResponse(
                deferred=False,
                candidate_id="T02",
                usage=offline_usage(),
            )

        candidates = consensus_candidate_set()
        agent = TriggerConsensusDSPyApplet(schema_selector, fused_selector)
        result = agent.forward(
            case=self.case,
            candidates=candidates,
            schema_m5_candidate_ids=("T01", "T02", "T03", "T04", "T05"),
            fused_m10_candidate_ids=tuple(f"T{index:02d}" for index in range(1, 11)),
            trigger_seed_support={f"T{index:02d}": 3 for index in range(1, 11)},
            routing_score=0.5,
            context_fields=consensus_context_fields(),
            context=consensus_context(),
            baseline_trigger_id="T01",
            baseline_action_id="A01",
        ).result

        self.assertEqual(observed_views, ["schema_m5", "fused_m10"])
        self.assertEqual(result.terminal_status, "executed")
        self.assertEqual(result.program.trigger.candidate_id, "T02")
        self.assertEqual(result.program.action.candidate_id, "A01")
        self.assertEqual(result.call_totals.logical_calls, 2)
        self.assertEqual(
            tuple(call.role for call in result.calls),
            ("trigger_schema_m5", "trigger_fused_m10"),
        )
        self.assertTrue(result.trigger_consensus_trace.exact_trigger_consensus)
        self.assertTrue(result.trigger_consensus_trace.baseline_action_retained)
        self.assertEqual(
            result.trigger_consensus_trace.decision_reason,
            "consensus_compiled_and_executed",
        )
        self.assertEqual(list(agent.named_predictors()), [])

    def test_trigger_consensus_router_and_first_view_are_true_early_exits(self) -> None:
        calls = []

        def selector(request):
            calls.append(request.view)
            return TriggerConsensusResponse(
                deferred=False,
                candidate_id="T01",
                usage=offline_usage(),
            )

        candidates = consensus_candidate_set()
        common = {
            "case": self.case,
            "candidates": candidates,
            "schema_m5_candidate_ids": ("T01", "T02", "T03", "T04", "T05"),
            "fused_m10_candidate_ids": tuple(f"T{index:02d}" for index in range(1, 11)),
            "trigger_seed_support": {f"T{index:02d}": 3 for index in range(1, 11)},
            "baseline_trigger_id": "T01",
            "baseline_action_id": "A01",
        }
        agent = TriggerConsensusDSPyApplet(selector, selector)
        below = agent.forward(routing_score=0.399, **common).result
        self.assertEqual(below.call_totals.logical_calls, 0)
        self.assertEqual(calls, [])
        self.assertEqual(
            below.trigger_consensus_trace.decision_reason, "router_below_0_40"
        )

        retained = agent.forward(routing_score=0.4, **common).result
        self.assertEqual(retained.call_totals.logical_calls, 1)
        self.assertEqual(calls, ["schema_m5"])
        self.assertEqual(
            retained.trigger_consensus_trace.decision_reason,
            "schema_m5_retained_baseline",
        )

    def test_trigger_consensus_requires_seed_support_and_exact_identity_agreement(self) -> None:
        second_calls = 0

        def first(_request):
            return TriggerConsensusResponse(
                deferred=False, candidate_id="T02", usage=offline_usage()
            )

        def disagree(_request):
            nonlocal second_calls
            second_calls += 1
            return TriggerConsensusResponse(
                deferred=False, candidate_id="T03", usage=offline_usage()
            )

        candidates = consensus_candidate_set()
        common = {
            "case": self.case,
            "candidates": candidates,
            "schema_m5_candidate_ids": ("T01", "T02", "T03", "T04", "T05"),
            "fused_m10_candidate_ids": tuple(f"T{index:02d}" for index in range(1, 11)),
            "routing_score": 0.8,
            "baseline_trigger_id": "T01",
            "baseline_action_id": "A01",
        }
        agent = TriggerConsensusDSPyApplet(first, disagree)
        unsupported = agent.forward(
            trigger_seed_support={"T02": 1}, **common
        ).result
        self.assertEqual(unsupported.call_totals.logical_calls, 1)
        self.assertEqual(second_calls, 0)
        self.assertEqual(
            unsupported.trigger_consensus_trace.decision_reason,
            "trigger_seed_support_below_2",
        )

        disagreement = agent.forward(
            trigger_seed_support={"T02": 2}, **common
        ).result
        self.assertEqual(disagreement.call_totals.logical_calls, 2)
        self.assertEqual(second_calls, 1)
        self.assertEqual(disagreement.terminal_status, "safe_fallback")
        self.assertEqual(disagreement.fallback.trigger_candidate_id, "T01")
        self.assertEqual(disagreement.fallback.action_candidate_id, "A01")
        self.assertEqual(
            disagreement.trigger_consensus_trace.decision_reason,
            "trigger_consensus_disagreement",
        )

    def test_contracts_forbid_hidden_cache_retry_and_adapter_behavior(self) -> None:
        with self.assertRaises(ValidationError):
            PortUsage(provider_requests=1, provider_model="model", cache_hits=1)
        with self.assertRaises(ValidationError):
            PortUsage(provider_requests=1, provider_model="model", cache_writes=1)
        with self.assertRaises(ValidationError):
            PortUsage(provider_requests=1, provider_model="model", retries=1)
        with self.assertRaises(ValidationError):
            PortUsage(provider_requests=1, provider_model="model", adapter_fallbacks=1)
        with self.assertRaises(ValidationError):
            PortUsage(provider_requests=1)
        with self.assertRaises(ValidationError):
            PlannerResponse(
                program=context_free_program(), usage=offline_usage(), unexpected=True
            )

        class HiddenPredictorPort(dspy.Module):
            def named_predictors(self):
                return [("hidden", object())]

            def forward(self, request):
                raise AssertionError("must never be invoked")

        with self.assertRaisesRegex(TypeError, "hidden DSPy predictors"):
            BoundedDSPyApplet(HiddenPredictorPort())

    def test_test_environment_uses_the_pinned_dspy_api(self) -> None:
        self.assertEqual(DSPY_REQUIRED_VERSION, "3.3.1")
        self.assertEqual(importlib.metadata.version("dspy"), DSPY_REQUIRED_VERSION)


if __name__ == "__main__":
    unittest.main()

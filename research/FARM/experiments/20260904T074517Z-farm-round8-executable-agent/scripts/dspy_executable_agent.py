#!/usr/bin/env python3
"""Bounded DSPy orchestration for executable FARM applets.

This module intentionally contains no ``dspy.Predict``, ``dspy.LM``, or
provider client.  Model-facing ports are injected callables, which makes the
orchestration independently testable and prevents DSPy from adding implicit
cache reads, retries, or adapter fallbacks.  A production port must perform at
most one provider request and return :class:`PortUsage`; an offline test port
reports zero provider requests.

The direct arm has one explicit state machine::

    plan -> deterministic compile -> sandbox execute
      -> (optional, at most once) repair -> compile -> execute
      -> non-executable top-1 fallback

The factorized arm calls trigger and action specialists independently (and, by
default, concurrently), then makes one binder/critic call.  Thus it has exactly
two or three logical model-port calls and never hides a fourth call.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import math
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TypeVar

import dspy
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from executable_applet import (
    ActionEndpointRef,
    AppletCompiler,
    AppletProgram,
    CandidateSet,
    CompilationError,
    ContextSource,
    EndpointField,
    ExecutionResult,
    FieldBinding,
    InferenceCandidate,
    InferenceEnvelope,
    SandboxConnectorRegistry,
    SandboxExecutionError,
    TriggerEndpointRef,
    TriggerOutputSource,
    ValidationIssue,
    build_inference_envelope,
)


DSPY_REQUIRED_VERSION = "3.3.1"

CallRole = Literal[
    "planner",
    "repair",
    "trigger_specialist",
    "action_specialist",
    "binder_critic",
    "trigger_schema_m5",
    "trigger_fused_m10",
]
CallStatus = Literal["returned", "invalid_contract", "exception"]
FailureStage = Literal["planning", "compilation", "execution"]
TerminalStatus = Literal["executed", "safe_fallback"]
Side = Literal["trigger", "action"]


class StrictModel(BaseModel):
    """Immutable strict contract for every orchestration boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def require_dspy_331() -> None:
    """Fail closed when the experiment is not running on the pinned API."""

    try:
        installed = version("dspy")
    except PackageNotFoundError:
        installed = getattr(dspy, "__version__", None)
    if installed != DSPY_REQUIRED_VERSION:
        raise RuntimeError(
            f"Round8 requires dspy=={DSPY_REQUIRED_VERSION}; found {installed!r}"
        )


class DSPyRuntimePolicy(StrictModel):
    """Auditable declaration of the framework behavior used by this module."""

    dspy_version: Literal["3.3.1"] = DSPY_REQUIRED_VERSION
    dspy_lm_calls: Literal[0] = 0
    cache_reads: Literal[0] = 0
    cache_writes: Literal[0] = 0
    dspy_retries: Literal[0] = 0
    adapter_fallbacks: Literal[0] = 0


class PortUsage(StrictModel):
    """Usage returned by one injected callable invocation.

    ``provider_requests=0`` is reserved for deterministic/offline callables.
    A real model adapter must report exactly one request.  Retries, cache use,
    and alternate-adapter fallback are forbidden rather than silently counted.
    """

    provider_requests: Literal[0, 1]
    provider_model: str | None = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    cache_hits: Literal[0] = 0
    cache_writes: Literal[0] = 0
    retries: Literal[0] = 0
    adapter_fallbacks: Literal[0] = 0

    @model_validator(mode="after")
    def provider_is_named_exactly_when_called(self) -> "PortUsage":
        if self.provider_requests == 0 and self.provider_model is not None:
            raise ValueError("offline port usage cannot name a provider model")
        if self.provider_requests == 1 and not self.provider_model:
            raise ValueError("a provider request requires provider_model")
        return self


class ModelCallRecord(StrictModel):
    call_index: int = Field(ge=1)
    role: CallRole
    status: CallStatus
    logical_calls: Literal[1] = 1
    usage: PortUsage | None


class CallTotals(StrictModel):
    logical_calls: int = Field(ge=0)
    returned_calls: int = Field(ge=0)
    failed_calls: int = Field(ge=0)
    provider_requests: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    unreported_usage_calls: int = Field(ge=0)
    dspy_lm_calls: Literal[0] = 0
    cache_hits: Literal[0] = 0
    cache_writes: Literal[0] = 0
    retries: Literal[0] = 0
    adapter_fallbacks: Literal[0] = 0


class PublicContextField(StrictModel):
    path: str = Field(min_length=1)
    value_type: str = Field(min_length=1)
    required: bool


class PlanningRequest(StrictModel):
    envelope: InferenceEnvelope
    context_fields: tuple[PublicContextField, ...] = ()


class RepairRequest(StrictModel):
    envelope: InferenceEnvelope
    context_fields: tuple[PublicContextField, ...] = ()
    failed_stage: FailureStage
    failed_program: AppletProgram | None
    issues: tuple[ValidationIssue, ...]


class PlannerResponse(StrictModel):
    program: AppletProgram
    usage: PortUsage


class EndpointSpecialistRequest(StrictModel):
    request_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=16_000)
    side: Side
    candidates: tuple[InferenceCandidate, ...]

    @model_validator(mode="after")
    def candidates_match_side(self) -> "EndpointSpecialistRequest":
        prefix = "T" if self.side == "trigger" else "A"
        if any(not item.candidate_id.startswith(prefix) for item in self.candidates):
            raise ValueError("specialist received a candidate from the other side")
        return self


class EndpointProposal(StrictModel):
    side: Side
    candidate_id: str = Field(pattern=r"^[TA][0-9]{2,4}$")
    confidence: float = Field(ge=0.0, le=1.0)


class SpecialistResponse(StrictModel):
    proposal: EndpointProposal
    usage: PortUsage


ConsensusView = Literal["schema_m5", "fused_m10"]


class TriggerConsensusRequest(StrictModel):
    """Internal request passed to an explicit, one-request native-tool port.

    The port owns the independently aliased public presentation.  These IDs
    are the private candidate-set IDs used only to validate its de-aliased
    response; neither retrieval order nor the baseline is identified here.
    """

    request_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=16_000)
    view: ConsensusView
    candidate_ids: tuple[str, ...] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def trigger_ids_are_unique(self) -> "TriggerConsensusRequest":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("consensus view contains duplicate trigger candidates")
        if any(not re_id.startswith("T") for re_id in self.candidate_ids):
            raise ValueError("consensus view contains a non-trigger candidate")
        return self


class TriggerConsensusResponse(StrictModel):
    """De-aliased result from one model-facing view.

    There is intentionally no confidence field.  Acceptance depends only on
    exact cross-view identity agreement and deterministic harness checks.
    """

    deferred: bool
    candidate_id: str | None = Field(default=None, pattern=r"^T[0-9]{2,4}$")
    usage: PortUsage

    @model_validator(mode="after")
    def defer_and_candidate_are_exclusive(self) -> "TriggerConsensusResponse":
        if self.deferred == (self.candidate_id is not None):
            raise ValueError("deferred response and candidate_id are inconsistent")
        return self


class TriggerConsensusTrace(StrictModel):
    routing_score: float = Field(ge=0.0, le=1.0)
    routing_threshold: Literal[0.4] = 0.4
    routed: bool
    schema_m5_called: bool
    fused_m10_called: bool
    schema_m5_candidate_id: str | None = None
    fused_m10_candidate_id: str | None = None
    selected_trigger_id: str | None = None
    selected_trigger_seed_support: int | None = Field(default=None, ge=0, le=3)
    exact_trigger_consensus: bool
    baseline_action_retained: Literal[True] = True
    decision_reason: str = Field(min_length=1)


class BinderCriticRequest(StrictModel):
    envelope: InferenceEnvelope
    context_fields: tuple[PublicContextField, ...] = ()
    trigger: EndpointProposal
    action: EndpointProposal


class BinderCriticResponse(StrictModel):
    approved: bool
    program: AppletProgram | None
    rejection_code: str | None = None
    usage: PortUsage

    @model_validator(mode="after")
    def verdict_is_complete(self) -> "BinderCriticResponse":
        if self.approved and self.program is None:
            raise ValueError("approved binder verdict requires a program")
        if self.approved and self.rejection_code is not None:
            raise ValueError("approved binder verdict cannot carry a rejection")
        if not self.approved and self.program is not None:
            raise ValueError("rejected binder verdict cannot carry a program")
        if not self.approved and not self.rejection_code:
            raise ValueError("rejected binder verdict requires rejection_code")
        return self


class SafeFallback(StrictModel):
    """Pair-only fallback; deliberately not represented as executable code."""

    trigger_candidate_id: str | None
    action_candidate_id: str | None
    executable: Literal[False] = False
    reason_code: str = Field(min_length=1)


class AgentRunResult(StrictModel):
    terminal_status: TerminalStatus
    program: AppletProgram | None
    execution: ExecutionResult | None
    fallback: SafeFallback | None
    attempted_programs: tuple[AppletProgram, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()
    calls: tuple[ModelCallRecord, ...] = ()
    call_totals: CallTotals
    compilation_attempts: int = Field(ge=0)
    execution_attempts: int = Field(ge=0)
    repairs_attempted: int = Field(ge=0, le=1)
    runtime_policy: DSPyRuntimePolicy = Field(default_factory=DSPyRuntimePolicy)
    trigger_consensus_trace: TriggerConsensusTrace | None = None

    @model_validator(mode="after")
    def result_is_internally_consistent(self) -> "AgentRunResult":
        expected = summarize_calls(self.calls)
        if self.call_totals != expected:
            raise ValueError("call_totals do not match call records")
        expected_indices = tuple(range(1, len(self.calls) + 1))
        if tuple(item.call_index for item in self.calls) != expected_indices:
            raise ValueError("call indices must be contiguous and ordered")
        if self.terminal_status == "executed":
            if self.program is None or self.execution is None or self.fallback is not None:
                raise ValueError("executed result requires program/execution and no fallback")
        else:
            if self.program is not None or self.execution is not None or self.fallback is None:
                raise ValueError("safe fallback cannot claim an executable program")
        return self


class PlannerPort(Protocol):
    def __call__(self, request: PlanningRequest) -> PlannerResponse | Mapping[str, Any]: ...


class RepairPort(Protocol):
    def __call__(self, request: RepairRequest) -> PlannerResponse | Mapping[str, Any]: ...


class SpecialistPort(Protocol):
    def __call__(
        self, request: EndpointSpecialistRequest
    ) -> SpecialistResponse | Mapping[str, Any]: ...


class BinderCriticPort(Protocol):
    def __call__(
        self, request: BinderCriticRequest
    ) -> BinderCriticResponse | Mapping[str, Any]: ...


class TriggerConsensusPort(Protocol):
    def __call__(
        self, request: TriggerConsensusRequest
    ) -> TriggerConsensusResponse | Mapping[str, Any]: ...


def summarize_calls(records: Sequence[ModelCallRecord]) -> CallTotals:
    usages = [record.usage for record in records if record.usage is not None]
    return CallTotals(
        logical_calls=len(records),
        returned_calls=sum(record.status == "returned" for record in records),
        failed_calls=sum(record.status != "returned" for record in records),
        provider_requests=sum(item.provider_requests for item in usages),
        prompt_tokens=sum(item.prompt_tokens for item in usages),
        completion_tokens=sum(item.completion_tokens for item in usages),
        unreported_usage_calls=sum(record.usage is None for record in records),
    )


def _public_context_fields(fields: Sequence[EndpointField]) -> tuple[PublicContextField, ...]:
    return tuple(
        PublicContextField(
            path=field.path,
            value_type=field.value_type,
            required=field.required,
        )
        for field in fields
    )


ResponseT = TypeVar("ResponseT", bound=StrictModel)


def _coerce_response(raw: Any, expected: type[ResponseT]) -> ResponseT:
    if isinstance(raw, expected):
        return raw
    if isinstance(raw, Mapping):
        return expected.model_validate(raw, strict=True)
    # Permit a separately implemented DSPy port to return Prediction(response=...).
    if isinstance(raw, dspy.Prediction) and hasattr(raw, "response"):
        return _coerce_response(raw.response, expected)
    raise TypeError(f"port must return {expected.__name__} or a strict mapping")


def _assert_explicit_port(port: Callable[[Any], Any], *, role: CallRole) -> None:
    """Reject DSPy predictors whose LM behavior would be implicit here."""

    for class_name in ("Predict", "LM"):
        dspy_class = getattr(dspy, class_name, None)
        if isinstance(dspy_class, type) and isinstance(port, dspy_class):
            raise TypeError(
                f"{role} must be an explicit one-request adapter, not dspy.{class_name}"
            )
    if isinstance(port, dspy.Module):
        named_predictors = getattr(port, "named_predictors", None)
        if callable(named_predictors) and list(named_predictors()):
            raise TypeError(f"{role} contains hidden DSPy predictors")


def _call_issue(role: CallRole, status: CallStatus) -> ValidationIssue:
    code = f"{role}_{'contract_invalid' if status == 'invalid_contract' else 'call_failed'}"
    return ValidationIssue(
        code=code,
        location=role,
        message="injected callable failed its typed boundary",
    )


def _invoke_port(
    port: Callable[[Any], Any],
    request: StrictModel,
    expected: type[ResponseT],
    *,
    call_index: int,
    role: CallRole,
) -> tuple[ResponseT | None, ModelCallRecord, ValidationIssue | None]:
    try:
        raw = port(request)
    except Exception:
        record = ModelCallRecord(
            call_index=call_index,
            role=role,
            status="exception",
            usage=None,
        )
        return None, record, _call_issue(role, "exception")
    try:
        response = _coerce_response(raw, expected)
    except (TypeError, ValidationError, ValueError):
        record = ModelCallRecord(
            call_index=call_index,
            role=role,
            status="invalid_contract",
            usage=None,
        )
        return None, record, _call_issue(role, "invalid_contract")
    record = ModelCallRecord(
        call_index=call_index,
        role=role,
        status="returned",
        usage=response.usage,
    )
    return response, record, None


@dataclass(frozen=True)
class _DeterministicAttempt:
    execution: ExecutionResult | None
    issues: tuple[ValidationIssue, ...]
    failed_stage: FailureStage | None
    compilation_attempts: int
    execution_attempts: int


def _compile_and_execute(
    program: AppletProgram,
    candidates: CandidateSet,
    *,
    context_fields: Sequence[EndpointField],
    context: Mapping[str, JsonValue] | None,
) -> _DeterministicAttempt:
    try:
        compiled = AppletCompiler(candidates, context_fields=context_fields).compile(program)
    except CompilationError as error:
        return _DeterministicAttempt(None, error.issues, "compilation", 1, 0)
    except Exception:
        issue = ValidationIssue(
            code="compiler_internal_error",
            location="compiler",
            message="deterministic compiler raised an unexpected error",
        )
        return _DeterministicAttempt(None, (issue,), "compilation", 1, 0)
    try:
        execution = SandboxConnectorRegistry(candidates).execute(compiled, context=context)
    except SandboxExecutionError as error:
        return _DeterministicAttempt(None, error.issues, "execution", 1, 1)
    except Exception:
        issue = ValidationIssue(
            code="executor_internal_error",
            location="executor",
            message="sandbox executor raised an unexpected error",
        )
        return _DeterministicAttempt(None, (issue,), "execution", 1, 1)
    return _DeterministicAttempt(execution, (), None, 1, 1)


def _resolve_baseline_pair(
    candidates: CandidateSet,
    *,
    baseline_trigger_id: str | None,
    baseline_action_id: str | None,
) -> tuple[str | None, str | None]:
    """Resolve the non-model baseline without assuming presentation rank.

    Omitted IDs preserve the convenient top-of-tuple behavior used by small
    synthetic tests.  Experiment runners must pass both explicit IDs so a
    rank-hidden/shuffled candidate presentation cannot change the fallback.
    """

    if baseline_trigger_id is None:
        trigger_id = candidates.triggers[0].candidate_id if candidates.triggers else None
    else:
        if candidates.resolve(baseline_trigger_id, "trigger") is None:
            raise ValueError(
                "baseline_trigger_id must identify a retrieved trigger candidate"
            )
        trigger_id = baseline_trigger_id

    if baseline_action_id is None:
        action_id = candidates.actions[0].candidate_id if candidates.actions else None
    else:
        if candidates.resolve(baseline_action_id, "action") is None:
            raise ValueError(
                "baseline_action_id must identify a retrieved action candidate"
            )
        action_id = baseline_action_id
    return trigger_id, action_id


def _fallback(
    baseline_trigger_id: str | None,
    baseline_action_id: str | None,
    reason_code: str,
) -> SafeFallback:
    return SafeFallback(
        trigger_candidate_id=baseline_trigger_id,
        action_candidate_id=baseline_action_id,
        reason_code=reason_code,
    )


def _prediction(result: AgentRunResult) -> dspy.Prediction:
    return dspy.Prediction(result=result)


class BoundedDSPyApplet(dspy.Module):
    """Direct planner with deterministic execution and a one-repair ceiling."""

    def __init__(
        self,
        planner: PlannerPort,
        *,
        repairer: RepairPort | None = None,
        max_repairs: Literal[0, 1] = 1,
    ) -> None:
        super().__init__()
        require_dspy_331()
        if max_repairs not in (0, 1):
            raise ValueError("max_repairs must be zero or one")
        _assert_explicit_port(planner, role="planner")
        if repairer is not None:
            _assert_explicit_port(repairer, role="repair")
        self._planner = planner
        self._repairer = repairer
        self._max_repairs = max_repairs
        self.runtime_policy = DSPyRuntimePolicy()

    def forward(
        self,
        *,
        case: Mapping[str, Any],
        candidates: CandidateSet,
        context_fields: Sequence[EndpointField] = (),
        context: Mapping[str, JsonValue] | None = None,
        baseline_trigger_id: str | None = None,
        baseline_action_id: str | None = None,
    ) -> dspy.Prediction:
        fallback_trigger_id, fallback_action_id = _resolve_baseline_pair(
            candidates,
            baseline_trigger_id=baseline_trigger_id,
            baseline_action_id=baseline_action_id,
        )
        envelope = build_inference_envelope(case, candidates)
        public_context = _public_context_fields(context_fields)
        planning_request = PlanningRequest(
            envelope=envelope,
            context_fields=public_context,
        )
        calls: list[ModelCallRecord] = []
        issues: list[ValidationIssue] = []
        attempted: list[AppletProgram] = []
        compilation_attempts = 0
        execution_attempts = 0

        response, call, call_issue = _invoke_port(
            self._planner,
            planning_request,
            PlannerResponse,
            call_index=1,
            role="planner",
        )
        calls.append(call)
        current_program = response.program if response is not None else None
        failed_stage: FailureStage = "planning"
        current_issues: tuple[ValidationIssue, ...]
        if call_issue is not None:
            current_issues = (call_issue,)
            issues.extend(current_issues)
        else:
            assert current_program is not None
            attempted.append(current_program)
            attempt = _compile_and_execute(
                current_program,
                candidates,
                context_fields=context_fields,
                context=context,
            )
            compilation_attempts += attempt.compilation_attempts
            execution_attempts += attempt.execution_attempts
            if attempt.execution is not None:
                result = AgentRunResult(
                    terminal_status="executed",
                    program=current_program,
                    execution=attempt.execution,
                    fallback=None,
                    attempted_programs=tuple(attempted),
                    issues=tuple(issues),
                    calls=tuple(calls),
                    call_totals=summarize_calls(calls),
                    compilation_attempts=compilation_attempts,
                    execution_attempts=execution_attempts,
                    repairs_attempted=0,
                    runtime_policy=self.runtime_policy,
                )
                return _prediction(result)
            current_issues = attempt.issues
            issues.extend(current_issues)
            failed_stage = attempt.failed_stage or "execution"

        can_repair = self._max_repairs == 1 and self._repairer is not None
        repairs_attempted = 0
        if can_repair:
            repairs_attempted = 1
            repair_request = RepairRequest(
                envelope=envelope,
                context_fields=public_context,
                failed_stage=failed_stage,
                failed_program=current_program,
                issues=current_issues,
            )
            repaired, repair_call, repair_issue = _invoke_port(
                self._repairer,
                repair_request,
                PlannerResponse,
                call_index=2,
                role="repair",
            )
            calls.append(repair_call)
            if repair_issue is not None:
                issues.append(repair_issue)
            else:
                assert repaired is not None
                attempted.append(repaired.program)
                attempt = _compile_and_execute(
                    repaired.program,
                    candidates,
                    context_fields=context_fields,
                    context=context,
                )
                compilation_attempts += attempt.compilation_attempts
                execution_attempts += attempt.execution_attempts
                if attempt.execution is not None:
                    result = AgentRunResult(
                        terminal_status="executed",
                        program=repaired.program,
                        execution=attempt.execution,
                        fallback=None,
                        attempted_programs=tuple(attempted),
                        issues=tuple(issues),
                        calls=tuple(calls),
                        call_totals=summarize_calls(calls),
                        compilation_attempts=compilation_attempts,
                        execution_attempts=execution_attempts,
                        repairs_attempted=1,
                        runtime_policy=self.runtime_policy,
                    )
                    return _prediction(result)
                issues.extend(attempt.issues)

        reason = issues[-1].code if issues else "planner_produced_no_program"
        result = AgentRunResult(
            terminal_status="safe_fallback",
            program=None,
            execution=None,
            fallback=_fallback(fallback_trigger_id, fallback_action_id, reason),
            attempted_programs=tuple(attempted),
            issues=tuple(issues),
            calls=tuple(calls),
            call_totals=summarize_calls(calls),
            compilation_attempts=compilation_attempts,
            execution_attempts=execution_attempts,
            repairs_attempted=repairs_attempted,
            runtime_policy=self.runtime_policy,
        )
        return _prediction(result)


def _autowire_consensus_program(
    candidates: CandidateSet,
    *,
    trigger_id: str,
    action_id: str,
    context_fields: Sequence[EndpointField],
) -> AppletProgram:
    """Bind a consensus trigger to the private baseline action deterministically."""

    trigger = candidates.resolve(trigger_id, "trigger")
    action = candidates.resolve(action_id, "action")
    if trigger is None or action is None:
        raise ValueError("consensus selection escaped retrieved candidates")
    outputs = {(field.path, field.value_type): field for field in trigger.outputs}
    available_context = {field.path for field in context_fields}
    bindings: list[FieldBinding] = []
    for target in action.inputs:
        if not target.required:
            continue
        matching_output = outputs.get((target.path, target.value_type))
        if matching_output is not None:
            source = TriggerOutputSource(
                kind="trigger_output", path=matching_output.path
            )
        else:
            context_path = f"config.{action_id}.{target.path}"
            if context_path not in available_context:
                raise ValueError("consensus autowire context contract missing")
            source = ContextSource(kind="context", path=context_path)
        bindings.append(FieldBinding(target_path=target.path, source=source))
    return AppletProgram(
        trigger=TriggerEndpointRef(candidate_id=trigger_id),
        action=ActionEndpointRef(candidate_id=action_id),
        bindings=tuple(bindings),
    )


class TriggerConsensusDSPyApplet(dspy.Module):
    """Sequential two-view trigger consensus with an executable safe fallback.

    This is a bounded DSPy control module, not a DSPy language-model program.
    Each injected selector is an explicit one-request native-tool adapter.  The
    second selector is skipped unless the first proposes a supported nonbaseline
    trigger that is present in the second view.  Model confidence is absent from
    the contracts and cannot influence acceptance.
    """

    ROUTING_THRESHOLD = 0.4

    def __init__(
        self,
        schema_m5_selector: TriggerConsensusPort,
        fused_m10_selector: TriggerConsensusPort,
        *,
        routing_threshold: float = ROUTING_THRESHOLD,
    ) -> None:
        super().__init__()
        require_dspy_331()
        if abs(float(routing_threshold) - self.ROUTING_THRESHOLD) > 1e-15:
            raise ValueError("trigger consensus routing threshold is frozen at 0.40")
        _assert_explicit_port(schema_m5_selector, role="trigger_schema_m5")
        _assert_explicit_port(fused_m10_selector, role="trigger_fused_m10")
        self._schema_m5_selector = schema_m5_selector
        self._fused_m10_selector = fused_m10_selector
        self.runtime_policy = DSPyRuntimePolicy()

    @staticmethod
    def _validate_view(
        candidates: CandidateSet,
        values: Sequence[str],
        *,
        expected: int,
        name: str,
    ) -> tuple[str, ...]:
        candidate_ids = tuple(values)
        if len(candidate_ids) != expected or len(set(candidate_ids)) != expected:
            raise ValueError(f"{name} must contain exactly {expected} unique triggers")
        if any(candidates.resolve(item, "trigger") is None for item in candidate_ids):
            raise ValueError(f"{name} escaped retrieved trigger candidates")
        return candidate_ids

    def forward(
        self,
        *,
        case: Mapping[str, Any],
        candidates: CandidateSet,
        schema_m5_candidate_ids: Sequence[str],
        fused_m10_candidate_ids: Sequence[str],
        trigger_seed_support: Mapping[str, int],
        routing_score: float,
        context_fields: Sequence[EndpointField] = (),
        context: Mapping[str, JsonValue] | None = None,
        baseline_trigger_id: str | None = None,
        baseline_action_id: str | None = None,
    ) -> dspy.Prediction:
        fallback_trigger_id, fallback_action_id = _resolve_baseline_pair(
            candidates,
            baseline_trigger_id=baseline_trigger_id,
            baseline_action_id=baseline_action_id,
        )
        if fallback_trigger_id is None or fallback_action_id is None:
            raise ValueError("trigger consensus requires a complete baseline pair")
        schema_ids = self._validate_view(
            candidates,
            schema_m5_candidate_ids,
            expected=5,
            name="schema_m5",
        )
        fused_ids = self._validate_view(
            candidates,
            fused_m10_candidate_ids,
            expected=10,
            name="fused_m10",
        )
        if not set(schema_ids).issubset(fused_ids):
            raise ValueError("schema_m5 must be a subset of fused_m10")
        score = float(routing_score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("routing_score must be finite and within [0, 1]")
        support: dict[str, int] = {}
        for candidate_id, raw in trigger_seed_support.items():
            if candidates.resolve(candidate_id, "trigger") is None:
                raise ValueError("seed-support map escaped retrieved triggers")
            if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 3:
                raise ValueError("seed support must be an integer within [0, 3]")
            support[candidate_id] = raw

        request_id = case.get("request_id")
        query = case.get("query")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("consensus case requires request_id")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("consensus case requires query")

        calls: list[ModelCallRecord] = []
        issues: list[ValidationIssue] = []
        first_candidate: str | None = None
        second_candidate: str | None = None
        selected_support: int | None = None
        attempted: tuple[AppletProgram, ...] = ()
        compilation_attempts = 0
        execution_attempts = 0

        def finish_fallback(reason: str, *, consensus: bool = False) -> dspy.Prediction:
            trace = TriggerConsensusTrace(
                routing_score=score,
                routed=score >= self.ROUTING_THRESHOLD,
                schema_m5_called=len(calls) >= 1,
                fused_m10_called=len(calls) >= 2,
                schema_m5_candidate_id=first_candidate,
                fused_m10_candidate_id=second_candidate,
                selected_trigger_id=first_candidate if consensus else None,
                selected_trigger_seed_support=selected_support,
                exact_trigger_consensus=consensus,
                decision_reason=reason,
            )
            return _prediction(
                AgentRunResult(
                    terminal_status="safe_fallback",
                    program=None,
                    execution=None,
                    fallback=_fallback(
                        fallback_trigger_id, fallback_action_id, reason
                    ),
                    attempted_programs=attempted,
                    issues=tuple(issues),
                    calls=tuple(calls),
                    call_totals=summarize_calls(calls),
                    compilation_attempts=compilation_attempts,
                    execution_attempts=execution_attempts,
                    repairs_attempted=0,
                    runtime_policy=self.runtime_policy,
                    trigger_consensus_trace=trace,
                )
            )

        if score < self.ROUTING_THRESHOLD:
            return finish_fallback("router_below_0_40")

        first_request = TriggerConsensusRequest(
            request_id=request_id,
            query=query.strip(),
            view="schema_m5",
            candidate_ids=schema_ids,
        )
        first, first_call, first_issue = _invoke_port(
            self._schema_m5_selector,
            first_request,
            TriggerConsensusResponse,
            call_index=1,
            role="trigger_schema_m5",
        )
        calls.append(first_call)
        if first_issue is not None:
            issues.append(first_issue)
            return finish_fallback("schema_m5_call_failed")
        assert first is not None
        if first.deferred:
            return finish_fallback("schema_m5_deferred")
        first_candidate = first.candidate_id
        if first_candidate not in schema_ids:
            issues.append(
                ValidationIssue(
                    code="schema_m5_candidate_invalid",
                    location="trigger_schema_m5",
                    message="first view selected outside its dynamic candidate set",
                )
            )
            return finish_fallback("schema_m5_candidate_invalid")
        if first_candidate == fallback_trigger_id:
            return finish_fallback("schema_m5_retained_baseline")
        selected_support = support.get(first_candidate, 0)
        if selected_support < 2:
            return finish_fallback("trigger_seed_support_below_2")
        if first_candidate not in fused_ids:
            return finish_fallback("trigger_absent_from_fused_m10")

        second_request = TriggerConsensusRequest(
            request_id=request_id,
            query=query.strip(),
            view="fused_m10",
            candidate_ids=fused_ids,
        )
        second, second_call, second_issue = _invoke_port(
            self._fused_m10_selector,
            second_request,
            TriggerConsensusResponse,
            call_index=2,
            role="trigger_fused_m10",
        )
        calls.append(second_call)
        if second_issue is not None:
            issues.append(second_issue)
            return finish_fallback("fused_m10_call_failed")
        assert second is not None
        if second.deferred:
            return finish_fallback("fused_m10_deferred")
        second_candidate = second.candidate_id
        if second_candidate not in fused_ids:
            issues.append(
                ValidationIssue(
                    code="fused_m10_candidate_invalid",
                    location="trigger_fused_m10",
                    message="second view selected outside its dynamic candidate set",
                )
            )
            return finish_fallback("fused_m10_candidate_invalid")
        if second_candidate != first_candidate:
            return finish_fallback("trigger_consensus_disagreement")

        try:
            program = _autowire_consensus_program(
                candidates,
                trigger_id=first_candidate,
                action_id=fallback_action_id,
                context_fields=context_fields,
            )
        except Exception:
            issues.append(
                ValidationIssue(
                    code="consensus_autowire_failed",
                    location="deterministic_binder",
                    message="deterministic consensus binder rejected the selection",
                )
            )
            return finish_fallback("consensus_autowire_failed", consensus=True)
        attempted = (program,)
        attempt = _compile_and_execute(
            program,
            candidates,
            context_fields=context_fields,
            context=context,
        )
        compilation_attempts = attempt.compilation_attempts
        execution_attempts = attempt.execution_attempts
        if attempt.execution is None:
            issues.extend(attempt.issues)
            return finish_fallback(
                attempt.issues[-1].code if attempt.issues else "consensus_execution_failed",
                consensus=True,
            )

        trace = TriggerConsensusTrace(
            routing_score=score,
            routed=True,
            schema_m5_called=True,
            fused_m10_called=True,
            schema_m5_candidate_id=first_candidate,
            fused_m10_candidate_id=second_candidate,
            selected_trigger_id=first_candidate,
            selected_trigger_seed_support=selected_support,
            exact_trigger_consensus=True,
            decision_reason="consensus_compiled_and_executed",
        )
        return _prediction(
            AgentRunResult(
                terminal_status="executed",
                program=program,
                execution=attempt.execution,
                fallback=None,
                attempted_programs=attempted,
                issues=tuple(issues),
                calls=tuple(calls),
                call_totals=summarize_calls(calls),
                compilation_attempts=compilation_attempts,
                execution_attempts=execution_attempts,
                repairs_attempted=0,
                runtime_policy=self.runtime_policy,
                trigger_consensus_trace=trace,
            )
        )


class FactorizedDSPyApplet(dspy.Module):
    """Two independent endpoint specialists followed by one binder/critic."""

    def __init__(
        self,
        trigger_specialist: SpecialistPort,
        action_specialist: SpecialistPort,
        binder_critic: BinderCriticPort,
        *,
        parallel_specialists: bool = True,
    ) -> None:
        super().__init__()
        require_dspy_331()
        _assert_explicit_port(trigger_specialist, role="trigger_specialist")
        _assert_explicit_port(action_specialist, role="action_specialist")
        _assert_explicit_port(binder_critic, role="binder_critic")
        self._trigger_specialist = trigger_specialist
        self._action_specialist = action_specialist
        self._binder_critic = binder_critic
        self._parallel_specialists = parallel_specialists
        self.runtime_policy = DSPyRuntimePolicy()

    def _specialists(
        self,
        envelope: InferenceEnvelope,
    ) -> tuple[
        tuple[SpecialistResponse | None, ModelCallRecord, ValidationIssue | None],
        tuple[SpecialistResponse | None, ModelCallRecord, ValidationIssue | None],
    ]:
        trigger_request = EndpointSpecialistRequest(
            request_id=envelope.request_id,
            query=envelope.query,
            side="trigger",
            candidates=envelope.trigger_candidates,
        )
        action_request = EndpointSpecialistRequest(
            request_id=envelope.request_id,
            query=envelope.query,
            side="action",
            candidates=envelope.action_candidates,
        )

        def trigger_call() -> tuple[SpecialistResponse | None, ModelCallRecord, ValidationIssue | None]:
            return _invoke_port(
                self._trigger_specialist,
                trigger_request,
                SpecialistResponse,
                call_index=1,
                role="trigger_specialist",
            )

        def action_call() -> tuple[SpecialistResponse | None, ModelCallRecord, ValidationIssue | None]:
            return _invoke_port(
                self._action_specialist,
                action_request,
                SpecialistResponse,
                call_index=2,
                role="action_specialist",
            )

        if not self._parallel_specialists:
            return trigger_call(), action_call()
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="farm-specialist") as pool:
            trigger_future = pool.submit(trigger_call)
            action_future = pool.submit(action_call)
            # Retrieval is intentionally ordered even if completion is not, so
            # call indices and experiment JSON remain deterministic.
            return trigger_future.result(), action_future.result()

    def forward(
        self,
        *,
        case: Mapping[str, Any],
        candidates: CandidateSet,
        context_fields: Sequence[EndpointField] = (),
        context: Mapping[str, JsonValue] | None = None,
        baseline_trigger_id: str | None = None,
        baseline_action_id: str | None = None,
    ) -> dspy.Prediction:
        fallback_trigger_id, fallback_action_id = _resolve_baseline_pair(
            candidates,
            baseline_trigger_id=baseline_trigger_id,
            baseline_action_id=baseline_action_id,
        )
        envelope = build_inference_envelope(case, candidates)
        public_context = _public_context_fields(context_fields)
        trigger_result, action_result = self._specialists(envelope)
        trigger_response, trigger_call, trigger_issue = trigger_result
        action_response, action_call, action_issue = action_result
        calls = [trigger_call, action_call]
        issues = [item for item in (trigger_issue, action_issue) if item is not None]

        trigger_ids = {item.candidate_id for item in envelope.trigger_candidates}
        action_ids = {item.candidate_id for item in envelope.action_candidates}
        if trigger_response is not None:
            proposal = trigger_response.proposal
            if proposal.side != "trigger" or proposal.candidate_id not in trigger_ids:
                issues.append(
                    ValidationIssue(
                        code="invalid_trigger_proposal",
                        location="trigger_specialist.proposal",
                        message="trigger specialist selected outside its retrieved side",
                    )
                )
        if action_response is not None:
            proposal = action_response.proposal
            if proposal.side != "action" or proposal.candidate_id not in action_ids:
                issues.append(
                    ValidationIssue(
                        code="invalid_action_proposal",
                        location="action_specialist.proposal",
                        message="action specialist selected outside its retrieved side",
                    )
                )

        attempted: tuple[AppletProgram, ...] = ()
        compilation_attempts = 0
        execution_attempts = 0
        if not issues and trigger_response is not None and action_response is not None:
            binder_request = BinderCriticRequest(
                envelope=envelope,
                context_fields=public_context,
                trigger=trigger_response.proposal,
                action=action_response.proposal,
            )
            binder, binder_call, binder_issue = _invoke_port(
                self._binder_critic,
                binder_request,
                BinderCriticResponse,
                call_index=3,
                role="binder_critic",
            )
            calls.append(binder_call)
            if binder_issue is not None:
                issues.append(binder_issue)
            elif binder is not None and not binder.approved:
                issues.append(
                    ValidationIssue(
                        code=f"binder_rejected:{binder.rejection_code}",
                        location="binder_critic",
                        message="binder/critic rejected the proposed applet",
                    )
                )
            elif binder is not None:
                assert binder.program is not None
                program = binder.program
                attempted = (program,)
                if (
                    program.trigger.candidate_id != trigger_response.proposal.candidate_id
                    or program.action.candidate_id != action_response.proposal.candidate_id
                ):
                    issues.append(
                        ValidationIssue(
                            code="binder_changed_specialist_selection",
                            location="binder_critic.program",
                            message="binder must bind and critique the independently selected endpoints",
                        )
                    )
                else:
                    attempt = _compile_and_execute(
                        program,
                        candidates,
                        context_fields=context_fields,
                        context=context,
                    )
                    compilation_attempts = attempt.compilation_attempts
                    execution_attempts = attempt.execution_attempts
                    if attempt.execution is not None:
                        result = AgentRunResult(
                            terminal_status="executed",
                            program=program,
                            execution=attempt.execution,
                            fallback=None,
                            attempted_programs=attempted,
                            issues=tuple(issues),
                            calls=tuple(calls),
                            call_totals=summarize_calls(calls),
                            compilation_attempts=compilation_attempts,
                            execution_attempts=execution_attempts,
                            repairs_attempted=0,
                            runtime_policy=self.runtime_policy,
                        )
                        return _prediction(result)
                    issues.extend(attempt.issues)

        reason = issues[-1].code if issues else "specialist_produced_no_proposal"
        result = AgentRunResult(
            terminal_status="safe_fallback",
            program=None,
            execution=None,
            fallback=_fallback(fallback_trigger_id, fallback_action_id, reason),
            attempted_programs=attempted,
            issues=tuple(issues),
            calls=tuple(calls),
            call_totals=summarize_calls(calls),
            compilation_attempts=compilation_attempts,
            execution_attempts=execution_attempts,
            repairs_attempted=0,
            runtime_policy=self.runtime_policy,
        )
        return _prediction(result)


__all__ = [
    "AgentRunResult",
    "BinderCriticRequest",
    "BinderCriticResponse",
    "BoundedDSPyApplet",
    "CallTotals",
    "DSPY_REQUIRED_VERSION",
    "DSPyRuntimePolicy",
    "EndpointProposal",
    "EndpointSpecialistRequest",
    "FactorizedDSPyApplet",
    "ModelCallRecord",
    "PlannerResponse",
    "PlanningRequest",
    "PortUsage",
    "PublicContextField",
    "RepairRequest",
    "SafeFallback",
    "SpecialistResponse",
    "TriggerConsensusDSPyApplet",
    "TriggerConsensusRequest",
    "TriggerConsensusResponse",
    "TriggerConsensusTrace",
    "require_dspy_331",
    "summarize_calls",
]

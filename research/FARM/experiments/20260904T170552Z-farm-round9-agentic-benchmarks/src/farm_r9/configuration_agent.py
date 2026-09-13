"""Bounded, fail-closed applet-configuration agent.

The endpoint pair is fixed before this program starts.  The program may cite
only exact spans from the accumulated user dialogue, actual trigger
ingredients, supplied resource observations, and opaque secret references.
The compiler establishes structural validity and grounding; it does not imply
semantic value accuracy when a benchmark has no value-level gold.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import TypeAdapter, ValidationError

from farm_r9.agent_state import AgentState, fresh_state
from farm_r9.artifact_io import assert_no_secrets
from farm_r9.compiler import AppletCompiler
from farm_r9.contracts import (
    AgentAction,
    AgentAsk,
    AgentCommit,
    AppletDraft,
    CompilationResult,
    EndpointCandidate,
    FieldSpec,
    IngredientSpec,
    ResourceObservation,
    SecretReference,
    ValidationIssue,
)
from farm_r9.ollama_client import OllamaChatClient, OllamaJSONError, parse_json_object
from farm_r9.privacy import DataClassification, DataSource


_SYSTEM_PROMPT_PREFIX = """You configure an already selected trigger-action applet for a nontechnical user.
The endpoint aliases are fixed: never replace them. Inspect every selected schema field.
"""

_SYSTEM_PROMPT_SUFFIX = """The draft must follow the supplied AppletDraft schema. Give every required and optional field exactly one decision.
Allowed sources only: query_literal (an exact [start,end) span of grounding_text), trigger_output (an actual selected-trigger ingredient, action fields only), resource_ref (an ID in a supplied observation), secret_ref (an available opaque alias, auth fields only), needs_input, or omit (optional fields only).
For every commit, draft.evidence_aliases must contain exactly the fixed trigger alias and fixed action alias, in that order. Prefer a compatible, clearly relevant trigger_output over asking the user to re-enter data that the trigger already supplies.
Never invent a value, resource, ingredient, secret, alias, or schema field. If evidence is absent, use needs_input; omit an optional field when appropriate. Do not expose credentials. Keep the preview nontechnical and evidence-grounded. Output JSON only."""


def system_prompt_for_arm(arm: str, *, max_questions: int) -> str:
    """Return the frozen shared prompt plus the arm's feasible action contract."""

    if arm == "single_shot_configurator":
        action_contract = """Return exactly one JSON object:
{"action":"commit","draft":{...}}
Asking is unavailable in this one-shot arm. Commit now. For every unresolved required field use needs_input. For each optional field use omit when appropriate or needs_input when user input is necessary."""
    elif arm == "bounded_configuration_agent":
        if not isinstance(max_questions, int) or isinstance(max_questions, bool) or max_questions < 0:
            raise ValueError("max_questions must be a nonnegative integer")
        question_noun = "question" if max_questions == 1 else "questions"
        action_contract = f"""Return exactly one JSON object representing either:
1. {{"action":"ask","component":"trigger:<field_slug> or action:<field_slug>","question":"one compact question"}}
2. {{"action":"commit","draft":{{...}}}}
You may ask at most {max_questions} clarification {question_noun} before committing."""
    else:
        raise ValueError("unsupported configuration arm")
    return _SYSTEM_PROMPT_PREFIX + action_contract + "\n" + _SYSTEM_PROMPT_SUFFIX


_ACTION_ADAPTER = TypeAdapter(AgentAction)
_FABRICATION_CODES = {
    "ungrounded_query_literal",
    "unknown_ingredient",
    "unobserved_resource",
    "unavailable_secret_ref",
    "unknown_field",
    "unsupported_source",
}
_SECURITY_CODES = {"secret_material", "secret_in_non_auth_field"}
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?:bearer\s+\S+|(?:api[_-]?key|password|passwd|access[_-]?token|authorization)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConfigurationEvidence:
    clarification_answers: Mapping[str, str]
    resource_observations: tuple[ResourceObservation, ...] = ()
    available_secret_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigurationAgentResult:
    draft: AppletDraft | None
    compilation: CompilationResult | None
    calls: tuple[dict[str, Any], ...]
    questions: tuple[dict[str, Any], ...]
    validation_errors: tuple[dict[str, Any], ...]
    protocol_failure: str | None
    semantic_accuracy_supported: bool = False

    def metrics(self) -> dict[str, Any]:
        result = self.compilation
        decisions = () if self.draft is None else (*self.draft.trigger_fields, *self.draft.action_fields)
        unresolved_disclosures = sum(
            1 for item in decisions if item.source.kind == "needs_input"
        )
        explicit_omissions = sum(1 for item in decisions if item.source.kind == "omit")
        attempted_codes = [str(item.get("code")) for item in self.validation_errors]
        final_codes = [] if result is None else [issue.code for issue in result.issues]
        return {
            "protocol_success": self.protocol_failure is None,
            "compiler_valid": bool(result and result.valid),
            "structurally_complete": bool(result and result.structurally_complete),
            "executable_ready_under_supplied_evidence": bool(result and result.executable_ready),
            "question_count": len(self.questions),
            "answered_question_count": sum(bool(item["answer_available"]) for item in self.questions),
            "needs_input_decision_count": unresolved_disclosures,
            "explicit_omit_decision_count": explicit_omissions,
            "fabrication_attempt_count": sum(code in _FABRICATION_CODES for code in attempted_codes),
            "accepted_fabrication_count": sum(code in _FABRICATION_CODES for code in final_codes),
            "security_violation_attempt_count": sum(code in _SECURITY_CODES for code in attempted_codes),
            "accepted_security_violation_count": sum(code in _SECURITY_CODES for code in final_codes),
            "semantic_value_accuracy": None,
            "semantic_accuracy_supported": False,
        }


def _field_specs(candidate: Mapping[str, Any]) -> tuple[FieldSpec, ...]:
    if isinstance(candidate.get("fields"), list):
        return tuple(FieldSpec.model_validate(item, strict=True) for item in candidate["fields"])
    # RecipeGen exposes field names but not requiredness, value type, or
    # binding metadata. Conservative required=True prevents silent omission;
    # downstream reports must retain requiredness_known=False.
    return tuple(
        FieldSpec(slug=str(name), label=str(name), required=True, bindable=False)
        for name in (candidate.get("field_names") or [])
    )


def endpoint_candidate(candidate: Mapping[str, Any]) -> EndpointCandidate:
    ingredients = tuple(
        IngredientSpec.model_validate(item, strict=True)
        for item in (candidate.get("ingredients") or [])
    )
    return EndpointCandidate(
        alias=str(candidate["alias"]), side=str(candidate["side"]),
        service=str(candidate["service"]),
        service_id=None if candidate.get("service_id") is None else str(candidate["service_id"]),
        function=str(candidate["function"]), description=str(candidate.get("description") or ""),
        fields=_field_specs(candidate), ingredients=ingredients,
    )


def evidence_from_case(case: Mapping[str, Any]) -> ConfigurationEvidence:
    raw = case.get("configuration_evidence") or {}
    answers = raw.get("clarification_answers") or {}
    if not isinstance(answers, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in answers.items()):
        raise ValueError("clarification_answers must map component strings to supplied strings")
    observations = tuple(ResourceObservation.model_validate(item, strict=True) for item in (raw.get("resource_observations") or []))
    secret_refs = tuple(str(item) for item in (raw.get("available_secret_refs") or []))
    # Clarification text and public mock resources are still outbound prompt
    # material. Reject credential-shaped content before it reaches any model.
    assert_no_secrets(dict(answers))
    if any(_CREDENTIAL_ASSIGNMENT.search(value) for value in answers.values()):
        raise ValueError("credential-shaped clarification answer is forbidden")
    assert_no_secrets([item.model_dump(mode="json") for item in observations])
    for reference in secret_refs:
        SecretReference(kind="secret_ref", reference=reference)
        assert_no_secrets(reference)
    return ConfigurationEvidence(dict(answers), observations, secret_refs)


def _selected(case: Mapping[str, Any], trigger_alias: str, action_alias: str) -> tuple[EndpointCandidate, EndpointCandidate]:
    public = case["public_evidence"]
    triggers = {item["alias"]: item for item in public["trigger_candidates"]}
    actions = {item["alias"]: item for item in public["action_candidates"]}
    if trigger_alias not in triggers or action_alias not in actions:
        raise ValueError("selected alias is not in candidate evidence")
    trigger = endpoint_candidate(triggers[trigger_alias])
    action = endpoint_candidate(actions[action_alias])
    return trigger, action


def _grounding_text(query: str, observations: Sequence[Mapping[str, str]]) -> str:
    parts = [query]
    parts.extend(f"\nclarification[{item['component']}]: {item['answer']}" for item in observations)
    return "".join(parts)


def _request_payload(
    *, query: str, grounding_text: str, trigger: EndpointCandidate,
    action: EndpointCandidate, evidence: ConfigurationEvidence,
) -> dict[str, Any]:
    return {
        "initial_request": query,
        "grounding_text": grounding_text,
        "fixed_selection": {"trigger_alias": trigger.alias, "action_alias": action.alias},
        "selected_trigger": trigger.model_dump(mode="json"),
        "selected_action": action.model_dump(mode="json"),
        "resource_observations": [item.model_dump(mode="json") for item in evidence.resource_observations],
        "available_secret_refs": list(evidence.available_secret_refs),
        "applet_draft_schema": AppletDraft.model_json_schema(),
    }


def _parse_action(content: str) -> tuple[AgentAction | None, list[dict[str, Any]], bool]:
    try:
        value, strict_json = parse_json_object(content)
    except OllamaJSONError:
        return None, [{"code": "json_parse_failure"}], False
    try:
        return _ACTION_ADAPTER.validate_python(value, strict=True), [], strict_json
    except ValidationError as error:
        return None, [
            {"code": "pydantic_validation", "location": [str(part) for part in item["loc"]], "type": item["type"]}
            for item in error.errors(include_input=False, include_url=False)
        ], strict_json


def _valid_component(component: str, trigger: EndpointCandidate, action: EndpointCandidate) -> bool:
    parts = component.split(":", 1)
    if len(parts) != 2 or parts[0] not in {"trigger", "action"}:
        return False
    endpoint = trigger if parts[0] == "trigger" else action
    return parts[1] in {field.slug for field in endpoint.fields}


def _fixed_selection_validation(
    draft: AppletDraft, trigger: EndpointCandidate, action: EndpointCandidate,
) -> CompilationResult | None:
    issues: list[ValidationIssue] = []
    if draft.selection.trigger_alias != trigger.alias:
        issues.append(ValidationIssue(code="changed_fixed_trigger", side="selection", message="trigger endpoint is fixed before configuration"))
    if draft.selection.action_alias != action.alias:
        issues.append(ValidationIssue(code="changed_fixed_action", side="selection", message="action endpoint is fixed before configuration"))
    expected_evidence = {trigger.alias, action.alias}
    if set(draft.evidence_aliases) != expected_evidence:
        issues.append(ValidationIssue(code="invalid_endpoint_citations", side="selection", message="evidence aliases must cite exactly the fixed endpoints"))
    if not issues:
        return None
    return CompilationResult(
        valid=False, structurally_complete=False, executable_ready=False,
        issues=tuple(issues), trigger_values=(), action_values=(),
    )


def run_configuration_agent(
    *, client: OllamaChatClient, case: Mapping[str, Any], trigger_alias: str,
    action_alias: str, benchmark: str, classification: DataClassification,
    data_source: DataSource, arm: str, max_questions: int = 2,
    think: str | bool | None = None,
) -> ConfigurationAgentResult:
    """Run a one-shot or bounded agent without correctness-aware retries."""
    if arm not in {"single_shot_configurator", "bounded_configuration_agent"}:
        raise ValueError("unsupported configuration arm")
    trigger, action = _selected(case, trigger_alias, action_alias)
    evidence = evidence_from_case(case)
    state = fresh_state(str(case["case_id"]), max_questions=0 if arm == "single_shot_configurator" else max_questions, max_repairs=0 if arm == "single_shot_configurator" else 1)
    observations: list[dict[str, str]] = []
    grounding = _grounding_text(str(case["input"]["query"]), observations)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": system_prompt_for_arm(arm, max_questions=state.max_questions),
        },
        {"role": "user", "content": json.dumps(_request_payload(query=str(case["input"]["query"]), grounding_text=grounding, trigger=trigger, action=action, evidence=evidence), ensure_ascii=False, sort_keys=True)},
    ]
    calls: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    # initial + each allowed clarification + one validation repair
    maximum_calls = 1 + state.max_questions + state.max_repairs
    for ordinal in range(1, maximum_calls + 1):
        state.begin_call()
        response = client.chat(
            semantic_id=f"{benchmark}/{case['case_id']}/{arm}/call-{ordinal}",
            benchmark_label=benchmark, data_classification=classification,
            data_source=data_source, messages=messages, temperature=0.0,
            seed=42,
            think=("low" if bool(getattr(client, "cloud", False)) else None)
            if think is None
            else think,
        )
        call = {
            "ordinal": ordinal, "request_sha256": response.request_sha256,
            "response_sha256": response.response_sha256, "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens, "provider_latency_ms": response.provider_latency_ms,
            "queue_wait_ms": response.queue_wait_ms, "physical_attempts": response.physical_attempts,
            "cache_hit": response.cache_hit,
        }
        calls.append(call)
        action_value, parse_errors, strict_json = _parse_action(response.content)
        call["strict_json"] = strict_json
        if action_value is None:
            errors.extend({"call": ordinal, **item} for item in parse_errors)
            return ConfigurationAgentResult(None, None, tuple(calls), tuple(questions), tuple(errors), parse_errors[0]["code"])
        if isinstance(action_value, AgentAsk):
            if arm == "single_shot_configurator":
                return ConfigurationAgentResult(None, None, tuple(calls), tuple(questions), tuple(errors), "question_forbidden_in_single_shot")
            if not _valid_component(action_value.component, trigger, action):
                errors.append({"call": ordinal, "code": "unknown_question_component"})
                return ConfigurationAgentResult(None, None, tuple(calls), tuple(questions), tuple(errors), "unknown_question_component")
            try:
                state.ask(action_value.component)
            except ValueError as error:
                errors.append({"call": ordinal, "code": "question_budget_or_repeat"})
                return ConfigurationAgentResult(None, None, tuple(calls), tuple(questions), tuple(errors), str(error))
            supplied = evidence.clarification_answers.get(action_value.component)
            answer_available = supplied is not None
            answer = supplied if supplied is not None else "No supplied answer is available. Keep this field unresolved with needs_input."
            state.observe(component=action_value.component, answer=answer, source="supplied_simulator" if answer_available else "evidence_absent")
            questions.append({
                "component": action_value.component, "question": action_value.question,
                "answer_available": answer_available,
            })
            # Only a supplied user/simulator answer becomes grounding text.
            # The deterministic absence message is an instruction, not evidence.
            if answer_available:
                observations.append({"component": action_value.component, "answer": answer})
            grounding = _grounding_text(str(case["input"]["query"]), observations)
            messages.extend([
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": json.dumps({
                    "clarification_component": action_value.component,
                    "answer": answer,
                    "answer_available": answer_available,
                    "grounding_text": grounding,
                    "instruction": "Continue with the fixed endpoints. Cite only exact spans in this updated grounding_text.",
                }, ensure_ascii=False, sort_keys=True)},
            ])
            continue

        assert isinstance(action_value, AgentCommit)
        draft = action_value.draft
        compiled = _fixed_selection_validation(draft, trigger, action)
        if compiled is None:
            compiler = AppletCompiler(
                query=grounding, trigger_candidates=(trigger,), action_candidates=(action,),
                resource_observations=evidence.resource_observations,
                available_secret_refs=evidence.available_secret_refs,
            )
            compiled = compiler.compile(draft)
        state.record_validation(draft=draft, result=compiled)
        if compiled.valid:
            return ConfigurationAgentResult(draft, compiled, tuple(calls), tuple(questions), tuple(errors), None)
        validation = [issue.model_dump(mode="json") for issue in compiled.issues]
        errors.extend({"call": ordinal, **item} for item in validation)
        if state.max_repairs == 0 or state.repair_count >= state.max_repairs or not any(issue.repairable for issue in compiled.issues):
            return ConfigurationAgentResult(draft, compiled, tuple(calls), tuple(questions), tuple(errors), "compiler_validation_failed")
        repairable = state.begin_repair()
        messages.extend([
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": json.dumps({
                "actual_compiler_errors": repairable,
                "instruction": "Repair only these compiler errors. Keep fixed endpoints and return one full action JSON object.",
            }, sort_keys=True)},
        ])
    return ConfigurationAgentResult(None, None, tuple(calls), tuple(questions), tuple(errors), "semantic_call_budget_exhausted")

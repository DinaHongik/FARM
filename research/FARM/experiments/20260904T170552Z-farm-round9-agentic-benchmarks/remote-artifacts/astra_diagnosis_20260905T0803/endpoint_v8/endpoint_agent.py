"""Strong one-shot and bounded validation-repair endpoint programs."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from farm_r9.contracts import EndpointPrediction
from farm_r9.ollama_client import OllamaChatClient, OllamaJSONError, parse_json_object
from farm_r9.privacy import DataClassification, DataSource


SYSTEM_PROMPT = """You compile one trigger-action program from a user request.
TRIGGER means the event/condition after IF/WHEN. ACTION means the operation after THEN.
Choose exactly one supplied trigger alias and one supplied action alias. Never invent an alias.
For each chosen endpoint, copy every documented configuration field label exactly and in schema order; use [] when none.
Write a short nontechnical preview supported only by the request and candidate evidence.
Return one JSON object only with keys: trigger_alias, action_alias, trigger_field_names, action_field_names, preview, evidence_aliases.
evidence_aliases must contain the chosen trigger and action aliases. Do not output credentials or hidden reasoning."""


@dataclass(frozen=True)
class AgentResult:
    prediction: EndpointPrediction | None
    calls: tuple[dict[str, Any], ...]
    validation_errors: tuple[dict[str, Any], ...]
    repaired: bool
    protocol_failure: str | None


def _field_labels(candidate: Mapping[str, Any]) -> list[str]:
    if isinstance(candidate.get("field_names"), list):
        return [str(value) for value in candidate["field_names"]]
    fields = candidate.get("fields") or []
    return [str(field.get("label") or field.get("slug")) for field in fields if isinstance(field, Mapping)]


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    allowed = ("alias", "side", "service", "service_id", "function", "function_event", "description", "field_names", "fields", "ingredients")
    return {key: candidate[key] for key in allowed if key in candidate}


def make_request(case: Mapping[str, Any]) -> dict[str, Any]:
    evidence = case["public_evidence"]
    return {
        "request": case["input"]["query"],
        "trigger_candidates": [_public_candidate(item) for item in evidence["trigger_candidates"]],
        "action_candidates": [_public_candidate(item) for item in evidence["action_candidates"]],
    }


def validate_prediction(value: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[EndpointPrediction | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    try:
        prediction = EndpointPrediction.model_validate(value, strict=True)
    except ValidationError as validation:
        return None, [
            {"code": "pydantic_validation", "location": [str(part) for part in item["loc"]], "type": item["type"]}
            for item in validation.errors(include_input=False, include_url=False)
        ]
    evidence = case["public_evidence"]
    triggers = {row["alias"]: row for row in evidence["trigger_candidates"]}
    actions = {row["alias"]: row for row in evidence["action_candidates"]}
    trigger, action = triggers.get(prediction.trigger_alias), actions.get(prediction.action_alias)
    if trigger is None:
        errors.append({"code": "unknown_trigger_alias"})
    if action is None:
        errors.append({"code": "unknown_action_alias"})
    if trigger is not None and list(prediction.trigger_field_names) != _field_labels(trigger):
        errors.append({"code": "trigger_field_names_not_exact_schema"})
    if action is not None and list(prediction.action_field_names) != _field_labels(action):
        errors.append({"code": "action_field_names_not_exact_schema"})
    allowed_aliases = set(triggers) | set(actions)
    if set(prediction.evidence_aliases) - allowed_aliases:
        errors.append({"code": "unknown_evidence_alias"})
    if trigger is not None and prediction.trigger_alias not in prediction.evidence_aliases:
        errors.append({"code": "missing_trigger_citation"})
    if action is not None and prediction.action_alias not in prediction.evidence_aliases:
        errors.append({"code": "missing_action_citation"})
    return (prediction if not errors else None), errors


def run_endpoint_program(
    *,
    client: OllamaChatClient,
    case: Mapping[str, Any],
    benchmark: str,
    classification: DataClassification,
    data_source: DataSource,
    arm: str,
) -> AgentResult:
    if arm not in {"same_model_one_shot", "bounded_tool_agent", "dspy_compiled_agent"}:
        raise ValueError("unsupported endpoint arm")
    public_request = make_request(case)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(public_request, ensure_ascii=False, sort_keys=True)},
    ]
    calls: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    maximum_calls = 1 if arm == "same_model_one_shot" else 2
    for ordinal in range(1, maximum_calls + 1):
        result = client.chat(
            semantic_id=f"{benchmark}/{case['case_id']}/{arm}/call-{ordinal}",
            benchmark_label=benchmark, data_classification=classification,
            data_source=data_source,
            messages=messages, temperature=0.0, seed=42, think="low",
        )
        call_record = {
            "ordinal": ordinal, "request_sha256": result.request_sha256,
            "response_sha256": result.response_sha256, "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens, "provider_latency_ms": result.provider_latency_ms,
            "queue_wait_ms": result.queue_wait_ms, "physical_attempts": result.physical_attempts,
            "cache_hit": result.cache_hit,
        }
        calls.append(call_record)
        try:
            value, strict_json = parse_json_object(result.content)
            prediction, errors = validate_prediction(value, case)
            call_record["strict_json"] = strict_json
        except OllamaJSONError:
            prediction, errors = None, [{"code": "json_parse_failure"}]
            call_record["strict_json"] = False
        if prediction is not None:
            return AgentResult(prediction, tuple(calls), tuple(all_errors), ordinal > 1, None)
        all_errors.extend({"call": ordinal, **error} for error in errors)
        if ordinal == maximum_calls:
            return AgentResult(None, tuple(calls), tuple(all_errors), ordinal > 1, errors[0]["code"] if errors else "invalid_output")
        # A repair call receives only actual validator feedback and the same
        # public evidence. It receives no gold, rank, or correctness signal.
        messages.extend([
            {"role": "assistant", "content": result.content},
            {"role": "user", "content": json.dumps({
                "validation_failed": errors,
                "instruction": "Repair only these validation failures using the original candidate evidence. Return the full JSON object.",
            }, sort_keys=True)},
        ])
    raise AssertionError("unreachable")

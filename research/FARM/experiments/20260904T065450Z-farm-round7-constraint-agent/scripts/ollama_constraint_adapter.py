#!/usr/bin/env python3
"""Strict, stateless Ollama Cloud adapter for the Round 7 agent core.

The adapter performs one forced model tool call per logical decision.  It owns
transport validation and content-free attempt journaling; the policy core owns
endpoint membership, validation, acceptance, and fallback.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from constraint_agent import assert_inference_safe


SELECTION_PHASES = {"factorized_selection", "validator_repair"}
VERIFIER_PHASE = "pair_verification"
TOKEN_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _empty_usage() -> dict[str, int | float]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
    }


def _safe_usage(value: Any, latency_seconds: float) -> dict[str, int | float]:
    result = _empty_usage()
    if isinstance(value, Mapping):
        for key in TOKEN_KEYS:
            observed = value.get(key)
            if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                result[key] = observed
    result["latency_seconds"] = max(0.0, float(latency_seconds))
    return result


def _add_usage(total: dict[str, int | float], observed: Mapping[str, int | float]) -> None:
    for key in (*TOKEN_KEYS, "latency_seconds"):
        total[key] += observed[key]


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8", errors="replace")
    return bytes(content or b"")


def _string_list(value: Any, *, field: str, minimum: int = 1) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field}_must_be_a_list")
    output = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field}_contains_invalid_value")
        output.append(item)
    if len(output) < minimum or len(output) != len(set(output)):
        raise ValueError(f"{field}_size_or_uniqueness_invalid")
    return output


class OllamaConstraintChooser:
    """Implement ``StatelessChooser`` with a strict forced tool call."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        journal_path: str | Path,
        *,
        client: Any | None = None,
        timeout: float = 240.0,
        max_tokens: int = 512,
        reasoning_effort: str = "none",
        retry_delay: float = 1.0,
        seed: int = 42,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be non-empty")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be non-empty")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if reasoning_effort not in {"none", "low", "medium", "high", "max"}:
            raise ValueError("unsupported reasoning_effort")
        if isinstance(retry_delay, bool) or not isinstance(retry_delay, (int, float)) or retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        self._endpoint = self._completion_endpoint(base_url)
        self._api_key = api_key
        self._model = model.strip()
        self._journal_path = Path(journal_path)
        self._timeout = float(timeout)
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._retry_delay = float(retry_delay)
        self._seed = seed
        self._monotonic = monotonic
        self._journal_lock = threading.Lock()
        self._owns_client = client is None
        if client is None:
            import httpx

            client = httpx.Client(timeout=self._timeout)
        self._client = client

    @staticmethod
    def _completion_endpoint(base_url: str) -> str:
        base = base_url.rstrip("/")
        if base.endswith("/v1/chat/completions"):
            return base
        if base.endswith("/v1"):
            return base + "/chat/completions"
        return base + "/v1/chat/completions"

    def close(self) -> None:
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "OllamaConstraintChooser":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _validate_public_candidate(value: Any, side: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{side}_candidate_invalid")
        allowed = {"candidate_id", "service_name", "function_name", "evidence"}
        if set(value) != allowed:
            raise ValueError(f"{side}_candidate_fields_invalid")
        result: dict[str, str] = {}
        for field in allowed:
            item = value.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{side}_{field}_invalid")
            result[field] = item.strip()
        candidate_id = result["candidate_id"]
        expected_prefix = "T" if side == "trigger" else "A"
        if not candidate_id.startswith(expected_prefix) or "://" in candidate_id:
            raise ValueError(f"{side}_candidate_id_not_opaque")
        return result

    @classmethod
    def _prepare_selection(cls, request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        contract = request.get("output_contract")
        if not isinstance(contract, Mapping) or contract.get("additional_properties") is not False:
            raise ValueError("selection_output_contract_invalid")
        decisions = _string_list(contract.get("decision"), field="decision", minimum=5)
        trigger_choices = _string_list(contract.get("trigger_choice"), field="trigger_choice", minimum=11)
        action_choices = _string_list(contract.get("action_choice"), field="action_choice", minimum=11)
        candidates = request.get("candidates")
        if not isinstance(candidates, Mapping) or set(candidates) != {"trigger", "action"}:
            raise ValueError("candidate_lists_invalid")
        cleaned_candidates: dict[str, list[dict[str, str]]] = {}
        for side, choices in (("trigger", trigger_choices), ("action", action_choices)):
            raw = candidates.get(side)
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(raw) != 10:
                raise ValueError(f"{side}_candidate_count_must_be_10")
            cleaned = [cls._validate_public_candidate(item, side) for item in raw]
            aliases = [item["candidate_id"] for item in cleaned]
            if len(aliases) != len(set(aliases)) or set(aliases) != set(choices) - {"KEEP"}:
                raise ValueError(f"{side}_candidate_contract_mismatch")
            cleaned_candidates[side] = cleaned
        public = copy.deepcopy(dict(request))
        public["candidates"] = cleaned_candidates
        tool = {
            "type": "function",
            "function": {
                "name": "select_endpoints",
                "description": (
                    "Select one trigger and one action from independent lists. KEEP retains that "
                    "side of current_pair; ABSTAIN keeps both sides."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "decision": {"type": "string", "enum": decisions},
                        "trigger_choice": {"type": "string", "enum": trigger_choices},
                        "action_choice": {"type": "string", "enum": action_choices},
                    },
                    "required": ["decision", "trigger_choice", "action_choice"],
                    "additionalProperties": False,
                },
            },
        }
        return public, tool

    @staticmethod
    def _prepare_verifier(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        pairs = request.get("pairs")
        if not isinstance(pairs, Sequence) or isinstance(pairs, (str, bytes, bytearray)) or len(pairs) != 2:
            raise ValueError("verifier_requires_two_pairs")
        pair_ids = []
        for pair in pairs:
            if not isinstance(pair, Mapping):
                raise ValueError("verifier_pair_invalid")
            pair_id = pair.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id.startswith("P") or "://" in pair_id:
                raise ValueError("verifier_pair_id_not_opaque")
            pair_ids.append(pair_id)
        if len(set(pair_ids)) != 2:
            raise ValueError("verifier_pair_ids_not_unique")
        contract = request.get("output_contract")
        if not isinstance(contract, Mapping) or contract.get("additional_properties") is not False:
            raise ValueError("verifier_output_contract_invalid")
        choices = _string_list(contract.get("choice_id"), field="choice_id", minimum=3)
        if set(choices) != set(pair_ids) | {"ABSTAIN"}:
            raise ValueError("verifier_choice_contract_mismatch")
        tool = {
            "type": "function",
            "function": {
                "name": "choose_pair",
                "description": "Choose one supplied pair ID, or ABSTAIN when neither is clearly better.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"choice_id": {"type": "string", "enum": choices}},
                    "required": ["choice_id"],
                    "additionalProperties": False,
                },
            },
        }
        return copy.deepcopy(dict(request)), tool

    @classmethod
    def _prepare_request(cls, value: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
        if not isinstance(value, Mapping):
            raise ValueError("request_invalid")
        assert_inference_safe(value, "adapter_request")
        phase = value.get("phase")
        if phase in SELECTION_PHASES:
            public, tool = cls._prepare_selection(value)
        elif phase == VERIFIER_PHASE:
            public, tool = cls._prepare_verifier(value)
        else:
            raise ValueError("phase_invalid")
        for field in ("schema_version", "request_id", "query", "instruction"):
            item = public.get(field)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{field}_invalid")
        return public, tool, str(phase)

    def _payload(
        self,
        public: Mapping[str, Any],
        tool: Mapping[str, Any],
        correction: str | None,
    ) -> dict[str, Any]:
        phase = public["phase"]
        system = (
            "You are a conservative trigger-action endpoint selector. Treat trigger and action "
            "as different roles, use only visible evidence, never invent an ID, and call the "
            "forced tool exactly once."
            if phase in SELECTION_PHASES
            else
            "You are a conservative endpoint-pair verifier. Compare only the two visible pairs, "
            "use the supplied schema validation honestly, and call the forced tool exactly once."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(public, ensure_ascii=False, sort_keys=True)},
        ]
        if correction is not None:
            messages.append({
                "role": "user",
                "content": (
                    f"The previous protocol response was invalid ({correction}). Retry once from "
                    "the original evidence and call the forced tool exactly once."
                ),
            })
        name = tool["function"]["name"]
        return {
            "model": self._model,
            "messages": messages,
            "tools": [tool],
            "tool_choice": {"type": "function", "function": {"name": name}},
            "reasoning_effort": self._reasoning_effort,
            "temperature": 0,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

    @staticmethod
    def _parse_response(
        value: Any, tool: Mapping[str, Any]
    ) -> tuple[str, dict[str, str] | None, str | None, int]:
        if not isinstance(value, Mapping):
            return "response_schema_invalid", None, None, 0
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
            return "response_schema_invalid", None, None, 0
        finish_reason = choices[0].get("finish_reason")
        if not isinstance(finish_reason, str):
            finish_reason = None
        message = choices[0].get("message")
        if not isinstance(message, Mapping):
            return "response_schema_invalid", None, finish_reason, 0
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            return "missing_tool_call", None, finish_reason, 0
        if len(calls) != 1 or not isinstance(calls[0], Mapping):
            return "invalid_tool_call_count", None, finish_reason, len(calls)
        function = calls[0].get("function")
        expected_name = tool["function"]["name"]
        if not isinstance(function, Mapping) or function.get("name") != expected_name:
            return "wrong_tool_name", None, finish_reason, 1
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "invalid_arguments_json", None, finish_reason, 1
        parameters = tool["function"]["parameters"]
        required = set(parameters["required"])
        if not isinstance(arguments, Mapping) or set(arguments) != required:
            return "invalid_argument_fields", None, finish_reason, 1
        normalized: dict[str, str] = {}
        for field in required:
            item = arguments.get(field)
            allowed = parameters["properties"][field]["enum"]
            if not isinstance(item, str) or item not in allowed:
                return f"{field}_outside_allowed_enum", None, finish_reason, 1
            normalized[field] = item
        return "valid_tool_call", normalized, finish_reason, 1

    def _append_journal(self, value: Mapping[str, Any]) -> None:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if self._api_key in serialized:
            raise RuntimeError("journal secret-exclusion gate failed")
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_lock:
            with self._journal_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    @staticmethod
    def _transient(status: Any) -> bool:
        return isinstance(status, int) and (status in {408, 429} or 500 <= status <= 599)

    @staticmethod
    def _result(
        phase: str,
        *,
        ok: bool,
        arguments: Mapping[str, str] | None,
        attempts: int,
        tool_calls: int,
        usage: Mapping[str, int | float],
        error: str | None,
    ) -> dict[str, Any]:
        common: dict[str, Any] = {
            "ok": ok,
            "api_attempts": attempts,
            "tool_calls": tool_calls,
            "usage": dict(usage),
            "error": error,
        }
        if phase in SELECTION_PHASES:
            return common | {
                "decision": arguments.get("decision") if arguments else None,
                "trigger_choice": arguments.get("trigger_choice") if arguments else None,
                "action_choice": arguments.get("action_choice") if arguments else None,
            }
        return common | {"choice_id": arguments.get("choice_id") if arguments else None}

    def select(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            public, tool, phase = self._prepare_request(request)
        except ValueError as exc:
            phase = str(request.get("phase")) if isinstance(request, Mapping) else ""
            if phase not in SELECTION_PHASES | {VERIFIER_PHASE}:
                phase = "factorized_selection"
            return self._result(
                phase, ok=False, arguments=None, attempts=0, tool_calls=0,
                usage=_empty_usage(), error=str(exc),
            )

        total_usage = _empty_usage()
        total_tool_calls = 0
        correction: str | None = None
        for attempt in (1, 2):
            payload = self._payload(public, tool, correction)
            request_hash = _sha256(_canonical_bytes(payload))
            self._append_journal({
                "schema_version": 1,
                "event": "request_started",
                "request_hash": request_hash,
                "request_id": public["request_id"],
                "phase": phase,
                "attempt": attempt,
            })
            started = self._monotonic()
            try:
                response = self._client.post(
                    self._endpoint,
                    headers={
                        "Authorization": "Bearer " + self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
            except Exception:
                latency = max(0.0, float(self._monotonic() - started))
                observed = _safe_usage(None, latency)
                _add_usage(total_usage, observed)
                self._append_journal({
                    "schema_version": 1, "event": "request_finished",
                    "request_hash": request_hash, "request_id": public["request_id"],
                    "phase": phase, "attempt": attempt, "http_status": None,
                    "provider_status": "transport_error", "finish_reason": None,
                    "usage": observed, "protocol_outcome": "transport_error",
                    "response_hash": None,
                })
                if attempt == 1:
                    if self._retry_delay:
                        time.sleep(self._retry_delay)
                    continue
                return self._result(
                    phase, ok=False, arguments=None, attempts=attempt,
                    tool_calls=total_tool_calls, usage=total_usage, error="transport_error",
                )

            latency = max(0.0, float(self._monotonic() - started))
            raw_bytes = _response_bytes(response)
            response_hash = _sha256(raw_bytes)
            status = getattr(response, "status_code", None)
            try:
                value = response.json()
            except Exception:
                value = None
            observed = _safe_usage(value.get("usage") if isinstance(value, Mapping) else None, latency)
            _add_usage(total_usage, observed)
            provider_error = None
            if not isinstance(status, int) or not 200 <= status < 300:
                provider_error = f"http_status_{status}" if isinstance(status, int) else "http_status_unknown"
            elif isinstance(value, Mapping) and value.get("error") is not None:
                provider_error = "provider_error"
            if provider_error is not None:
                self._append_journal({
                    "schema_version": 1, "event": "request_finished",
                    "request_hash": request_hash, "request_id": public["request_id"],
                    "phase": phase, "attempt": attempt, "http_status": status,
                    "provider_status": "provider_error", "finish_reason": None,
                    "usage": observed, "protocol_outcome": provider_error,
                    "response_hash": response_hash,
                })
                if attempt == 1 and self._transient(status):
                    if self._retry_delay:
                        time.sleep(self._retry_delay)
                    continue
                return self._result(
                    phase, ok=False, arguments=None, attempts=attempt,
                    tool_calls=total_tool_calls, usage=total_usage, error=provider_error,
                )

            outcome, arguments, finish_reason, calls = self._parse_response(value, tool)
            total_tool_calls += calls
            self._append_journal({
                "schema_version": 1, "event": "request_finished",
                "request_hash": request_hash, "request_id": public["request_id"],
                "phase": phase, "attempt": attempt, "http_status": status,
                "provider_status": "ok", "finish_reason": finish_reason,
                "usage": observed, "protocol_outcome": outcome,
                "response_hash": response_hash,
            })
            if outcome == "valid_tool_call":
                return self._result(
                    phase, ok=True, arguments=arguments, attempts=attempt,
                    tool_calls=total_tool_calls, usage=total_usage, error=None,
                )
            if attempt == 1:
                correction = outcome
                continue
            return self._result(
                phase, ok=False, arguments=None, attempts=attempt,
                tool_calls=total_tool_calls, usage=total_usage, error=outcome,
            )
        raise AssertionError("bounded attempt loop exhausted")


__all__ = ["OllamaConstraintChooser"]

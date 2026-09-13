#!/usr/bin/env python3
"""Audited Ollama OpenAI-compatible selector for FARM round four.

The adapter is deliberately a narrow external boundary: callers provide an
already public candidate view, and receive either a validated opaque-ID pair or
an explicit failure.  It never substitutes a baseline decision on failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable


_PHASES = {"function", "service", "function_within_service"}
_TOKEN_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")
_USAGE_KEYS = (*_TOKEN_USAGE_KEYS, "latency_seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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


def _safe_usage(value: Any) -> dict[str, int | float]:
    result = _empty_usage()
    if not isinstance(value, dict):
        return result
    for key in _TOKEN_USAGE_KEYS:
        observed = value.get(key)
        if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
            result[key] = observed
    return result


def _add_usage(
    total: dict[str, int | float],
    observed: dict[str, int | float],
) -> None:
    for key in _USAGE_KEYS:
        total[key] += observed[key]


def _response_bytes(response: Any) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, bytes):
        return content
    if isinstance(content, str):
        return content.encode("utf-8", errors="replace")
    return bytes(content or b"")


class OllamaSelector:
    """Implement the resolver's ``Selector.select(request)`` port.

    ``client`` is injectable at the external HTTP seam.  Production use lazily
    creates an ``httpx.Client`` so importing pure resolver code does not require
    the HTTP dependency.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        journal_path: str | Path,
        *,
        client: Any | None = None,
        timeout: float = 120.0,
        max_tokens: int = 384,
        reasoning_effort: str = "low",
        retry_delay: float = 0.25,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if reasoning_effort not in {"none", "low", "medium", "high", "max"}:
            raise ValueError("reasoning_effort is unsupported")
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, (int, float))
            or retry_delay < 0
        ):
            raise ValueError("retry_delay must be non-negative")
        if not callable(monotonic):
            raise ValueError("monotonic must be callable")

        self._endpoint = self._completion_endpoint(base_url)
        self._api_key = api_key
        self._model = model
        self._journal_path = Path(journal_path)
        self._timeout = float(timeout)
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._retry_delay = float(retry_delay)
        self._monotonic = monotonic
        self._journal_lock = threading.Lock()
        self._owns_client = client is None
        if client is None:
            import httpx  # Imported only at the external production boundary.

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

    def __enter__(self) -> "OllamaSelector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _candidate_id(candidate: dict[str, Any]) -> str:
        value = candidate.get("candidate_id", candidate.get("id"))
        if not isinstance(value, str) or not value:
            raise ValueError("candidate_missing_opaque_id")
        if "://" in value or value.startswith(("/", "\\")):
            raise ValueError("candidate_id_must_be_opaque")
        return value

    @classmethod
    def _public_candidate(cls, candidate: Any) -> dict[str, str]:
        if not isinstance(candidate, dict):
            raise ValueError("candidate_must_be_an_object")
        candidate_id = cls._candidate_id(candidate)

        service = next(
            (
                value
                for key in ("service_name", "service", "service_slug", "channel_display")
                if isinstance((value := candidate.get(key)), str) and value
            ),
            None,
        )
        display_parts: list[str] = []
        for key in ("display", "text", "function_name", "evidence"):
            value = candidate.get(key)
            if isinstance(value, str) and value and value not in display_parts:
                display_parts.append(value)
        summaries = candidate.get("function_summaries")
        if isinstance(summaries, (list, tuple)):
            for summary in summaries:
                if not isinstance(summary, dict):
                    continue
                parts = [
                    summary.get(key)
                    for key in ("function_name", "evidence")
                    if isinstance(summary.get(key), str) and summary.get(key)
                ]
                if parts:
                    display_parts.append(" — ".join(parts))
        if not display_parts and service:
            display_parts.append(service)
        if not display_parts:
            raise ValueError("candidate_missing_display_evidence")

        public = {"candidate_id": candidate_id, "display_text": "\n".join(display_parts)}
        if service:
            public["service_name"] = service
        return public

    @classmethod
    def _prepare_request(
        cls,
        request: Any,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        if not isinstance(request, dict):
            raise ValueError("request_must_be_an_object")
        query = request.get("query")
        phase = request.get("phase")
        instruction = request.get("instruction")
        if not isinstance(query, str) or not query:
            raise ValueError("request_query_invalid")
        if phase not in _PHASES:
            raise ValueError("request_phase_invalid")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("request_instruction_invalid")

        scope = request.get("evidence_scope")
        if not isinstance(scope, dict):
            raise ValueError("evidence_scope_invalid")
        top_k = scope.get("top_k")
        cumulative = scope.get("cumulative")
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
            or not isinstance(cumulative, bool)
        ):
            raise ValueError("evidence_scope_invalid")

        public: dict[str, Any] = {
            "query": query,
            "phase": phase,
            "instruction": instruction,
            "evidence_scope": {"top_k": top_k, "cumulative": cumulative},
        }
        allowed: list[list[str]] = []
        for side in ("trigger", "action"):
            candidates = request.get(f"{side}_candidates")
            if not isinstance(candidates, (list, tuple)) or not candidates:
                raise ValueError(f"{side}_candidates_invalid")
            normalized = [cls._public_candidate(candidate) for candidate in candidates]
            ids = [candidate["candidate_id"] for candidate in normalized]
            if len(ids) != len(set(ids)):
                raise ValueError(f"{side}_candidate_ids_not_unique")
            public[f"{side}_candidates"] = normalized
            allowed.append(ids)

        selected = request.get("selected_services")
        if selected is not None:
            if (
                not isinstance(selected, dict)
                or set(selected) != {"trigger_id", "action_id"}
                or not all(isinstance(selected[key], str) and selected[key] for key in selected)
            ):
                raise ValueError("selected_services_invalid")
            public["selected_services"] = {
                "trigger_id": selected["trigger_id"],
                "action_id": selected["action_id"],
            }
        return public, allowed[0], allowed[1]

    @staticmethod
    def _tool(trigger_ids: list[str], action_ids: list[str]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "submit_pair",
                "description": "Submit exactly one supplied trigger/action candidate pair.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {"type": "string", "enum": trigger_ids},
                        "action_id": {"type": "string", "enum": action_ids},
                    },
                    "required": ["trigger_id", "action_id"],
                    "additionalProperties": False,
                },
            },
        }

    def _payload(
        self,
        public_request: dict[str, Any],
        trigger_ids: list[str],
        action_ids: list[str],
        correction: str | None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Select only from the supplied opaque candidate IDs. Judge the trigger and "
                    "action jointly from the visible evidence. Call submit_pair exactly once."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(public_request, ensure_ascii=False, sort_keys=True),
            },
        ]
        if correction is not None:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid ({correction}). Retry once and call "
                        "submit_pair exactly once with exactly two keys. "
                        f"Allowed trigger IDs: {json.dumps(trigger_ids)}. "
                        f"Allowed action IDs: {json.dumps(action_ids)}."
                    ),
                }
            )
        return {
            "model": self._model,
            "messages": messages,
            "tools": [self._tool(trigger_ids, action_ids)],
            "tool_choice": {"type": "function", "function": {"name": "submit_pair"}},
            "reasoning_effort": self._reasoning_effort,
            "temperature": 0,
            "seed": 42,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

    @staticmethod
    def _parse_response(
        value: Any,
        trigger_ids: set[str],
        action_ids: set[str],
    ) -> tuple[str, str | None, str | None, str | None, int]:
        if not isinstance(value, dict):
            return "response_schema_invalid", None, None, None, 0
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return "response_schema_invalid", None, None, None, 0
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str):
            finish_reason = None
        message = choice.get("message")
        if not isinstance(message, dict):
            return "response_schema_invalid", None, None, finish_reason, 0
        calls = message.get("tool_calls")
        if calls is None or calls == []:
            return "missing_submit_pair", None, None, finish_reason, 0
        if not isinstance(calls, list):
            return "tool_calls_schema_invalid", None, None, finish_reason, 0
        tool_count = len(calls)
        if tool_count != 1 or not isinstance(calls[0], dict):
            return "invalid_tool_call_count", None, None, finish_reason, tool_count
        function = calls[0].get("function")
        if not isinstance(function, dict) or function.get("name") != "submit_pair":
            return "wrong_tool_name", None, None, finish_reason, tool_count
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "invalid_arguments_json", None, None, finish_reason, tool_count
        if not isinstance(arguments, dict):
            return "invalid_arguments_type", None, None, finish_reason, tool_count
        if set(arguments) != {"trigger_id", "action_id"}:
            return "invalid_argument_keys", None, None, finish_reason, tool_count
        trigger_id = arguments["trigger_id"]
        action_id = arguments["action_id"]
        if not isinstance(trigger_id, str) or not isinstance(action_id, str):
            return "invalid_argument_types", None, None, finish_reason, tool_count
        if trigger_id not in trigger_ids or action_id not in action_ids:
            return "candidate_id_outside_allowed_enum", None, None, finish_reason, tool_count
        return "valid_submit_pair", trigger_id, action_id, finish_reason, tool_count

    def _append_journal(self, record: dict[str, Any]) -> None:
        serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
        if self._api_key in serialized:
            raise RuntimeError("journal secret-exclusion gate failed")
        self._journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self._journal_lock:
            with self._journal_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _wait_before_retry(self) -> None:
        if self._retry_delay:
            time.sleep(self._retry_delay)

    @staticmethod
    def _transient_http_status(status: Any) -> bool:
        return isinstance(status, int) and (
            status in {408, 429} or 500 <= status <= 599
        )

    @staticmethod
    def _failure(
        *,
        api_attempts: int,
        tool_calls: int,
        usage: dict[str, int | float],
        error: str,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "trigger_id": None,
            "action_id": None,
            "api_attempts": api_attempts,
            "tool_calls": tool_calls,
            "usage": dict(usage),
            "error": error,
        }

    def select(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return one validated selection, or an explicit non-decision."""
        try:
            public_request, trigger_ids, action_ids = self._prepare_request(request)
        except ValueError as exc:
            return self._failure(
                api_attempts=0,
                tool_calls=0,
                usage=_empty_usage(),
                error=str(exc),
            )

        total_usage = _empty_usage()
        total_tool_calls = 0
        correction: str | None = None
        for attempt in (1, 2):
            payload = self._payload(public_request, trigger_ids, action_ids, correction)
            request_hash = _sha256(_canonical_bytes(payload))
            attempt_started = self._monotonic()
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
            except Exception:  # The injected/HTTP client is the external boundary.
                latency_seconds = max(0.0, float(self._monotonic() - attempt_started))
                observed_usage = _empty_usage()
                observed_usage["latency_seconds"] = latency_seconds
                _add_usage(total_usage, observed_usage)
                error = "transport_error"
                self._append_journal(
                    {
                        "schema_version": 1,
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "http_status": None,
                        "provider_status": error,
                        "finish_reason": None,
                        "usage": observed_usage,
                        "latency_seconds": latency_seconds,
                        "protocol_outcome": "transport_error",
                        "response_hash": None,
                    }
                )
                if attempt == 1:
                    self._wait_before_retry()
                    continue
                return self._failure(
                    api_attempts=attempt,
                    tool_calls=total_tool_calls,
                    usage=total_usage,
                    error=error,
                )

            latency_seconds = max(0.0, float(self._monotonic() - attempt_started))
            response_raw = _response_bytes(response)
            response_hash = _sha256(response_raw)
            status = getattr(response, "status_code", None)
            try:
                value = response.json()
            except Exception:
                value = None
            observed_usage = _safe_usage(value.get("usage") if isinstance(value, dict) else None)
            observed_usage["latency_seconds"] = latency_seconds
            _add_usage(total_usage, observed_usage)

            provider_status = "ok"
            provider_error = None
            if not isinstance(status, int) or not 200 <= status < 300:
                provider_status = "http_error"
                provider_error = f"http_status_{status}" if isinstance(status, int) else "http_status_unknown"
            elif isinstance(value, dict) and value.get("error") is not None:
                provider_status = "provider_error"
                provider_error = "provider_error"

            if provider_error is not None:
                self._append_journal(
                    {
                        "schema_version": 1,
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "http_status": status,
                        "provider_status": provider_status,
                        "finish_reason": None,
                        "usage": observed_usage,
                        "latency_seconds": latency_seconds,
                        "protocol_outcome": "provider_error",
                        "response_hash": response_hash,
                    }
                )
                if attempt == 1 and self._transient_http_status(status):
                    self._wait_before_retry()
                    continue
                return self._failure(
                    api_attempts=attempt,
                    tool_calls=total_tool_calls,
                    usage=total_usage,
                    error=provider_error,
                )

            outcome, trigger_id, action_id, finish_reason, observed_calls = self._parse_response(
                value,
                set(trigger_ids),
                set(action_ids),
            )
            total_tool_calls += observed_calls
            self._append_journal(
                {
                    "schema_version": 1,
                    "request_hash": request_hash,
                    "attempt": attempt,
                    "http_status": status,
                    "provider_status": provider_status,
                    "finish_reason": finish_reason,
                    "usage": observed_usage,
                    "latency_seconds": latency_seconds,
                    "protocol_outcome": outcome,
                    "response_hash": response_hash,
                }
            )
            if outcome == "valid_submit_pair":
                return {
                    "ok": True,
                    "trigger_id": trigger_id,
                    "action_id": action_id,
                    "api_attempts": attempt,
                    "tool_calls": total_tool_calls,
                    "usage": dict(total_usage),
                    "error": None,
                }
            if attempt == 1:
                correction = outcome
                continue
            return self._failure(
                api_attempts=attempt,
                tool_calls=total_tool_calls,
                usage=total_usage,
                error=outcome,
            )

        raise AssertionError("bounded attempt loop exhausted")


__all__ = ["OllamaSelector"]

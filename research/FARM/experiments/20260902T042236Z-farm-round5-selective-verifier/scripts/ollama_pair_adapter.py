#!/usr/bin/env python3
"""Strict Ollama boundary for the round-five pair-card experiment.

The policy owns fallback and acceptance.  This adapter only exposes blinded
pair cards to the model, validates one ``choose_card`` tool call, and records a
content-free write-ahead audit entry for every HTTP attempt.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PHASES = {"pair_proposal", "pair_verification"}
EVIDENCE_VIEWS = {"plain", "schema"}
REFERENCE_KEYS = {
    "valid_pairs",
    "gold_pairs",
    "gold",
    "gold_pair",
    "gold_trigger_urls",
    "gold_action_urls",
    "gold_channel_pairs",
    "gold_service_pairs",
    "ground_truth",
    "reference",
    "references",
    "reference_answer",
    "expected_pair",
    "observed_pairs",
    "is_correct",
    "label",
    "pair_rank",
}
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
    usage = _empty_usage()
    if isinstance(value, Mapping):
        for key in TOKEN_KEYS:
            observed = value.get(key)
            if isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0:
                usage[key] = observed
    usage["latency_seconds"] = max(0.0, float(latency_seconds))
    return usage


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


def _assert_no_references(value: Any, *, path: str = "request") -> None:
    """Reject evaluation labels at the outermost inference boundary."""
    if isinstance(value, Mapping):
        normalized = {str(key).lower() for key in value}
        leaked = {
            key
            for key in normalized
            if key in REFERENCE_KEYS
            or key.startswith("gold_")
            or key.startswith("reference_")
            or key.startswith("ground_truth_")
        }
        if leaked:
            raise ValueError(f"reference_fields_forbidden_at_{path}: {sorted(leaked)}")
        for key, item in value.items():
            _assert_no_references(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_no_references(item, path=f"{path}[{index}]")


class OllamaPairChooser:
    """Implement the pair policy's ``CardChooser.select`` protocol."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        journal_path: str | Path,
        *,
        client: Any | None = None,
        timeout: float = 240.0,
        max_tokens: int = 384,
        reasoning_effort: str = "low",
        retry_delay: float = 0.25,
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
        if (
            isinstance(retry_delay, bool)
            or not isinstance(retry_delay, (int, float))
            or retry_delay < 0
        ):
            raise ValueError("retry_delay must be non-negative")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
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

    def __enter__(self) -> "OllamaPairChooser":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @staticmethod
    def _opaque_card_id(value: Any) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("card_id_invalid")
        if value == "ABSTAIN" or "://" in value or value.startswith(("/", "\\")):
            raise ValueError("card_id_must_be_opaque")
        return value

    @staticmethod
    def _public_side(value: Any, side: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{side}_card_side_invalid")
        allowed = {"service_name", "function_name", "evidence"}
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"{side}_card_side_has_private_fields: {sorted(unexpected)}")
        result: dict[str, str] = {}
        for key in ("service_name", "function_name", "evidence"):
            item = value.get(key)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{side}_{key}_invalid")
            result[key] = item.strip()
        return result

    @classmethod
    def _public_card(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("card_invalid")
        if set(value) != {"card_id", "trigger", "action"}:
            raise ValueError("card_fields_invalid")
        return {
            "card_id": cls._opaque_card_id(value["card_id"]),
            "trigger": cls._public_side(value["trigger"], "trigger"),
            "action": cls._public_side(value["action"], "action"),
        }

    @classmethod
    def _prepare_request(cls, request: Any) -> tuple[dict[str, Any], list[str], bool]:
        if not isinstance(request, Mapping):
            raise ValueError("request_invalid")
        _assert_no_references(request)
        phase = request.get("phase")
        query = request.get("query")
        instruction = request.get("instruction")
        evidence_view = request.get("evidence_view")
        allow_abstain = request.get("allow_abstain")
        if phase not in PHASES:
            raise ValueError("phase_invalid")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query_invalid")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction_invalid")
        if evidence_view not in EVIDENCE_VIEWS:
            raise ValueError("evidence_view_invalid")
        if not isinstance(allow_abstain, bool):
            raise ValueError("allow_abstain_invalid")
        if (phase == "pair_proposal" and allow_abstain) or (
            phase == "pair_verification" and not allow_abstain
        ):
            raise ValueError("allow_abstain_phase_mismatch")
        cards = request.get("cards")
        if not isinstance(cards, Sequence) or isinstance(cards, (str, bytes)):
            raise ValueError("cards_invalid")
        if not 2 <= len(cards) <= 10:
            raise ValueError("card_count_outside_2_to_10")
        public_cards = [cls._public_card(card) for card in cards]
        card_ids = [card["card_id"] for card in public_cards]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("card_ids_not_unique")

        # case_id/group_id intentionally stays local; ordering, source scores,
        # baseline roles and gold labels are never part of this payload.
        public = {
            "query": query.strip(),
            "phase": phase,
            "instruction": instruction.strip(),
            "evidence_view": evidence_view,
            "cards": public_cards,
            "allow_abstain": allow_abstain,
        }
        return public, card_ids, allow_abstain

    @staticmethod
    def _tool(card_ids: list[str], allow_abstain: bool) -> dict[str, Any]:
        choices = list(card_ids)
        if allow_abstain:
            choices.append("ABSTAIN")
        return {
            "type": "function",
            "function": {
                "name": "choose_card",
                "description": (
                    "Choose exactly one supplied card ID. Use ABSTAIN only when the visible "
                    "evidence cannot justify a card."
                ),
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"choice_id": {"type": "string", "enum": choices}},
                    "required": ["choice_id"],
                    "additionalProperties": False,
                },
            },
        }

    def _payload(
        self,
        public: dict[str, Any],
        card_ids: list[str],
        allow_abstain: bool,
        correction: str | None,
    ) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a conservative semantic pair selector. Treat every card as an "
                    "atomic trigger-action pair, use only visible evidence, infer no hidden "
                    "ranking, and call choose_card exactly once."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(public, ensure_ascii=False, sort_keys=True),
            },
        ]
        if correction is not None:
            allowed = card_ids + (["ABSTAIN"] if allow_abstain else [])
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The response was invalid ({correction}). Retry once. Call choose_card "
                        f"exactly once with one choice_id from: {json.dumps(allowed)}."
                    ),
                }
            )
        return {
            "model": self._model,
            "messages": messages,
            "tools": [self._tool(card_ids, allow_abstain)],
            "tool_choice": {"type": "function", "function": {"name": "choose_card"}},
            "reasoning_effort": self._reasoning_effort,
            "temperature": 0,
            "seed": self._seed,
            "max_tokens": self._max_tokens,
            "stream": False,
        }

    @staticmethod
    def _parse_response(value: Any, allowed: set[str]) -> tuple[str, str | None, str | None, int]:
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
            return "missing_choose_card", None, finish_reason, 0
        if len(calls) != 1 or not isinstance(calls[0], Mapping):
            return "invalid_tool_call_count", None, finish_reason, len(calls)
        function = calls[0].get("function")
        if not isinstance(function, Mapping) or function.get("name") != "choose_card":
            return "wrong_tool_name", None, finish_reason, 1
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return "invalid_arguments_json", None, finish_reason, 1
        if not isinstance(arguments, Mapping) or set(arguments) != {"choice_id"}:
            return "invalid_argument_fields", None, finish_reason, 1
        choice_id = arguments.get("choice_id")
        if not isinstance(choice_id, str) or choice_id not in allowed:
            return "choice_outside_allowed_enum", None, finish_reason, 1
        return "valid_choose_card", choice_id, finish_reason, 1

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
    def _failure(
        attempts: int,
        tool_calls: int,
        usage: Mapping[str, int | float],
        error: str,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "choice_id": None,
            "api_attempts": attempts,
            "tool_calls": tool_calls,
            "usage": dict(usage),
            "error": error,
        }

    @staticmethod
    def _transient(status: Any) -> bool:
        return isinstance(status, int) and (status in {408, 429} or 500 <= status <= 599)

    def select(self, request: Mapping[str, Any]) -> dict[str, Any]:
        try:
            public, card_ids, allow_abstain = self._prepare_request(request)
        except ValueError as exc:
            return self._failure(0, 0, _empty_usage(), str(exc))

        total_usage = _empty_usage()
        total_tool_calls = 0
        correction: str | None = None
        allowed = set(card_ids) | ({"ABSTAIN"} if allow_abstain else set())
        for attempt in (1, 2):
            payload = self._payload(public, card_ids, allow_abstain, correction)
            request_hash = _sha256(_canonical_bytes(payload))
            # Persist intent before the external side effect.  If the process
            # dies in-flight, the unmatched request_started row makes the
            # ambiguous call visible instead of silently losing it.
            self._append_journal(
                {
                    "schema_version": 1,
                    "event": "request_started",
                    "request_hash": request_hash,
                    "attempt": attempt,
                }
            )
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
                self._append_journal(
                    {
                        "schema_version": 1,
                        "event": "request_finished",
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "http_status": None,
                        "provider_status": "transport_error",
                        "finish_reason": None,
                        "usage": observed,
                        "latency_seconds": latency,
                        "protocol_outcome": "transport_error",
                        "response_hash": None,
                    }
                )
                if attempt == 1:
                    if self._retry_delay:
                        time.sleep(self._retry_delay)
                    continue
                return self._failure(attempt, total_tool_calls, total_usage, "transport_error")

            latency = max(0.0, float(self._monotonic() - started))
            raw = _response_bytes(response)
            response_hash = _sha256(raw)
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
                self._append_journal(
                    {
                        "schema_version": 1,
                        "event": "request_finished",
                        "request_hash": request_hash,
                        "attempt": attempt,
                        "http_status": status,
                        "provider_status": "provider_error",
                        "finish_reason": None,
                        "usage": observed,
                        "latency_seconds": latency,
                        "protocol_outcome": "provider_error",
                        "response_hash": response_hash,
                    }
                )
                if attempt == 1 and self._transient(status):
                    if self._retry_delay:
                        time.sleep(self._retry_delay)
                    continue
                return self._failure(attempt, total_tool_calls, total_usage, provider_error)

            outcome, choice_id, finish_reason, calls = self._parse_response(value, allowed)
            total_tool_calls += calls
            self._append_journal(
                {
                    "schema_version": 1,
                    "event": "request_finished",
                    "request_hash": request_hash,
                    "attempt": attempt,
                    "http_status": status,
                    "provider_status": "ok",
                    "finish_reason": finish_reason,
                    "usage": observed,
                    "latency_seconds": latency,
                    "protocol_outcome": outcome,
                    "response_hash": response_hash,
                }
            )
            if outcome == "valid_choose_card":
                return {
                    "ok": True,
                    "choice_id": choice_id,
                    "api_attempts": attempt,
                    "tool_calls": total_tool_calls,
                    "usage": dict(total_usage),
                    "error": None,
                }
            if attempt == 1:
                correction = outcome
                continue
            return self._failure(attempt, total_tool_calls, total_usage, outcome)

        raise AssertionError("bounded attempt loop exhausted")


__all__ = ["OllamaPairChooser", "REFERENCE_KEYS"]

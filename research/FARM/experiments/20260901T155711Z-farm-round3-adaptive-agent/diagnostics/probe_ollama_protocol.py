#!/usr/bin/env python3
"""Sanitized Ollama Cloud protocol probe for structured output versus tools."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def load_environment(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def content_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def chat_options(max_generation_tokens: int) -> dict:
    return {"temperature": 0, "seed": 42, "num_predict": max_generation_tokens}


def think_value(label: str) -> bool | str | None:
    return {"default": None, "false": False, "low": "low"}[label]


def message_diagnostics(response: dict) -> dict:
    message = response.get("message") or {}
    return {
        "content_length": len(str(message.get("content") or "")),
        "thinking_length": len(str(message.get("thinking") or "")),
        "done_reason": response.get("done_reason"),
    }


def structured_probe(
    client: Any,
    model: str,
    *,
    max_generation_tokens: int,
    think: bool | str | None,
) -> dict:
    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["A"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    request = {
        "model": model,
        "messages": [{"role": "user", "content": "Return the permitted choice."}],
        "format": schema,
        "stream": False,
        "options": chat_options(max_generation_tokens),
    }
    if think is not None:
        request["think"] = think
    response = plain(client.chat(**request))
    content = str((response.get("message") or {}).get("content") or "")
    try:
        value = json.loads(content)
        valid = value == {"choice": "A"}
        error = None if valid else "schema_value_invalid"
    except json.JSONDecodeError:
        valid = False
        error = "json_invalid"
    return message_diagnostics(response) | {
        "valid": valid,
        "error": error,
        "content_preview": content[:200],
        "content_sha256_12": content_fingerprint(content),
        "prompt_tokens": int(response.get("prompt_eval_count") or 0),
        "completion_tokens": int(response.get("eval_count") or 0),
    }


def tool_probe(
    client: Any,
    model: str,
    *,
    max_generation_tokens: int,
    think: bool | str | None,
) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": "submit_choice",
            "description": "Submit the permitted choice.",
            "parameters": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": ["A"]}},
                "required": ["choice"],
                "additionalProperties": False,
            },
        },
    }
    request = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Call submit_choice exactly once with the permitted choice.",
        }],
        "tools": [tool],
        "stream": False,
        "options": chat_options(max_generation_tokens),
    }
    if think is not None:
        request["think"] = think
    response = plain(client.chat(**request))
    calls = (response.get("message") or {}).get("tool_calls") or []
    valid = False
    error = "tool_call_missing"
    observed = []
    for call in calls:
        function = call.get("function", call)
        observed.append({
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        })
    if len(calls) == 1:
        function = calls[0].get("function", calls[0])
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        valid = function.get("name") == "submit_choice" and arguments == {"choice": "A"}
        error = None if valid else "tool_call_invalid"
    return message_diagnostics(response) | {
        "valid": valid,
        "error": error,
        "observed_calls": observed,
        "tool_call_count": len(calls),
        "prompt_tokens": int(response.get("prompt_eval_count") or 0),
        "completion_tokens": int(response.get("eval_count") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--host-env", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-generation-tokens", type=int, default=384)
    parser.add_argument("--think", choices=("default", "false", "low"), default="default")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    if not 32 <= args.max_generation_tokens <= 2048:
        raise ValueError("max-generation-tokens must be between 32 and 2048")

    environment = load_environment(args.env_file)
    key = environment.get(args.key_env)
    host = environment.get(args.host_env) or "https://ollama.com"
    if not key:
        raise RuntimeError("Ollama key slot is unconfigured")

    from ollama import Client

    client = Client(host=host, headers={"Authorization": "Bearer " + key}, timeout=120)
    models = plain(client.list()).get("models") or []
    record = next(
        (item for item in models if (item.get("model") or item.get("name")) == args.model),
        None,
    )
    if not record or record.get("digest") != args.expected_digest:
        raise RuntimeError("model digest is unavailable or changed")

    thinking = think_value(args.think)
    result = {
        "model": args.model,
        "digest": args.expected_digest,
        "repeats": args.repeats,
        "max_generation_tokens": args.max_generation_tokens,
        "think": args.think,
        "structured": [
            structured_probe(
                client,
                args.model,
                max_generation_tokens=args.max_generation_tokens,
                think=thinking,
            )
            for _ in range(args.repeats)
        ],
        "tool": [
            tool_probe(
                client,
                args.model,
                max_generation_tokens=args.max_generation_tokens,
                think=thinking,
            )
            for _ in range(args.repeats)
        ],
    }
    serialized = json.dumps(result, sort_keys=True)
    if key in serialized:
        raise RuntimeError("secret persistence gate failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Probe Ollama Cloud's OpenAI-compatible structured and forced-tool paths."""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path


def load_environment(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def post(url: str, key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return {"status": response.status, "body": json.loads(response.read())}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return {"status": exc.code, "error_preview": body[:300]}


def structured_probe(url: str, key: str, model: str) -> dict:
    schema = {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": ["A"]}},
        "required": ["choice"],
        "additionalProperties": False,
    }
    result = post(url, key, {
        "model": model,
        "messages": [{"role": "user", "content": "Return the permitted choice."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "choice_response", "strict": True, "schema": schema},
        },
        "temperature": 0,
        "seed": 42,
        "max_tokens": 384,
        "reasoning_effort": "low",
    })
    if "body" not in result:
        return result | {"valid": False}
    message = result["body"].get("choices", [{}])[0].get("message", {})
    content = str(message.get("content") or "")
    try:
        valid = json.loads(content) == {"choice": "A"}
    except json.JSONDecodeError:
        valid = False
    return {
        "status": result["status"],
        "valid": valid,
        "content_preview": content[:200],
        "finish_reason": result["body"].get("choices", [{}])[0].get("finish_reason"),
        "usage": result["body"].get("usage"),
    }


def tool_probe(url: str, key: str, model: str) -> dict:
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
    result = post(url, key, {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Call submit_choice exactly once with the permitted choice.",
        }],
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "submit_choice"}},
        "temperature": 0,
        "seed": 42,
        "max_tokens": 384,
        "reasoning_effort": "low",
    })
    if "body" not in result:
        return result | {"valid": False}
    choice = result["body"].get("choices", [{}])[0]
    calls = choice.get("message", {}).get("tool_calls") or []
    observed = []
    for call in calls:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        observed.append({"name": function.get("name"), "arguments": arguments})
    return {
        "status": result["status"],
        "valid": observed == [{"name": "submit_choice", "arguments": {"choice": "A"}}],
        "observed_calls": observed,
        "finish_reason": choice.get("finish_reason"),
        "usage": result["body"].get("usage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--host-env", required=True)
    args = parser.parse_args()
    environment = load_environment(args.env_file)
    key = environment.get(args.key_env)
    host = environment.get(args.host_env) or "https://ollama.com"
    if not key:
        raise RuntimeError("Ollama key slot is unconfigured")
    url = host.rstrip("/") + "/v1/chat/completions"
    result = {
        "model": args.model,
        "structured": structured_probe(url, key, args.model),
        "forced_tool": tool_probe(url, key, args.model),
    }
    serialized = json.dumps(result, sort_keys=True)
    if key in serialized:
        raise RuntimeError("secret persistence gate failed")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

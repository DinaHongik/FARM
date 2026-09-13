#!/usr/bin/env python3
"""Bounded one-shot and tool-loop Ollama Cloud function selectors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import DATASET_ID, read_json, redact_text, sha256_file, utc_now, write_json  # noqa: E402

SYSTEM_PROMPT = """You select one valid IFTTT trigger/action function pair for the user's request.
Never invent URLs. Use only candidates returned by tools or shown in the prompt. Preserve a coherent
trigger/action pair. Be concise and deterministic."""


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [plain(item) for item in value]
    return value


def parse_json_pair(text: str) -> dict[str, str]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.S | re.I)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    value = json.loads(cleaned)
    if set(value) != {"trigger_url", "action_url"}:
        raise ValueError("selection must have exactly trigger_url and action_url")
    if not all(isinstance(value[key], str) and value[key].startswith("https://ifttt.com/") for key in value):
        raise ValueError("selection URLs are invalid")
    return value


def response_usage(response: Any, wall_seconds: float) -> dict[str, Any]:
    value = plain(response)
    return {
        "wall_seconds": wall_seconds,
        "prompt_eval_count": value.get("prompt_eval_count"),
        "eval_count": value.get("eval_count"),
        "total_duration_ns": value.get("total_duration"),
        "load_duration_ns": value.get("load_duration"),
    }


def call_chat(client, *, model: str, messages: list[dict], tools=None):
    started = time.monotonic()
    response = client.chat(
        model=model,
        messages=messages,
        tools=tools,
        stream=False,
        options={"temperature": 0, "seed": 42},
    )
    return response, response_usage(response, time.monotonic() - started)


def one_shot(client, model: str, row: dict) -> dict:
    allowed = {
        "trigger_candidates": [
            {key: candidate[key] for key in ("url", "channel", "function_name", "text_schema")}
            for candidate in row["trigger_candidates"]
        ],
        "action_candidates": [
            {key: candidate[key] for key in ("url", "channel", "function_name", "text_schema")}
            for candidate in row["action_candidates"]
        ],
    }
    prompt = (
        f"User request: {row['query']}\nCandidates:\n{json.dumps(allowed, ensure_ascii=False)}\n"
        "Return ONLY JSON with exactly trigger_url and action_url. Cloud schema enforcement is unavailable."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    usages = []
    parse_failures = 0
    response, usage = call_chat(client, model=model, messages=messages)
    usages.append(usage)
    content = plain(response)["message"].get("content", "")
    try:
        pair = parse_json_pair(content)
    except (ValueError, json.JSONDecodeError):
        parse_failures += 1
        messages.extend([
            {"role": "assistant", "content": content},
            {"role": "user", "content": "Repair once: output only the required two-key JSON using candidate URLs."},
        ])
        response, usage = call_chat(client, model=model, messages=messages)
        usages.append(usage)
        pair = parse_json_pair(plain(response)["message"].get("content", ""))
    return {"pair": pair, "parse_failures": parse_failures, "calls": len(usages), "usage": usages}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_trigger_candidates",
            "description": "Retrieve the frozen top trigger functions for this user request.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_action_candidates",
            "description": "Retrieve the frozen top action functions for this user request.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_function",
            "description": "Inspect one candidate URL's full frozen schema text.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_pair",
            "description": "Submit the final trigger/action pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trigger_url": {"type": "string"},
                    "action_url": {"type": "string"},
                },
                "required": ["trigger_url", "action_url"],
                "additionalProperties": False,
            },
        },
    },
]


def tool_loop(client, model: str, row: dict, max_turns: int = 4) -> dict:
    trigger_by_url = {candidate["url"]: candidate for candidate in row["trigger_candidates"]}
    action_by_url = {candidate["url"]: candidate for candidate in row["action_candidates"]}
    all_by_url = trigger_by_url | action_by_url
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Configure this request: {row['query']}. Retrieve evidence, then call submit_pair."},
    ]
    trace, usages = [], []
    submitted = None
    invalid_calls = 0
    for turn in range(max_turns):
        final_turn = turn == max_turns - 1
        if final_turn:
            messages.append({
                "role": "user",
                "content": "Final turn: use the evidence already retrieved and call submit_pair now.",
            })
        available_tools = [TOOLS[-1]] if final_turn else TOOLS
        response, usage = call_chat(client, model=model, messages=messages, tools=available_tools)
        usages.append(usage)
        message = plain(response)["message"]
        messages.append(message)
        calls = message.get("tool_calls") or []
        if not calls:
            trace.append({"turn": turn + 1, "error": "no_tool_call"})
            break
        for call in calls:
            function = call.get("function", call)
            name = function.get("name")
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
                    invalid_calls += 1
            if name == "retrieve_trigger_candidates":
                output = list(trigger_by_url.values())
            elif name == "retrieve_action_candidates":
                output = list(action_by_url.values())
            elif name == "inspect_function":
                output = all_by_url.get(arguments.get("url"))
                if output is None:
                    output = {"error": "URL is not in the frozen candidate set"}
                    invalid_calls += 1
            elif name == "submit_pair":
                candidate = {
                    "trigger_url": arguments.get("trigger_url"),
                    "action_url": arguments.get("action_url"),
                }
                valid_catalog = candidate["trigger_url"] in trigger_by_url and candidate["action_url"] in action_by_url
                output = {"accepted": valid_catalog}
                if valid_catalog:
                    submitted = candidate
                else:
                    output["error"] = "Both URLs must come from their frozen candidate lists"
                    invalid_calls += 1
            else:
                output = {"error": "unknown tool"}
                invalid_calls += 1
            trace.append({"turn": turn + 1, "tool": name, "arguments": arguments, "output": output})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(output, ensure_ascii=False)})
        if submitted:
            break
    return {
        "pair": submitted,
        "turns": len(usages),
        "calls": len(usages),
        "usage": usages,
        "trace": trace,
        "invalid_tool_calls": invalid_calls,
        "protocol_valid": submitted is not None and invalid_calls == 0,
        "final_submit_turn_reserved": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-rows", type=int, default=5)
    parser.add_argument("--host", default=os.environ.get("OLLAMA_CLOUD_HOST", "https://ollama.com"))
    args = parser.parse_args()

    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError("OLLAMA_API_KEY is not configured in the process environment")
    from ollama import Client

    client = Client(host=args.host, headers={"Authorization": f"Bearer {key}"})
    tags = plain(client.list())
    models = tags.get("models") or []
    names = [model.get("model") or model.get("name") for model in models]
    preferred = ("gpt-oss:20b", "gpt-oss:120b")
    model_name = args.model or next((name for name in preferred if name in names), None)
    if not model_name or model_name not in names:
        raise RuntimeError("requested/preferred Ollama Cloud model is unavailable; inspect /api/tags")
    model_record = next(model for model in models if (model.get("model") or model.get("name")) == model_name)

    rows = read_json(args.candidates.resolve())[: args.max_rows]
    if not rows:
        raise RuntimeError("candidate subset is empty")
    manifest_path = args.candidate_manifest or args.candidates.with_suffix(".manifest.json")
    candidate_manifest = read_json(manifest_path.resolve())
    if candidate_manifest.get("dataset_id") != DATASET_ID or candidate_manifest.get("split") != "dev":
        raise RuntimeError("candidate manifest is not immutable Dataset v2 dev")
    candidate_hash = sha256_file(args.candidates.resolve())
    if candidate_manifest.get("output_sha256") != candidate_hash:
        raise RuntimeError("candidate file hash does not match its immutable dev manifest")

    output_rows = []
    for row in rows:
        valid_pairs = {(pair["trigger_url"], pair["action_url"]) for pair in row["valid_pairs"]}
        record = {"group_id": row["group_id"], "query": row["query"]}
        for name, runner in (("one_shot", one_shot), ("tool_agent", tool_loop)):
            try:
                outcome = runner(client, model_name, row)
                pair = outcome.get("pair")
                pair_tuple = (pair.get("trigger_url"), pair.get("action_url")) if pair else None
                outcome["exact_valid_pair"] = pair_tuple in valid_pairs
                outcome["catalog_valid"] = pair_tuple is not None and (
                    pair_tuple[0] in {x["url"] for x in row["trigger_candidates"]}
                    and pair_tuple[1] in {x["url"] for x in row["action_candidates"]}
                )
                record[name] = outcome
            except Exception as exc:
                # Never serialize request headers or a client repr.
                record[name] = {"error_type": type(exc).__name__, "error": "bounded call failed", "exact_valid_pair": False}
        output_rows.append(record)

    def aggregate(name: str) -> dict:
        values = [row[name] for row in output_rows]
        successful = [value for value in values if "error_type" not in value]
        submitted = [value for value in successful if value.get("pair") is not None]
        return {
            "rows": len(values),
            "calls_completed": len(successful),
            "completed": len(submitted),
            "submission_rate": len(submitted) / len(values),
            "exact_pair_accuracy": sum(bool(value.get("exact_valid_pair")) for value in values) / len(values),
            "catalog_valid_rate": sum(bool(value.get("catalog_valid")) for value in values) / len(values),
            "mean_calls": sum(int(value.get("calls", 0)) for value in values) / len(values),
            "parse_failures": sum(int(value.get("parse_failures", 0)) for value in values),
            "invalid_tool_calls": sum(int(value.get("invalid_tool_calls", 0)) for value in values),
        }

    result = {
        "status": "completed",
        "completed_at": utc_now(),
        "dataset_id": DATASET_ID,
        "split": "dev",
        "candidate_file": str(args.candidates.resolve()),
        "candidate_sha256": candidate_hash,
        "model": model_name,
        "model_record": {key: value for key, value in model_record.items() if key not in {"details"}} | {"details": model_record.get("details")},
        "temperature": 0,
        "seed": 42,
        "structured_output_enforced": False,
        "structured_output_note": "Ollama Cloud does not support schema enforcement; local parse/validation plus one repair is used",
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "max_agent_turns": 4,
        "final_agent_turn_reserved_for_submit": True,
        "metrics": {"one_shot": aggregate("one_shot"), "tool_agent": aggregate("tool_agent")},
        "ragas": "not executed: final URL success is exact and valid tool trajectories are non-unique",
        "rows": output_rows,
    }
    payload = json.dumps(result, ensure_ascii=False)
    if key in payload:
        raise RuntimeError("secret-redaction gate failed")
    write_json(args.output.resolve(), result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

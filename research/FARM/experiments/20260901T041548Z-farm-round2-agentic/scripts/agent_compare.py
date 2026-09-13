#!/usr/bin/env python3
"""Resumable exact-URL comparison of Ollama selection and agent strategies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    DATASET_ID, read_json, sha256_file, update_agent_status, utc_now, write_json,
)

SYSTEM = (
    "Select the exact IFTTT trigger and action functions that satisfy the request. "
    "Never invent URLs. Use only supplied candidates or tool results."
)
ARMS = (
    "retrieval_top1", "one_shot_plain", "one_shot_schema", "reflect_agent",
    "role_agent", "tool_agent",
)
MAX_GENERATION_TOKENS = 512


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def function_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties,
                "required": required, "additionalProperties": False,
            },
        },
    }


SUBMIT = function_tool(
    "submit_pair", "Submit the final candidate pair.",
    {"trigger_url": {"type": "string"}, "action_url": {"type": "string"}},
    ["trigger_url", "action_url"],
)
SELECT_TRIGGER = function_tool(
    "select_trigger", "Select one trigger candidate URL.",
    {"trigger_url": {"type": "string"}}, ["trigger_url"],
)
SELECT_ACTION = function_tool(
    "select_action", "Select one action candidate URL.",
    {"action_url": {"type": "string"}}, ["action_url"],
)
AGENT_TOOLS = [
    function_tool("retrieve_triggers", "Return frozen trigger candidates.", {}, []),
    function_tool("retrieve_actions", "Return frozen action candidates.", {}, []),
    function_tool(
        "inspect_function", "Return schema evidence for one candidate URL.",
        {"url": {"type": "string"}}, ["url"],
    ),
    SUBMIT,
]


def usage_record(response: Any, seconds: float, attempts: int) -> dict:
    value = plain(response)
    return {
        "wall_seconds": seconds,
        "api_attempts": attempts,
        "prompt_eval_count": value.get("prompt_eval_count"),
        "eval_count": value.get("eval_count"),
        "total_duration_ns": value.get("total_duration"),
    }


def call_chat(client, model: str, messages: list[dict], tools: list[dict]) -> tuple[dict, dict]:
    last_error: Exception | None = None
    started = time.monotonic()
    for attempt in range(1, 3):
        try:
            response = client.chat(
                model=model, messages=messages, tools=tools, stream=False,
                options={"temperature": 0, "seed": 42, "num_predict": MAX_GENERATION_TOKENS},
            )
            usage = usage_record(response, time.monotonic() - started, attempt)
            return plain(response), usage
        except Exception as exc:  # cloud errors are retained generically, never with headers
            last_error = exc
            if attempt < 2:
                time.sleep(2)
    raise RuntimeError(f"Ollama call failed after retry: {type(last_error).__name__}")


def calls_from(message: dict) -> list[dict]:
    result = []
    for call in message.get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        result.append({"name": function.get("name"), "arguments": arguments})
    return result


def first_tool_args(response: dict, expected: str) -> dict | None:
    calls = calls_from(response.get("message") or {})
    return next((call["arguments"] for call in calls if call["name"] == expected), None)


def candidate_payload(row: dict, view: str) -> dict:
    key = f"text_{view}"
    return {
        f"{side}_candidates": [
            {
                "url": item["url"], "channel": item["channel"],
                "function_name": item["function_name"], key: item[key],
                "retrieval_rank": item["retrieval_rank"],
            }
            for item in row[f"{side}_candidates"]
        ]
        for side in ("trigger", "action")
    }


def normalize_pair(arguments: dict | None) -> dict | None:
    if not arguments:
        return None
    pair = {"trigger_url": arguments.get("trigger_url"), "action_url": arguments.get("action_url")}
    if not all(isinstance(value, str) for value in pair.values()):
        return None
    return pair


def retrieval_top1(row: dict) -> dict:
    return {
        "pair": {
            "trigger_url": row["trigger_candidates"][0]["url"],
            "action_url": row["action_candidates"][0]["url"],
        },
        "calls": 0, "usage": [], "protocol_valid": True,
    }


def one_shot(client, model: str, row: dict, view: str) -> dict:
    payload = candidate_payload(row, view)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": (
            f"Request: {row['query']}\nCandidates: {json.dumps(payload, ensure_ascii=False)}\n"
            "Choose one coherent pair and call submit_pair exactly once."
        )},
    ]
    response, usage = call_chat(client, model, messages, [SUBMIT])
    args = first_tool_args(response, "submit_pair")
    pair = normalize_pair(args)
    accepted = bool(
        pair
        and pair["trigger_url"] in {item["url"] for item in row["trigger_candidates"]}
        and pair["action_url"] in {item["url"] for item in row["action_candidates"]}
    )
    return {
        "pair": pair, "calls": 1, "usage": [usage],
        "protocol_valid": accepted, "view": view,
    }


def role_agent(client, model: str, row: dict) -> dict:
    plain_payload = candidate_payload(row, "plain")
    usages = []
    trigger_response, usage = call_chat(client, model, [
        {"role": "system", "content": SYSTEM + " Act only as the trigger-intent analyst."},
        {"role": "user", "content": f"Request: {row['query']}\n{json.dumps(plain_payload['trigger_candidates'], ensure_ascii=False)}"},
    ], [SELECT_TRIGGER])
    usages.append(usage)
    trigger_args = first_tool_args(trigger_response, "select_trigger") or {}
    action_response, usage = call_chat(client, model, [
        {"role": "system", "content": SYSTEM + " Act only as the action-intent analyst."},
        {"role": "user", "content": f"Request: {row['query']}\n{json.dumps(plain_payload['action_candidates'], ensure_ascii=False)}"},
    ], [SELECT_ACTION])
    usages.append(usage)
    action_args = first_tool_args(action_response, "select_action") or {}
    schema_payload = candidate_payload(row, "schema")
    verifier_prompt = {
        "request": row["query"], "trigger_proposal": trigger_args.get("trigger_url"),
        "action_proposal": action_args.get("action_url"), **schema_payload,
    }
    final_response, usage = call_chat(client, model, [
        {"role": "system", "content": SYSTEM + " Verify both specialists jointly; revise either if needed."},
        {"role": "user", "content": json.dumps(verifier_prompt, ensure_ascii=False)},
    ], [SUBMIT])
    usages.append(usage)
    final_args = first_tool_args(final_response, "submit_pair")
    final_pair = normalize_pair(final_args)
    accepted = bool(
        final_pair
        and final_pair["trigger_url"] in {item["url"] for item in row["trigger_candidates"]}
        and final_pair["action_url"] in {item["url"] for item in row["action_candidates"]}
    )
    return {
        "pair": final_pair, "calls": 3, "usage": usages,
        "protocol_valid": accepted,
        "specialists": {
            "trigger_url": trigger_args.get("trigger_url"),
            "action_url": action_args.get("action_url"),
        },
    }


def reflect_agent(client, model: str, row: dict, max_turns: int = 4) -> dict:
    """Max-call-matched single-agent propose/inspect/revise loop without retrieval tools."""
    triggers = {item["url"]: item for item in row["trigger_candidates"]}
    actions = {item["url"]: item for item in row["action_candidates"]}
    messages = [
        {"role": "system", "content": SYSTEM + " Reconsider your own proposal using feedback on every turn."},
        {"role": "user", "content": (
            f"Request: {row['query']}\nPlain candidates: "
            f"{json.dumps(candidate_payload(row, 'plain'), ensure_ascii=False)}\n"
            f"You have {max_turns} proposal turns. Call submit_pair on every turn."
        )},
    ]
    usages, proposals = [], []
    final_pair = None
    invalid = 0
    for turn in range(max_turns):
        response, usage = call_chat(client, model, messages, [SUBMIT])
        usages.append(usage)
        message = response.get("message") or {}
        messages.append(message)
        tool_args = first_tool_args(response, "submit_pair")
        pair = normalize_pair(tool_args)
        accepted = bool(pair and pair["trigger_url"] in triggers and pair["action_url"] in actions)
        invalid += int(not accepted)
        proposals.append({"turn": turn + 1, "pair": pair, "catalog_valid": accepted})
        # The scored proposal is always the last turn's output. Retaining an
        # earlier valid proposal would silently turn this into best-of-N.
        final_pair = pair if accepted else None
        if tool_args is not None:
            messages.append({
                "role": "tool", "tool_name": "submit_pair",
                "content": json.dumps({"accepted": accepted}),
            })
        else:
            # A tool response without a matching assistant tool call is an
            # invalid conversation for Ollama and can fail on the next turn.
            messages.append({
                "role": "user",
                "content": "No submit_pair call was received; call submit_pair with two candidate URLs.",
            })
        if turn < max_turns - 1:
            if accepted:
                evidence = {
                    "trigger_schema": triggers[pair["trigger_url"]]["text_schema"],
                    "action_schema": actions[pair["action_url"]]["text_schema"],
                }
            else:
                evidence = {"error": "proposal used a missing or cross-side URL"}
            messages.append({
                "role": "user",
                "content": (
                    f"Turn {turn + 1} evidence: {json.dumps(evidence, ensure_ascii=False)}\n"
                    "Critique the semantic fit, reconsider alternatives, and call submit_pair again."
                ),
            })
    return {
        "pair": final_pair, "calls": len(usages), "usage": usages,
        "turns": len(usages), "proposals": proposals,
        "invalid_proposals": invalid,
        "protocol_valid": final_pair is not None,
        "max_turns": max_turns,
    }


def tool_agent(client, model: str, row: dict, max_turns: int = 4) -> dict:
    trigger_by_url = {item["url"]: item for item in row["trigger_candidates"]}
    action_by_url = {item["url"]: item for item in row["action_candidates"]}
    all_by_url = trigger_by_url | action_by_url
    messages = [
        {"role": "system", "content": SYSTEM + " Retrieve evidence, inspect schemas when useful, then submit."},
        {"role": "user", "content": f"Configure this request: {row['query']}"},
    ]
    trace, usages = [], []
    submitted = None
    invalid_calls = 0
    inspections = 0
    for turn in range(max_turns):
        final_turn = turn == max_turns - 1
        if final_turn:
            messages.append({"role": "user", "content": "Final turn: call submit_pair now."})
        response, usage = call_chat(client, model, messages, [SUBMIT] if final_turn else AGENT_TOOLS)
        usages.append(usage)
        message = response.get("message") or {}
        messages.append(message)
        calls = calls_from(message)
        if not calls:
            trace.append({"turn": turn + 1, "error": "no_tool_call"})
            if not final_turn:
                messages.append({
                    "role": "user",
                    "content": "You must use a provided tool. Continue gathering evidence before final submission.",
                })
                continue
            break
        for call in calls:
            name, arguments = call["name"], call["arguments"]
            if name == "retrieve_triggers":
                output = [
                    {k: item[k] for k in ("url", "channel", "function_name", "text_plain", "retrieval_rank")}
                    for item in trigger_by_url.values()
                ]
            elif name == "retrieve_actions":
                output = [
                    {k: item[k] for k in ("url", "channel", "function_name", "text_plain", "retrieval_rank")}
                    for item in action_by_url.values()
                ]
            elif name == "inspect_function":
                item = all_by_url.get(arguments.get("url"))
                output = None if item is None else {
                    k: item[k] for k in ("url", "channel", "function_name", "text_schema")
                }
                inspections += int(item is not None)
                if item is None:
                    output = {"error": "URL is outside the frozen candidate set"}
                    invalid_calls += 1
            elif name == "submit_pair":
                pair = normalize_pair(arguments)
                accepted = bool(
                    pair and pair["trigger_url"] in trigger_by_url and pair["action_url"] in action_by_url
                )
                output = {"accepted": accepted}
                if accepted:
                    submitted = pair
                else:
                    output["error"] = "URLs must come from the correct frozen candidate sides"
                    invalid_calls += 1
            else:
                output = {"error": "unknown tool"}
                invalid_calls += 1
            trace.append({"turn": turn + 1, "tool": name, "arguments": arguments, "output": output})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(output, ensure_ascii=False)})
        if submitted:
            break
    return {
        "pair": submitted, "calls": len(usages), "usage": usages,
        "turns": len(usages), "inspections": inspections,
        "invalid_tool_calls": invalid_calls, "trace": trace,
        "protocol_valid": submitted is not None and invalid_calls == 0,
        "max_turns": max_turns, "final_submit_turn_reserved": True,
    }


def attach_scores(outcome: dict, row: dict) -> dict:
    pair = outcome.get("pair")
    pair_tuple = (pair.get("trigger_url"), pair.get("action_url")) if pair else None
    trigger_ids = {item["url"] for item in row["trigger_candidates"]}
    action_ids = {item["url"] for item in row["action_candidates"]}
    valid = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
    outcome["candidate_side_valid"] = bool(
        pair_tuple and pair_tuple[0] in trigger_ids and pair_tuple[1] in action_ids
    )
    # Compatibility alias; this means side-correct membership in the frozen
    # candidate set, not membership in the full FARM catalog.
    outcome["catalog_valid"] = outcome["candidate_side_valid"]
    outcome["trigger_exact"] = bool(pair_tuple and any(pair_tuple[0] == gold[0] for gold in valid))
    outcome["action_exact"] = bool(pair_tuple and any(pair_tuple[1] == gold[1] for gold in valid))
    outcome["exact_valid_pair"] = pair_tuple in valid
    outcome["gold_in_candidates"] = any(t in trigger_ids and a in action_ids for t, a in valid)
    return outcome


def run_scored(runner: Callable[[], dict], row: dict) -> dict:
    """Score a model outcome while letting infrastructure/code failures abort.

    Malformed or missing tool calls are explicit protocol outcomes returned by
    the arm itself. Exceptions are not predictions: the row must remain
    uncheckpointed so a resume can retry it without corrupting accuracy.
    """
    return attach_scores(runner(), row)


def token_sum(outcome: dict, key: str) -> int:
    return sum(int(item.get(key) or 0) for item in outcome.get("usage") or [])


def aggregate(rows: list[dict], arm: str) -> dict:
    values = [row["arms"][arm] for row in rows]
    ceiling = [value for value in values if value["gold_in_candidates"]]
    return {
        "rows": len(values),
        "trigger_exact": sum(value["trigger_exact"] for value in values) / len(values),
        "action_exact": sum(value["action_exact"] for value in values) / len(values),
        "joint_exact": sum(value["exact_valid_pair"] for value in values) / len(values),
        "conditional_joint_exact": (
            sum(value["exact_valid_pair"] for value in ceiling) / len(ceiling) if ceiling else None
        ),
        "candidate_ceiling": len(ceiling) / len(values),
        "candidate_side_valid_rate": sum(value["candidate_side_valid"] for value in values) / len(values),
        "catalog_valid_rate": sum(value["catalog_valid"] for value in values) / len(values),
        "protocol_valid_rate": sum(bool(value.get("protocol_valid")) for value in values) / len(values),
        "mean_calls": sum(int(value.get("calls", 0)) for value in values) / len(values),
        "total_calls": sum(int(value.get("calls", 0)) for value in values),
        "prompt_tokens": sum(token_sum(value, "prompt_eval_count") for value in values),
        "completion_tokens": sum(token_sum(value, "eval_count") for value in values),
        "failures": sum("error_type" in value for value in values),
        "unknown_failed_call_cost_rows": sum(bool(value.get("failed_call_cost_unknown")) for value in values),
        "exact_hit_vector": [int(value["exact_valid_pair"]) for value in values],
        "conditional_exact_hit_vector": [
            int(value["exact_valid_pair"]) for value in values if value["gold_in_candidates"]
        ],
    }


def exact_mcnemar(a: list[int], b: list[int]) -> dict:
    recoveries = sum(x == 0 and y == 1 for x, y in zip(a, b))
    regressions = sum(x == 1 and y == 0 for x, y in zip(a, b))
    n = recoveries + regressions
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(min(recoveries, regressions) + 1)) / (2 ** n)
        p = min(1.0, 2 * tail)
    return {"recoveries": recoveries, "regressions": regressions, "discordant": n, "exact_p": p}


def bootstrap_delta(a: list[int], b: list[int], seed: int = 42, iterations: int = 3000) -> dict:
    if len(a) != len(b):
        raise ValueError("paired vectors must have equal length")
    if not a:
        return {"delta": None, "ci95": [None, None], "iterations": iterations}
    rng = random.Random(seed)
    deltas = []
    n = len(a)
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(b[i] - a[i] for i in indices) / n)
    deltas.sort()
    return {
        "delta": sum(y - x for x, y in zip(a, b)) / n,
        "ci95": [deltas[int(0.025 * iterations)], deltas[int(0.975 * iterations) - 1]],
        "iterations": iterations,
    }


def compare(a: list[int], b: list[int]) -> dict:
    return exact_mcnemar(a, b) | bootstrap_delta(a, b)


def read_work(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"corrupt work JSONL at line {line_number}") from exc
    if len({row["group_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate group ID in resumable work file")
    return rows


def append_work(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-file", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--status-id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--host-env", required=True)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if bool(args.run_root) != bool(args.status_id):
        raise ValueError("run-root and status-id must be provided together")
    key = os.environ.get(args.key_env)
    host = os.environ.get(args.host_env) or "https://ollama.com"
    if not key:
        raise RuntimeError(f"credential environment variable is unconfigured: {args.key_env}")
    candidate_path = args.candidates.resolve()
    manifest = read_json(args.candidate_manifest.resolve())
    if manifest.get("dataset_id") != DATASET_ID or manifest.get("split") != "dev":
        raise RuntimeError("agent candidates are not Dataset v2 dev")
    candidate_hash = sha256_file(candidate_path)
    if manifest.get("output_sha256") != candidate_hash:
        raise RuntimeError("candidate hash does not match immutable manifest")
    router = read_json(args.router.resolve())
    if router.get("dataset_id") != DATASET_ID or router.get("calibration_split") != "reranker_train":
        raise RuntimeError("invalid router calibration")
    threshold = float(router["threshold"])
    if not math.isfinite(threshold):
        raise RuntimeError("router threshold is non-finite")
    candidates = read_json(candidate_path)[: args.max_rows]
    if not candidates:
        raise RuntimeError("empty agent candidate set")
    from ollama import Client
    client = Client(host=host, headers={"Authorization": f"Bearer {key}"}, timeout=240)
    tags = plain(client.list()).get("models") or []
    model_record = next((row for row in tags if (row.get("model") or row.get("name")) == args.model), None)
    if not model_record:
        raise RuntimeError("requested Ollama model is unavailable")
    digest = model_record.get("digest")
    if digest != args.expected_digest:
        raise RuntimeError(f"model digest changed: {digest}")

    output = args.output.resolve()
    work_file = args.work_file.resolve()
    progress_path = args.progress.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing final agent output: {output}")
    if work_file.exists() and not args.resume:
        raise RuntimeError(f"work file exists; use --resume: {work_file}")
    completed = read_work(work_file) if args.resume else []
    done_ids = {row["group_id"] for row in completed}
    binding = {
        "dataset_id": DATASET_ID, "candidate_sha256": candidate_hash,
        "router_sha256": sha256_file(args.router.resolve()), "model": args.model,
        "model_digest": digest, "temperature": 0, "seed": 42,
        "max_generation_tokens_per_call": MAX_GENERATION_TOKENS,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "arms": [*ARMS, "routed_tool_agent"],
        "max_rows": args.max_rows,
        "selected_group_ids_sha256": hashlib.sha256(
            json.dumps([row["group_id"] for row in candidates], separators=(",", ":")).encode()
        ).hexdigest(),
        "protocol": {"reflect_max_turns": 4, "tool_max_turns": 4, "role_calls": 3},
        "system_prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
    }
    if args.resume:
        if not progress_path.is_file():
            raise RuntimeError("resume requires the original progress binding")
        old_binding = read_json(progress_path).get("binding")
        if old_binding != binding:
            raise RuntimeError("resume binding changed")
    else:
        write_json(progress_path, {
            "status": "starting", "updated_at": utc_now(), "binding": binding,
            "completed_rows": 0, "target_rows": len(candidates),
        })
    if args.status_id:
        update_agent_status(args.run_root.resolve(), args.status_id, {
            "phase": "running", "model": args.model, "model_digest": digest,
            "completed_rows": len(completed), "target_rows": len(candidates),
            "started_at": utc_now(),
        })

    for index, row in enumerate(candidates, start=1):
        if row["group_id"] in done_ids:
            continue
        arms = {
            "retrieval_top1": attach_scores(retrieval_top1(row), row),
            "one_shot_plain": run_scored(lambda: one_shot(client, args.model, row, "plain"), row),
            "one_shot_schema": run_scored(lambda: one_shot(client, args.model, row, "schema"), row),
            "reflect_agent": run_scored(lambda: reflect_agent(client, args.model, row), row),
            "role_agent": run_scored(lambda: role_agent(client, args.model, row), row),
            "tool_agent": run_scored(lambda: tool_agent(client, args.model, row), row),
        }
        successful_llm_arms = [
            arm for arm in ARMS[1:]
            if "error_type" not in arms[arm] and int(arms[arm].get("calls", 0)) > 0
        ]
        if not successful_llm_arms:
            raise RuntimeError("all Ollama arms failed for one row; aborting instead of producing false progress")
        routed = dict(arms["tool_agent"] if float(row["routing_confidence"]) <= threshold else arms["retrieval_top1"])
        routed["routed_to_agent"] = float(row["routing_confidence"]) <= threshold
        routed["source_arm"] = "tool_agent" if routed["routed_to_agent"] else "retrieval_top1"
        arms["routed_tool_agent"] = routed
        record = {
            "group_id": row["group_id"], "query": row["query"],
            "routing_confidence": row["routing_confidence"], "arms": arms,
        }
        payload = json.dumps(record, ensure_ascii=False)
        if key in payload:
            raise RuntimeError("secret persistence gate failed")
        append_work(work_file, record)
        completed.append(record)
        write_json(progress_path, {
            "status": "running", "updated_at": utc_now(), "binding": binding,
            "completed_rows": len(completed), "target_rows": len(candidates),
            "last_group_id": row["group_id"],
        })
        if args.status_id:
            update_agent_status(args.run_root.resolve(), args.status_id, {
                "phase": "running", "completed_rows": len(completed),
                "target_rows": len(candidates), "updated_at": utc_now(),
                "latest_successful_llm_arms": successful_llm_arms,
                "latest_protocol_valid_llm_arms": [
                    arm for arm in ARMS[1:] if bool(arms[arm].get("protocol_valid"))
                ],
            })
        print(json.dumps({"completed": len(completed), "target": len(candidates), "group_id": row["group_id"]}), flush=True)

    completed.sort(key=lambda row: next(i for i, source in enumerate(candidates) if source["group_id"] == row["group_id"]))
    metrics = {arm: aggregate(completed, arm) for arm in (*ARMS, "routed_tool_agent")}
    baseline = metrics["retrieval_top1"]["exact_hit_vector"]
    baseline_conditional = metrics["retrieval_top1"]["conditional_exact_hit_vector"]
    comparisons = {
        f"{arm}_vs_retrieval_top1": {
            "end_to_end": compare(baseline, metrics[arm]["exact_hit_vector"]),
            "gold_in_candidates_subset": compare(
                baseline_conditional, metrics[arm]["conditional_exact_hit_vector"]
            ),
        }
        for arm in (*ARMS[1:], "routed_tool_agent")
    }
    comparisons["tool_agent_vs_reflect_agent"] = {
        "end_to_end": compare(
            metrics["reflect_agent"]["exact_hit_vector"], metrics["tool_agent"]["exact_hit_vector"]
        ),
        "gold_in_candidates_subset": compare(
            metrics["reflect_agent"]["conditional_exact_hit_vector"],
            metrics["tool_agent"]["conditional_exact_hit_vector"],
        ),
    }
    result = {
        "status": "completed", "completed_at": utc_now(), **binding,
        "split": "dev", "rows": len(completed), "router": router,
        "model_record": {"model": args.model, "digest": digest},
        "arms": list(metrics), "metrics": metrics, "comparisons": comparisons,
        "primary_comparison": "tool_agent_vs_reflect_agent",
        "cloud_latency_policy": "wall time is diagnostic only; calls and tokens are resource metrics",
        "ragas": "not executed; exact final URL pair is objective and valid tool paths are non-unique",
        "work_file": str(work_file), "work_sha256": sha256_file(work_file),
        "candidate_file": str(candidate_path), "candidate_manifest": manifest,
        "rows_detail": completed,
    }
    serialized = json.dumps(result, ensure_ascii=False)
    if key in serialized:
        raise RuntimeError("secret persistence gate failed")
    write_json(output, result)
    write_json(progress_path, {
        "status": "completed", "completed_at": utc_now(), "binding": binding,
        "completed_rows": len(completed), "target_rows": len(candidates),
        "output": str(output), "output_sha256": sha256_file(output),
    })
    if args.status_id:
        update_agent_status(args.run_root.resolve(), args.status_id, {
            "phase": "complete", "completed_rows": len(completed),
            "target_rows": len(candidates), "completed_at": utc_now(),
            "output": str(output), "output_sha256": sha256_file(output),
        })
    print(json.dumps({
        "status": "completed", "model": args.model, "digest": digest,
        "rows": len(completed),
        "joint_exact": {arm: values["joint_exact"] for arm, values in metrics.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Record only the exception class; exception text could contain a
        # provider response. The full traceback remains in the private log.
        try:
            run_arg = sys.argv[sys.argv.index("--run-root") + 1]
            status_arg = sys.argv[sys.argv.index("--status-id") + 1]
            update_agent_status(Path(run_arg).resolve(), status_arg, {
                "phase": "failed", "failed_at": utc_now(),
                "error_type": type(exc).__name__,
            })
        except (ValueError, IndexError, OSError):
            pass
        raise

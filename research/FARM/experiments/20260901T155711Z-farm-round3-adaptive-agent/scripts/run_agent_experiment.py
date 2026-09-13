#!/usr/bin/env python3
"""Resumable bounded Ollama evaluation over a frozen Dataset-v2 candidate lattice."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from adaptive_agent import (
    AdaptivePolicy,
    run_adaptive_case,
    run_one_shot_case,
    score_agent_trace,
    summarize_agent_rows,
)
from calibrate_router import feature_row
from experiment_core import DATASET_ID, load_candidate_artifact, read_json, sha256_file, write_json
from train_reranker import bootstrap_delta, exact_mcnemar


ARMS = (
    "retrieval_top1",
    "routed_names",
    "routed_plain",
    "routed_names_to_schema",
    "routed_plain_to_schema",
    "always_plain_to_schema",
)
MAX_GENERATION_TOKENS = 384


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def tool_calls(response: dict) -> list[dict]:
    result = []
    for call in (response.get("message") or {}).get("tool_calls") or []:
        function = call.get("function", call)
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        result.append({"name": function.get("name"), "arguments": arguments})
    return result


class OllamaPairClient:
    """Production adapter that maps opaque candidate IDs to catalog URLs."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    @staticmethod
    def _candidate_payload(request: dict) -> tuple[dict, dict[str, str], dict[str, str]]:
        trigger_map, action_map = {}, {}
        payload = {"trigger_candidates": [], "action_candidates": []}
        for side, prefix, mapping in (("trigger", "T", trigger_map), ("action", "A", action_map)):
            for item in request[f"{side}_candidates"]:
                candidate_id = f"{prefix}{item['retrieval_rank']}"
                mapping[candidate_id] = item["url"]
                public = {key: value for key, value in item.items() if key != "url"}
                public["candidate_id"] = candidate_id
                payload[f"{side}_candidates"].append(public)
        for label, source in (("PRIOR", request.get("prior_proposal")), ("BASELINE", request.get("baseline_proposal"))):
            if not source:
                continue
            for side, prefix, mapping in (("trigger", "T", trigger_map), ("action", "A", action_map)):
                candidate_id = f"{prefix}_{label}"
                mapping[candidate_id] = source[f"{side}_url"]
                enriched = source.get(f"{side}_candidate")
                if isinstance(enriched, dict):
                    public = {key: value for key, value in enriched.items() if key != "url"}
                    public.update({"candidate_id": candidate_id, "role": label.lower()})
                else:
                    public = {
                        "candidate_id": candidate_id,
                        "service": source[f"{side}_url"].split("/")[3] if "/" in source[f"{side}_url"] else source[f"{side}_url"],
                        "role": label.lower(),
                    }
                payload[f"{side}_candidates"].append(public)
        return payload, trigger_map, action_map

    def select_pair(self, request: dict) -> dict:
        payload, trigger_map, action_map = self._candidate_payload(request)
        tool = {
            "type": "function",
            "function": {
                "name": "submit_pair",
                "description": "Submit one coherent trigger/action candidate pair.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "trigger_id": {"type": "string", "enum": sorted(trigger_map)},
                        "action_id": {"type": "string", "enum": sorted(action_map)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "request_more": {"type": "boolean"},
                    },
                    "required": ["trigger_id", "action_id", "confidence", "request_more"],
                    "additionalProperties": False,
                },
            },
        }

        prompt = {
            "request": request["query"], "iteration": request["iteration"],
            "evidence_view": request["view"], **payload,
            "rules": [
                "Reason jointly about the event and resulting action.",
                "Use only candidate IDs shown here; never invent a service or function.",
                "Set request_more=true only if this batch cannot resolve the request.",
                "Call submit_pair exactly once.",
            ],
        }
        accumulated_usage = {"prompt_eval_count": 0, "eval_count": 0, "total_duration": 0}
        last_error: Exception | None = None
        logical_started = time.monotonic()
        for attempt in range(1, 3):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": (
                            "You configure IFTTT requests by selecting supplied candidates. "
                            "Candidate IDs are opaque; judge only the supplied service/function/schema evidence."
                        )},
                        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                    ],
                    tools=[tool], stream=False,
                    options={"temperature": 0, "seed": 42, "num_predict": MAX_GENERATION_TOKENS},
                )
                value = plain(response)
                for key in accumulated_usage:
                    accumulated_usage[key] += int(value.get(key) or 0)
                call = next((item for item in tool_calls(value) if item["name"] == "submit_pair"), None)
                if not call:
                    raise ValueError("model omitted submit_pair")
                arguments = call["arguments"]
                required = {"trigger_id", "action_id", "confidence", "request_more"}
                if (
                    not isinstance(arguments, dict)
                    or set(arguments) != required
                    or not isinstance(arguments["trigger_id"], str)
                    or not isinstance(arguments["action_id"], str)
                    or isinstance(arguments["confidence"], bool)
                    or not isinstance(arguments["confidence"], (int, float))
                    or not math.isfinite(float(arguments["confidence"]))
                    or not 0.0 <= float(arguments["confidence"]) <= 1.0
                    or not isinstance(arguments["request_more"], bool)
                ):
                    raise ValueError("selection_schema_invalid")
                if arguments["trigger_id"] not in trigger_map or arguments["action_id"] not in action_map:
                    raise ValueError("candidate_id_outside_batch")
                return {
                    "trigger_url": trigger_map[arguments["trigger_id"]],
                    "action_url": action_map[arguments["action_id"]],
                    "confidence": float(arguments["confidence"]),
                    "request_more": arguments["request_more"],
                    "api_attempts": attempt,
                    "usage": accumulated_usage | {"wall_seconds": time.monotonic() - logical_started},
                }
            except Exception as exc:  # external provider/protocol boundary; only the class is retained
                last_error = exc
                if attempt < 2:
                    time.sleep(1)
        safe_codes = {
            "model omitted submit_pair": "tool_call_missing",
            "selection_schema_invalid": "selection_schema_invalid",
            "candidate_id_outside_batch": "candidate_id_outside_batch",
        }
        detail = safe_codes.get(str(last_error))
        if detail is None:
            status_code = getattr(last_error, "status_code", None)
            status_suffix = f"_status_{int(status_code)}" if isinstance(status_code, int) else ""
            detail = f"provider_{type(last_error).__name__}{status_suffix}"
        if detail.startswith("provider_"):
            raise RuntimeError(f"Ollama selection failed after bounded retry: {detail}")
        baseline = request.get("baseline_proposal")
        if baseline:
            fallback_trigger = baseline["trigger_url"]
            fallback_action = baseline["action_url"]
        else:
            fallback_trigger = min(
                request["trigger_candidates"], key=lambda item: int(item["retrieval_rank"]),
            )["url"]
            fallback_action = min(
                request["action_candidates"], key=lambda item: int(item["retrieval_rank"]),
            )["url"]
        return {
            "trigger_url": fallback_trigger,
            "action_url": fallback_action,
            "confidence": 0.0,
            "request_more": False,
            "api_attempts": 2,
            "protocol_fallback": detail,
            "usage": accumulated_usage | {"wall_seconds": time.monotonic() - logical_started},
        }


def router_probability(features: list[float], model: dict) -> float:
    standardized = [
        (value - mean) / scale if scale else 0.0
        for value, mean, scale in zip(features, model["mean"], model["scale"])
    ]
    logit = float(model["intercept"]) + sum(value * weight for value, weight in zip(standardized, model["coef"]))
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


def router_score(row: dict, artifact: dict) -> float:
    features = feature_row(row)
    return (
        router_probability(features, artifact["models"]["wrong"])
        * router_probability(features, artifact["models"]["covered"])
    )


def inference_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if key not in {"valid_pairs", "gold_trigger_urls", "gold_action_urls"}}


def retrieval_trace(row: dict) -> dict:
    return {
        "case_id": row["group_id"],
        "final_pair": {
            "trigger_url": row["trigger_candidates"][0]["url"],
            "action_url": row["action_candidates"][0]["url"],
        },
        "calls": 0, "api_attempts": 0, "iterations": [],
        "stateless_between_iterations": True,
    }


def attach_score(trace: dict, row: dict, corpora: dict[str, dict[str, dict]], routed: bool) -> dict:
    score = score_agent_trace(trace, row["valid_pairs"])
    pair = sorted(
        row["valid_pairs"],
        key=lambda value: (value["trigger_url"], value["action_url"]),
    )[0]
    return trace | score | {
        "routed": routed,
        "trigger_service": corpora["trigger"][pair["trigger_url"]]["channel"],
        "action_service": corpora["action"][pair["action_url"]]["channel"],
    }


def rank_bucket(row: dict) -> str:
    valid = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
    trigger_rank = {item["url"]: index for index, item in enumerate(row["trigger_candidates"], start=1)}
    action_rank = {item["url"]: index for index, item in enumerate(row["action_candidates"], start=1)}
    depth = min(
        (max(trigger_rank[t], action_rank[a]) for t, a in valid if t in trigger_rank and a in action_rank),
        default=None,
    )
    if depth == 1:
        return "rank1"
    if depth is not None and depth <= 5:
        return "rank2_5"
    if depth is not None and depth <= 10:
        return "rank6_10"
    return "outside10"


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def usage_totals(trace: dict) -> dict:
    usage = [iteration["selection"].get("usage") or {} for iteration in trace.get("iterations", [])]
    return {
        "prompt_tokens": sum(int(item.get("prompt_eval_count") or 0) for item in usage),
        "completion_tokens": sum(int(item.get("eval_count") or 0) for item in usage),
        "provider_duration_ns": sum(int(item.get("total_duration") or 0) for item in usage),
    }


def aggregate(records: list[dict], arm: str) -> dict:
    rows = [record["arms"][arm] | usage_totals(record["arms"][arm]) for record in records]
    summary = summarize_agent_rows(rows)
    summary.update({
        "api_attempts": sum(int(row.get("api_attempts", 0)) for row in rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
        "completion_tokens": sum(row["completion_tokens"] for row in rows),
    })
    buckets = {}
    for bucket in ("rank1", "rank2_5", "rank6_10", "outside10"):
        selected = [record["arms"][arm] for record in records if record["rank_bucket"] == bucket]
        buckets[bucket] = {
            "cases": len(selected),
            "accuracy": sum(bool(row["exact_pair"]) for row in selected) / len(selected) if selected else None,
            "calls": sum(int(row.get("calls", 0)) for row in selected),
        }
    summary["by_rank_bucket"] = buckets
    summary["exact_hit_vector"] = [int(record["arms"][arm]["exact_pair"]) for record in records]
    return summary


def load_environment(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--key-env", required=True)
    parser.add_argument("--host-env", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-file", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--no-always-on", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite final agent output")
    rows, manifest = load_candidate_artifact(args.candidates, args.manifest)
    if manifest["split"] != "dev":
        raise ValueError("agent evaluation requires dev")
    rows = sorted(rows, key=lambda row: hashlib.sha256(row["group_id"].encode()).hexdigest())[:args.max_rows]
    router = read_json(args.router)
    if router.get("dataset_id") != DATASET_ID or router.get("calibration_split") != "reranker_train":
        raise ValueError("invalid recoverability router")
    environment = load_environment(args.env_file)
    key = environment.get(args.key_env)
    host = environment.get(args.host_env) or "https://ollama.com"
    if not key:
        raise RuntimeError("Ollama key slot is unconfigured")

    from ollama import Client
    client = Client(host=host, headers={"Authorization": "Bearer " + key}, timeout=240)
    models = plain(client.list()).get("models") or []
    model_record = next((item for item in models if (item.get("model") or item.get("name")) == args.model), None)
    if not model_record or model_record.get("digest") != args.expected_digest:
        raise RuntimeError("Ollama model/digest is unavailable or changed")
    llm = OllamaPairClient(client, args.model)
    corpus = {
        "trigger": {row["url"]: row for row in read_json(args.data_root / "corpus" / "triggers.json")},
        "action": {row["url"]: row for row in read_json(args.data_root / "corpus" / "actions.json")},
    }
    active_arms = list(ARMS[:-1] if args.no_always_on else ARMS)
    binding = {
        "dataset_id": DATASET_ID, "split": "dev", "candidate_sha256": manifest["output_sha256"],
        "router_sha256": sha256_file(args.router), "model": args.model,
        "model_digest": args.expected_digest, "max_rows": len(rows), "arms": active_arms,
        "selected_group_ids_sha256": hashlib.sha256(json.dumps([row["group_id"] for row in rows]).encode()).hexdigest(),
        "temperature": 0, "seed": 42, "max_generation_tokens": MAX_GENERATION_TOKENS,
        "opaque_candidate_ids": True, "fresh_context_each_iteration": True,
    }
    completed = read_jsonl(args.work_file) if args.resume else []
    if args.work_file.exists() and not args.resume:
        raise RuntimeError("work file exists; use --resume")
    if args.resume:
        old = read_json(args.progress).get("binding")
        if old != binding:
            raise RuntimeError("resume binding changed")
    done = {row["group_id"] for row in completed}
    write_json(args.progress, {"phase": "running", "binding": binding, "completed_rows": len(completed), "target_rows": len(rows)})

    for row in rows:
        if row["group_id"] in done:
            continue
        score = router_score(row, router)
        routed = score >= float(router["threshold"])
        infer = inference_row(row)
        base = retrieval_trace(row)
        arms = {"retrieval_top1": attach_score(base, row, corpus, False)}
        if routed:
            arms["routed_names"] = attach_score(run_one_shot_case(infer, view="names", client=llm), row, corpus, True)
            arms["routed_plain"] = attach_score(run_one_shot_case(infer, view="plain", client=llm), row, corpus, True)
            arms["routed_names_to_schema"] = attach_score(run_adaptive_case(
                infer,
                AdaptivePolicy(view="names", second_view="schema", confidence_threshold=0.60, verify_overrides=True),
                llm,
            ), row, corpus, True)
            arms["routed_plain_to_schema"] = attach_score(run_adaptive_case(
                infer,
                AdaptivePolicy(view="plain", second_view="schema", confidence_threshold=0.60, verify_overrides=True),
                llm,
            ), row, corpus, True)
        else:
            for arm in ("routed_names", "routed_plain", "routed_names_to_schema", "routed_plain_to_schema"):
                arms[arm] = attach_score(dict(base), row, corpus, False)
        if "always_plain_to_schema" in active_arms:
            arms["always_plain_to_schema"] = attach_score(run_adaptive_case(
                infer,
                AdaptivePolicy(view="plain", second_view="schema", confidence_threshold=0.60, verify_overrides=True),
                llm,
            ), row, corpus, True)
        record = {
            "group_id": row["group_id"], "router_score": score, "routed": routed,
            "rank_bucket": rank_bucket(row), "arms": arms,
        }
        serialized = json.dumps(record, ensure_ascii=False)
        if key in serialized:
            raise RuntimeError("secret persistence gate failed")
        append_jsonl(args.work_file, record)
        completed.append(record)
        write_json(args.progress, {
            "phase": "running", "binding": binding, "completed_rows": len(completed),
            "target_rows": len(rows), "last_group_id": row["group_id"],
        })
        print(json.dumps({"completed": len(completed), "target": len(rows), "routed": routed}), flush=True)

    order = {row["group_id"]: index for index, row in enumerate(rows)}
    completed.sort(key=lambda row: order[row["group_id"]])
    metrics = {arm: aggregate(completed, arm) for arm in active_arms}
    baseline = metrics["retrieval_top1"]["exact_hit_vector"]
    comparisons = {
        f"{arm}_vs_retrieval_top1": exact_mcnemar(baseline, metrics[arm]["exact_hit_vector"])
        | bootstrap_delta(baseline, metrics[arm]["exact_hit_vector"], seed=42)
        for arm in active_arms if arm != "retrieval_top1"
    }
    result = {
        "status": "completed", **binding, "router": router, "metrics": metrics,
        "comparisons": comparisons,
        "primary_comparison": "routed_plain_to_schema_vs_retrieval_top1",
        "ragas_policy": "selection exactness is primary; Ragas runs separately as an audited ID-metric adapter",
        "work_sha256": sha256_file(args.work_file), "rows_detail": completed,
    }
    if key in json.dumps(result, ensure_ascii=False):
        raise RuntimeError("secret persistence gate failed")
    write_json(args.output, result)
    write_json(args.progress, {
        "phase": "complete", "binding": binding, "completed_rows": len(completed),
        "target_rows": len(rows), "output": str(args.output), "output_sha256": sha256_file(args.output),
    })
    print(json.dumps({"status": "completed", "model": args.model, "rows": len(completed)}, indent=2))


if __name__ == "__main__":
    main()

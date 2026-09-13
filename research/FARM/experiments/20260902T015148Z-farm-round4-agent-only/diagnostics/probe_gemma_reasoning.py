#!/usr/bin/env python3
"""Probe Gemma reasoning budgets on one exact exploratory FARM request."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "scripts"))

from ollama_adapter import OllamaSelector  # noqa: E402
from resolver import ResolverPolicy, resolve  # noqa: E402
from run_round4 import _load_environment, inference_case, read_json  # noqa: E402


class CaptureSelector:
    def __init__(self) -> None:
        self.request: dict | None = None

    def select(self, request: dict) -> dict:
        self.request = request
        return {
            "ok": False,
            "trigger_id": None,
            "action_id": None,
            "api_attempts": 0,
            "tool_calls": 0,
            "usage": {},
            "error": "capture_only",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args()

    rows = read_json(args.candidates)
    row = sorted(rows, key=lambda value: hashlib.sha256(value["group_id"].encode()).hexdigest())[0]
    corpora = {
        "trigger": read_json(args.data_root / "corpus/triggers.json"),
        "action": read_json(args.data_root / "corpus/actions.json"),
    }
    prompt = (RUN_ROOT / "configs/resolver_prompt.txt").read_text(encoding="utf-8").strip()
    capture = CaptureSelector()
    resolve(
        inference_case(row, corpora, hierarchy=False),
        ResolverPolicy.function_topk(5, view="plain", instruction=prompt),
        capture,
    )
    if capture.request is None:
        raise AssertionError("resolver did not produce a request")

    environment = _load_environment(args.env_file)
    key = environment["OLLAMA_API_KEY"]
    host = environment["OLLAMA_CLOUD_HOST"]
    selector = OllamaSelector(
        host,
        key,
        "gemma4:31b",
        Path("/tmp/farm-round4-gemma-probe-wal.jsonl"),
        max_tokens=768,
    )
    public, trigger_ids, action_ids = selector._prepare_request(capture.request)
    results = []
    try:
        for effort, budget in (("none", 768), ("none", 1536), ("low", 1536)):
            payload = selector._payload(public, trigger_ids, action_ids, None)
            payload["reasoning_effort"] = effort
            payload["max_tokens"] = budget
            response = selector._client.post(
                selector._endpoint,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=payload,
                timeout=240,
            )
            value = response.json()
            outcome, _trigger, _action, finish_reason, observed_calls = selector._parse_response(
                value, set(trigger_ids), set(action_ids)
            )
            results.append({
                "reasoning_effort": effort,
                "max_tokens": budget,
                "http_status": response.status_code,
                "finish_reason": finish_reason,
                "protocol_outcome": outcome,
                "observed_tool_calls": observed_calls,
                "usage": value.get("usage"),
            })
    finally:
        selector.close()
    serialized = json.dumps(results, sort_keys=True)
    if key in serialized:
        raise RuntimeError("secret persistence gate failed")
    print(json.dumps({"model": "gemma4:31b", "results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

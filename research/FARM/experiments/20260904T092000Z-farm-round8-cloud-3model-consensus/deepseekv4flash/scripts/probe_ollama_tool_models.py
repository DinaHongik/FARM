#!/usr/bin/env python3
"""Probe Ollama models with one synthetic strict FARM-shaped tool request.

The request contains no FARM query, endpoint, schema, split, or credential.
Only protocol/accounting metadata and the synthetic choice are persisted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def synthetic_request() -> dict[str, Any]:
    triggers = [
        {
            "candidate_id": f"T{index:02d}",
            "service_name": f"synthetic-trigger-service-{index}",
            "function_name": f"synthetic-trigger-function-{index}",
            "evidence": f"Synthetic trigger {index}; no external data.",
        }
        for index in range(1, 11)
    ]
    actions = [
        {
            "candidate_id": f"A{index:02d}",
            "service_name": f"synthetic-action-service-{index}",
            "function_name": f"synthetic-action-function-{index}",
            "evidence": f"Synthetic action {index}; no external data.",
        }
        for index in range(1, 11)
    ]
    return {
        "schema_version": "farm_r7_factorized_request_v1",
        "request_id": "R0000000000000000",
        "phase": "factorized_selection",
        "query": "When the synthetic source emits a red token, store that token.",
        "instruction": (
            "Choose independently. Prefer trigger T02 and action A03 because their "
            "synthetic descriptions are declared as the intended pair."
        ),
        "current_pair": {"trigger_choice": "T01", "action_choice": "A01"},
        "candidates": {"trigger": triggers, "action": actions},
        "output_contract": {
            "decision": [
                "KEEP", "CHANGE_TRIGGER", "CHANGE_ACTION", "CHANGE_BOTH", "ABSTAIN"
            ],
            "trigger_choice": ["KEEP", *[f"T{i:02d}" for i in range(1, 11)]],
            "action_choice": ["KEEP", *[f"A{i:02d}" for i in range(1, 11)]],
            "additional_properties": False,
        },
    }


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write_json_new(path: Path, value: Mapping[str, Any], secret: str) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if secret in payload:
        raise RuntimeError("credential persistence gate failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round7-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in [1,8]")

    scripts = args.round7_root.resolve() / "scripts"
    import sys

    sys.path.insert(0, str(scripts))
    from ollama_constraint_adapter import OllamaConstraintChooser

    environment = load_env(args.env_file.resolve())
    key = environment.get("OLLAMA_API_KEY", "")
    host = environment.get("OLLAMA_CLOUD_HOST", "")
    if not key or not host:
        raise RuntimeError("Ollama environment slots are not configured")

    from ollama import Client

    listed = plain(
        Client(host=host, headers={"Authorization": "Bearer " + key}, timeout=240).list()
    ).get("models", [])
    digests = {
        str(item.get("model") or item.get("name")): str(item.get("digest") or "")
        for item in listed
        if isinstance(item, Mapping)
    }
    missing = [model for model in args.models if model not in digests]
    if missing:
        raise RuntimeError(f"requested models are unavailable: {missing}")

    request = synthetic_request()

    def probe(model: str) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="farm-r8-ollama-probe-") as temporary:
            chooser = OllamaConstraintChooser(
                base_url=host,
                api_key=key,
                model=model,
                journal_path=Path(temporary) / "attempts.jsonl",
                timeout=240,
                max_tokens=256,
                reasoning_effort="none",
                retry_delay=1.0,
                seed=42,
            )
            started = time.monotonic()
            try:
                result = chooser.select(request)
            finally:
                chooser.close()
            return {
                "model": model,
                "digest": digests[model],
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "api_attempts": int(result.get("api_attempts", 0)),
                "tool_calls": int(result.get("tool_calls", 0)),
                "decision": result.get("decision"),
                "trigger_choice": result.get("trigger_choice"),
                "action_choice": result.get("action_choice"),
                "usage": result.get("usage"),
                "wall_seconds": time.monotonic() - started,
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(probe, args.models))
    output = {
        "schema_version": "farm_round8_synthetic_ollama_probe_v1",
        "input_policy": {
            "farm_data_sent": False,
            "synthetic_only": True,
            "request_sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "credential_persisted": False,
        },
        "results": results,
    }
    write_json_new(args.output.resolve(), output, key)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "models": len(results),
        "valid": sum(bool(row["ok"]) for row in results),
    }, sort_keys=True))


if __name__ == "__main__":
    main()


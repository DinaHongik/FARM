#!/usr/bin/env python3
"""Run exactly one public synthetic native-tool request through Ollama Cloud."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import sha256_file, write_json_atomic  # noqa: E402
from farm_r9.bfcl_runner import (  # noqa: E402
    BFCLCloudTransport,
    build_public_provider_gate_summary,
)
from farm_r9.cloud_limiter import OllamaCloudLimiter  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host-env", default="OLLAMA_CLOUD_HOST")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--shared-lock-directory", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9052026)
    parser.add_argument("--think", default="low")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.journal, args.cache_dir, args.summary):
        if path.exists():
            raise RuntimeError("provider-gate artifacts must use fresh absent paths")

    transport = BFCLCloudTransport(
        host=_required_environment(args.host_env),
        api_key=_required_environment(args.api_key_env),
        model=args.model,
        limiter=OllamaCloudLimiter(args.shared_lock_directory, max_concurrent=3),
        journal_path=args.journal,
        cache_directory=args.cache_dir,
        timeout_seconds=args.timeout_seconds,
        # The gate is contractually one physical request, with no retry.
        max_transport_attempts=1,
        temperature=args.temperature,
        seed=args.seed,
        think=None if args.think.casefold() == "none" else args.think,
    )
    response = transport.chat(
        semantic_id="bfcl-public-provider-gate-v1",
        messages=[
            {
                "role": "system",
                "content": "Call the provided function exactly once; do not answer in prose.",
            },
            {
                "role": "user",
                "content": "Look up the public code for item alpha.",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_public_code",
                    "description": "Return the public code for an item.",
                    "parameters": {
                        "type": "object",
                        "properties": {"item": {"type": "string"}},
                        "required": ["item"],
                    },
                },
            }
        ],
    )
    summary = build_public_provider_gate_summary(
        response,
        model=args.model,
        journal_sha256=sha256_file(args.journal),
    )
    write_json_atomic(args.summary, summary)
    os.chmod(args.summary, 0o600)
    # The terminal contract contains aggregates and hashes only—never provider
    # output, arguments, prompts, schemas, or credentials.
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Queue RecipeGen Noisy only after the paired BFCL arm is fully scored."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-summary", type=Path, required=True)
    parser.add_argument("--expected-registry", required=True)
    parser.add_argument("--round9-root", type=Path, required=True)
    parser.add_argument("--farm-root", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=43_200.0)
    return parser.parse_args(argv)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_exclusive(temporary, value)
    os.replace(temporary, path)


def _summary_complete(path: Path, expected_registry: str) -> tuple[bool, str]:
    if not path.is_file():
        return False, "summary_pending"
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False, "summary_not_stable"
    if not isinstance(value, dict):
        raise RuntimeError("BFCL summary is not an object")
    if value.get("registry_name") != expected_registry:
        raise RuntimeError("BFCL summary registry binding changed")
    if value.get("arm") != "native_tool_agent":
        raise RuntimeError("BFCL summary is not the native tool-agent arm")
    generation = value.get("generation")
    evaluation = value.get("evaluation")
    if not isinstance(generation, dict) or not isinstance(evaluation, dict):
        return False, "summary_incomplete"
    complete = (
        generation.get("n") == 150
        and generation.get("resumable_completed_cases") == 150
        and evaluation.get("n") == 150
        and evaluation.get("official_evaluator") is True
        and value.get("official_state_based_evaluator") is True
    )
    return complete, "summary_complete" if complete else "summary_incomplete"


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if not 5.0 <= args.poll_seconds <= 300.0:
        raise SystemExit("--poll-seconds must be between 5 and 300")
    if not 60.0 <= args.max_wait_seconds <= 86_400.0:
        raise SystemExit("--max-wait-seconds must be between 60 and 86400")
    wrapper = args.round9_root / "scripts" / "launch_deepseek_recipegen_noisy_dgx.sh"
    if not wrapper.is_file() or not args.farm_root.is_dir():
        raise SystemExit("required launch wrapper or FARM root is missing")
    if args.state_file.exists() or args.state_file.is_symlink():
        raise SystemExit("refusing to overwrite successor state")

    state = {
        "schema_version": "round9-recipegen-successor-state-v1",
        "status": "waiting_for_complete_bfcl_native_summary",
        "expected_registry": args.expected_registry,
        "successor": "deepseek_recipegen_noisy_n150",
    }
    _write_exclusive(args.state_file, state)
    deadline = time.monotonic() + args.max_wait_seconds
    while True:
        ready, observation = _summary_complete(
            args.bfcl_summary,
            args.expected_registry,
        )
        if ready:
            break
        if time.monotonic() >= deadline:
            state.update({"status": "timed_out", "last_observation": observation})
            _replace(args.state_file, state)
            raise SystemExit("timed out waiting for the BFCL native-agent summary")
        time.sleep(args.poll_seconds)

    completed = subprocess.run(
        ["bash", str(wrapper), str(args.round9_root), str(args.farm_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        state.update({"status": "launch_failed", "returncode": completed.returncode})
        _replace(args.state_file, state)
        raise SystemExit(
            "RecipeGen Noisy launch failed; inspect the access-controlled log"
        )
    state.update({"status": "launched"})
    _replace(args.state_file, state)
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

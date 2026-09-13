#!/usr/bin/env python3
"""Launch the paired Yao arm only after a complete reference arm exists.

This is an operational hand-off helper for long unattended runs.  It never
reads credentials from disk and never prints credential values; the named API
key variable must already be present in the launcher's environment.
"""

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
    parser.add_argument("--reference-records", type=Path, required=True)
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--child-log", type=Path, required=True)
    parser.add_argument("--child-pid-file", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--lease-dir", type=Path, required=True)
    parser.add_argument("--expected-n", type=int, default=150)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-wait-seconds", type=float, default=43_200.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def _secure_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _secure_replace_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_complete_reference(
    records_path: Path,
    aggregate_path: Path,
    *,
    expected_n: int,
) -> tuple[bool, str]:
    if not records_path.is_file() or not aggregate_path.is_file():
        return False, "reference_artifacts_pending"
    try:
        with aggregate_path.open("r", encoding="utf-8") as handle:
            aggregate = json.load(handle)
        with records_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False, "reference_artifacts_not_stable"
    if not isinstance(aggregate, dict) or aggregate.get("n") != expected_n:
        return False, "reference_aggregate_incomplete"
    if aggregate.get("arm") != "same_model_one_shot":
        raise RuntimeError("reference aggregate is not the required one-shot arm")
    if len(rows) != expected_n or not all(isinstance(row, dict) for row in rows):
        return False, "reference_records_incomplete"
    case_ids = [row.get("case_id") for row in rows]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise RuntimeError("reference records contain an invalid case ID")
    if len(case_ids) != len(set(case_ids)):
        raise RuntimeError("reference records contain duplicate case IDs")
    return True, "reference_complete"


def _run_command(args: argparse.Namespace, *, preflight: bool) -> list[str]:
    command = [
        str(args.python),
        "scripts/run_yao_experiment.py",
        "--cases",
        "prepared/interactive_ifttt/cases.jsonl",
        "--sample-manifest",
        "manifests/samples/interactive_ifttt.json",
        "--converted",
        "source/yao-converted.json",
        "--arm",
        "bounded_clarification_agent",
        "--output-dir",
        str(args.output_dir.resolve()),
        "--paired-reference",
        str(args.reference_records.resolve()),
        "--host",
        args.host,
        "--model",
        args.model,
        "--cloud",
        "--api-key-env",
        args.api_key_env,
        "--lease-dir",
        str(args.lease_dir.resolve()),
        "--timeout-seconds",
        str(args.timeout_seconds),
    ]
    if preflight:
        command.append("--preflight-only")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.expected_n != 150:
        raise SystemExit("the frozen Yao successor requires --expected-n 150")
    if not 5.0 <= args.poll_seconds <= 300.0:
        raise SystemExit("--poll-seconds must be between 5 and 300")
    if not 60.0 <= args.max_wait_seconds <= 86_400.0:
        raise SystemExit("--max-wait-seconds must be between 60 and 86400")
    if not os.environ.get(args.api_key_env):
        raise SystemExit(f"nonempty {args.api_key_env} must already be exported")
    if not args.worker_root.is_dir() or not args.python.is_file():
        raise SystemExit("worker root and Python executable must already exist")
    for path in (args.output_dir, args.child_log, args.child_pid_file, args.state_file):
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing to overwrite existing path: {path}")

    state = {
        "schema_version": "round9-successor-launch-state-v1",
        "status": "waiting_for_complete_reference",
        "expected_n": args.expected_n,
        "successor_arm": "bounded_clarification_agent",
        "model": args.model,
    }
    _secure_exclusive_json(args.state_file, state)
    deadline = time.monotonic() + args.max_wait_seconds
    while True:
        ready, reason = _load_complete_reference(
            args.reference_records,
            args.reference_aggregate,
            expected_n=args.expected_n,
        )
        if ready:
            break
        if time.monotonic() >= deadline:
            state.update({"status": "timed_out", "last_observation": reason})
            _secure_replace_json(args.state_file, state)
            raise SystemExit("timed out waiting for the complete reference arm")
        time.sleep(args.poll_seconds)

    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise SystemExit(
            "successor output appeared while waiting; refusing duplicate launch"
        )
    preflight = subprocess.run(
        _run_command(args, preflight=True),
        cwd=args.worker_root,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if preflight.returncode != 0:
        state.update({"status": "preflight_failed", "returncode": preflight.returncode})
        _secure_replace_json(args.state_file, state)
        raise SystemExit(
            "successor preflight failed; inspect the access-controlled launcher log"
        )

    args.child_log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_descriptor = os.open(
        args.child_log,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(log_descriptor, "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            _run_command(args, preflight=False),
            cwd=args.worker_root,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
    _secure_exclusive_json(args.child_pid_file, {"pid": process.pid})
    state.update({"status": "launched", "child_pid": process.pid})
    _secure_replace_json(args.state_file, state)
    print(json.dumps({"status": "launched", "child_pid": process.pid}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

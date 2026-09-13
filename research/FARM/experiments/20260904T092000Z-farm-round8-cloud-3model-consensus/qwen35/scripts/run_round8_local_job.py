#!/usr/bin/env python3
"""Exec one resumable Round8 arm against its frozen local Ollama port."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


RUN_ID = "20260904T074517Z-farm-round8-executable-agent"
MODEL = "granite4:small-h"
DIGEST = "2f8a7367d4416b508336f6170c295e8465ff004731ae563631196781299f750d"
ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
    "trigger_consensus_executable",
)
SMOKE_ROWS = 2
CONSENSUS_SMOKE_ROWS = 5


def _read_live_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        if pid <= 1:
            return None
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, OSError, ValueError):
        return None


def _atomic_pid(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _acquire_arm_lock(path: Path) -> int:
    """Acquire a nonblocking lock that remains held by the exec'd runner."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"another process owns local arm lock: {path.name}") from error
    try:
        # Python file descriptors are non-inheritable by default.  The runner
        # replaces this wrapper via execve, so inheritance is what keeps the
        # lock held for the complete experiment process lifetime.
        os.set_inheritable(descriptor, True)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--gpu", type=int, choices=range(5), required=True)
    parser.add_argument("--port", type=int, choices=range(11434, 11439), required=True)
    parser.add_argument("--model", choices=(MODEL,), required=True)
    parser.add_argument("--expected-digest", choices=(DIGEST,), required=True)
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument(
        "--smoke", type=int, choices=(SMOKE_ROWS, CONSENSUS_SMOKE_ROWS)
    )
    args = parser.parse_args(argv)
    expected_gpu = ARMS.index(args.arm)
    if args.gpu != expected_gpu or args.port != 11434 + expected_gpu:
        parser.error("arm, physical GPU, and loopback port do not match the frozen mapping")
    expected_smoke = (
        CONSENSUS_SMOKE_ROWS
        if args.arm == "trigger_consensus_executable"
        else SMOKE_ROWS
    )
    if args.smoke is not None and args.smoke != expected_smoke:
        parser.error(f"the frozen smoke size for {args.arm} is {expected_smoke}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    python = args.python.resolve()
    runner = run_root / "scripts" / "run_round8.py"
    if run_root.name != RUN_ID:
        raise RuntimeError(f"unexpected Round8 run root: {run_root}")
    if not runner.is_file():
        raise FileNotFoundError(f"missing Round8 runner: {runner}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"FARM Python is unavailable: {python}")
    if not (run_root / ".deps" / "dspy" / "__init__.py").is_file():
        raise FileNotFoundError("Round8 isolated dependencies are missing under .deps")

    namespace = "full" if args.smoke is None else f"smoke-{args.smoke}"
    artifact_root = run_root / "local_granite" / DIGEST / namespace
    paths = {
        name: artifact_root / name
        for name in ("logs", "pids", "progress", "records", "results", "attempts")
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    lock_descriptor = _acquire_arm_lock(
        artifact_root / "state" / f"{args.arm}.run.lock"
    )
    result_path = paths["results"] / f"{args.arm}.json"
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"malformed existing result: {result_path}") from error
        if isinstance(payload, dict):
            raise FileExistsError(f"refusing to rerun completed local arm: {args.arm}")
        raise RuntimeError(f"existing result is not a JSON object: {result_path}")
    pid_path = paths["pids"] / f"{args.arm}.pid"
    live_pid = _read_live_pid(pid_path)
    if live_pid is not None and live_pid != os.getpid():
        raise RuntimeError(f"another process owns local arm {args.arm}: pid={live_pid}")

    os.umask(0o077)
    _atomic_pid(pid_path)
    log_path = paths["logs"] / f"{args.arm}.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    if descriptor not in {sys.stdout.fileno(), sys.stderr.fileno()}:
        os.close(descriptor)
    host = f"http://127.0.0.1:{args.port}"
    print(
        f"START_LOCAL {datetime.now(timezone.utc).isoformat()} arm={args.arm} "
        f"gpu={args.gpu} host={host} pid={os.getpid()}",
        flush=True,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "FARM_ROUND8_PROCESS_SLOT": str(args.gpu),
            "FARM_ROUND8_TRANSPORT": "local_loopback",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "DSP_CACHEBOOL": "false",
            "PYTHONPATH": f"{run_root / '.deps'}:{run_root / 'scripts'}",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    command = [
        str(python),
        str(runner),
        "--run-root",
        str(run_root),
        "--arm",
        args.arm,
        "--resume",
        "--local-ollama-host",
        host,
        "--model",
        args.model,
        "--expected-digest",
        args.expected_digest,
    ]
    if args.smoke is not None:
        command.extend(("--smoke", str(args.smoke)))
    # Keep an explicit reference through argument construction; the descriptor
    # itself is inherited by execve and released automatically when the runner
    # terminates.
    assert lock_descriptor >= 0
    os.execve(str(python), command, environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ROUND8_LOCAL_JOB_REFUSED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error

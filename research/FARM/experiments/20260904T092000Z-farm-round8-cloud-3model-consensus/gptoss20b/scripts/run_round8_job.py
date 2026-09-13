#!/usr/bin/env python3
"""Single-arm exec wrapper used inside one detached Round8 tmux session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
)


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _existing_live_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        if pid <= 1:
            return None
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, OSError, ValueError, ProcessLookupError):
        return None


def _atomic_pid(path: Path, pid: int) -> None:
    temporary = path.with_name(f".{path.name}.{pid}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, f"{pid}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--slot", type=int, choices=range(4), required=True)
    parser.add_argument("--resume", action="store_true", required=True)
    parser.add_argument("--smoke", type=int, default=None)
    args = parser.parse_args(argv)
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be positive")
    if ARMS[args.slot] != args.arm:
        parser.error("arm/slot mapping is not the frozen Round8 mapping")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    env_file = args.env_file.resolve()
    python = args.python.resolve()
    runner = run_root / "scripts" / "run_round8.py"
    if not runner.is_file():
        raise FileNotFoundError(f"missing Round8 runner: {runner}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Python is missing or not executable: {python}")
    if not env_file.is_file() or _inside(env_file, run_root):
        raise ValueError("the required .env must exist outside the run artifacts")
    if not (run_root / ".deps" / "dspy" / "__init__.py").is_file():
        raise FileNotFoundError("run-local .deps does not contain DSPy")

    paths = {
        "logs": run_root / "logs",
        "pids": run_root / "pids",
        "progress": run_root / "progress",
        "records": run_root / "records",
        "results": run_root / "results",
        "attempts": run_root / "attempts",
    }
    for directory in paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    result_path = paths["results"] / f"{args.arm}.json"
    if result_path.exists():
        try:
            json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"malformed existing result: {result_path}") from error
        raise FileExistsError(f"refusing to rerun completed arm: {args.arm}")
    pid_path = paths["pids"] / f"{args.arm}.pid"
    live_pid = _existing_live_pid(pid_path)
    if live_pid is not None and live_pid != os.getpid():
        raise RuntimeError(f"another process owns arm {args.arm}: pid={live_pid}")

    os.umask(0o077)
    _atomic_pid(pid_path, os.getpid())
    log_path = paths["logs"] / f"{args.arm}.log"
    log_descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(log_descriptor, sys.stdout.fileno())
    os.dup2(log_descriptor, sys.stderr.fileno())
    if log_descriptor not in {sys.stdout.fileno(), sys.stderr.fileno()}:
        os.close(log_descriptor)
    print(
        f"START {datetime.now(timezone.utc).isoformat()} arm={args.arm} "
        f"pid={os.getpid()} process_slot={args.slot}",
        flush=True,
    )
    print(
        "INFO Ollama Cloud performs inference; CUDA_VISIBLE_DEVICES is only a "
        "process-isolation label and this wrapper does not load a V100 model.",
        flush=True,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.slot),
            "FARM_ROUND8_PROCESS_SLOT": str(args.slot),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "DSP_CACHEBOOL": "false",
            "PYTHONPATH": f"{run_root / '.deps'}:{run_root / 'scripts'}",
        }
    )
    command = [
        str(python),
        str(runner),
        "--run-root",
        str(run_root),
        "--env-file",
        str(env_file),
        "--arm",
        args.arm,
        "--resume",
    ]
    if args.smoke is not None:
        command.extend(("--smoke", str(args.smoke)))
    os.execve(str(python), command, environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ROUND8_JOB_REFUSED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error

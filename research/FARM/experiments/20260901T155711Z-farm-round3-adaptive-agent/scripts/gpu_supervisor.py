#!/usr/bin/env python3
"""Bounded restart supervisor for resumable cross-encoder jobs."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from experiment_core import write_json


def restart_delay(attempt: int) -> int:
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    return min(300, 30 * (2 ** min(attempt, 4)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--max-restarts", type=int, default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("supervisor command is empty")
    if args.max_restarts < 0:
        raise ValueError("max restarts must be non-negative")

    for attempt in range(args.max_restarts + 1):
        if args.final_output.is_file():
            write_json(args.status, {
                "phase": "complete", "attempt": attempt,
                "output": str(args.final_output),
            })
            return
        write_json(args.status, {"phase": "running", "attempt": attempt})
        result = subprocess.run(command, check=False)
        if result.returncode == 0 and args.final_output.is_file():
            write_json(args.status, {
                "phase": "complete", "attempt": attempt,
                "output": str(args.final_output),
            })
            return
        if attempt == args.max_restarts:
            break
        delay = restart_delay(attempt)
        write_json(args.status, {
            "phase": "backoff", "attempt": attempt,
            "returncode": result.returncode, "next_delay_seconds": delay,
        })
        time.sleep(delay)
    write_json(args.status, {"phase": "failed", "attempts": args.max_restarts + 1})
    raise SystemExit(1)


if __name__ == "__main__":
    main()

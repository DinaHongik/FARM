#!/usr/bin/env python3
"""Bounded restart supervisor for one resumable round-four arm."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from run_round4 import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--work-file", type=Path, required=True)
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise ValueError("supervisor command is empty")
    for attempt in range(args.max_restarts + 1):
        if args.final_output.is_file():
            write_json_atomic(args.status, {
                "phase": "complete", "restart_attempt": attempt,
                "output": str(args.final_output),
            })
            return
        current = list(command)
        if args.work_file.is_file() and "--resume" not in current:
            current.append("--resume")
        write_json_atomic(args.status, {
            "phase": "running", "restart_attempt": attempt,
            "resuming": args.work_file.is_file(),
        })
        completed = subprocess.run(current, check=False)
        if completed.returncode == 0 and args.final_output.is_file():
            write_json_atomic(args.status, {
                "phase": "complete", "restart_attempt": attempt,
                "output": str(args.final_output),
            })
            return
        if attempt == args.max_restarts:
            break
        delay = min(300, 15 * (2 ** min(attempt, 4)))
        write_json_atomic(args.status, {
            "phase": "backoff", "restart_attempt": attempt,
            "returncode": completed.returncode, "next_delay_seconds": delay,
        })
        time.sleep(delay)
    write_json_atomic(args.status, {
        "phase": "failed", "restart_attempts": args.max_restarts + 1,
    })
    raise SystemExit(1)


if __name__ == "__main__":
    main()

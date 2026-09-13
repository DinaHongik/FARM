#!/usr/bin/env python3
"""Print a compact, machine-readable round-five launch snapshot."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/raid/session/aicontents/farm/experiments/20260902T042236Z-farm-round5-selective-verifier")


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()


def main() -> None:
    sessions = command("tmux", "list-sessions", "-F", "#{session_name}:#{session_pid}:#{session_dead}").splitlines()
    wanted = [line for line in sessions if line.startswith("farm-r5-")]
    gpu_rows = command(
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    processes = command(
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ).splitlines()
    progress = {}
    for path in sorted((ROOT / "manifests").glob("*.json")):
        try:
            progress[path.name] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            progress[path.name] = {"unreadable": True}
    logs = {}
    for path in sorted((ROOT / "logs").glob("*.log")):
        logs[path.name] = {
            "bytes": path.stat().st_size,
            "tail": path.read_text(errors="replace").splitlines()[-5:],
        }
    snapshot = {
        "sessions": wanted,
        "gpus": gpu_rows[:5],
        "compute_processes": processes,
        "progress": progress,
        "logs": logs,
    }
    output = ROOT / "manifests" / "launch-verification-latest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


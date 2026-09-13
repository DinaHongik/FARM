#!/usr/bin/env python3
"""Write and print a compact Round6 launch snapshot."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path("/raid/session/aicontents/farm/experiments/20260904T043613Z-farm-round6-safe-agent")


def command(*args: str, check: bool = True) -> str:
    return subprocess.run(args, check=check, text=True, capture_output=True).stdout.strip()


def main() -> None:
    sessions = command("tmux", "list-sessions", "-F", "#{session_name}:#{session_pid}:#{session_dead}", check=False).splitlines()
    progress, logs, wal = {}, {}, {}
    for path in sorted((ROOT / "manifests").glob("*.json")):
        try:
            progress[path.name] = json.loads(path.read_text())
        except Exception:
            progress[path.name] = {"unreadable": True}
    for path in sorted((ROOT / "logs").glob("*.log")):
        logs[path.name] = {"bytes": path.stat().st_size, "tail": path.read_text(errors="replace").splitlines()[-4:]}
    for path in sorted((ROOT / "attempts").glob("*.jsonl")):
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        wal[path.name] = {
            "starts": sum(row.get("event") == "request_started" for row in events),
            "finishes": sum(row.get("event") == "request_finished" for row in events),
        }
    snapshot = {
        "sessions": [line for line in sessions if line.startswith("farm-r6-")],
        "gpus": command("nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits").splitlines()[:5],
        "compute_processes": command("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits", check=False).splitlines(),
        "progress": progress, "logs": logs, "wal": wal,
    }
    target = ROOT / "manifests/launch-verification-latest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, target)
    print(json.dumps(snapshot, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch-matrix validation and advancing-process health snapshots."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from experiment_core import read_json, write_json


def validate_launch_matrix(matrix: list[dict]) -> set[int]:
    if not matrix:
        raise ValueError("launch matrix is empty")
    ids = [row.get("id") for row in matrix]
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("experiment IDs must be unique non-empty strings")
    gpus = [row.get("gpu") for row in matrix]
    if any(not isinstance(value, int) or value < 0 for value in gpus):
        raise ValueError("GPU IDs must be non-negative integers")
    if len(set(gpus)) != len(gpus):
        raise ValueError("each concurrent GPU experiment needs a unique device")
    return set(gpus)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def gpu_snapshot() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, name, memory, utilization = [item.strip() for item in line.split(",", 3)]
        rows.append({
            "index": int(index), "name": name, "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
        })
    return rows


def snapshot(run_root: Path, matrix_path: Path) -> dict:
    matrix = read_json(matrix_path)["gpu_experiments"]
    validate_launch_matrix(matrix)
    jobs = []
    for item in matrix:
        pid_path = run_root / "pids" / f"{item['id']}.pid"
        pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
        progress_path = run_root / "manifests" / f"{item['id']}.progress.json"
        progress = read_json(progress_path) if progress_path.is_file() else None
        jobs.append({
            "id": item["id"], "gpu": item["gpu"], "pid": pid,
            "alive": bool(pid and process_alive(pid)), "progress": progress,
        })
    return {"gpu_experiments": jobs, "gpus": gpu_snapshot()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = snapshot(args.run_root.resolve(), args.matrix.resolve())
    if args.output:
        write_json(args.output.resolve(), value)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

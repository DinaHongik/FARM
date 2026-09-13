#!/usr/bin/env python3
"""Inspect and optionally persist a reviewer-grade health check."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import CONFIG_NAMES, DATASET_ID, config_from_path, read_json, utc_now, write_json  # noqa: E402
PHASE_ORDER = {
    "not_started": 0, "preflight": 1, "trigger_training": 2,
    "action_training": 3, "dev_evaluation": 4, "complete": 5,
    "failed": -1,
}
ERROR_RE = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|non-finite|\bloss[=: ]+nan\b", re.I)


def command_ok(command: list[str]) -> bool:
    return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def gpu_status(index: int) -> dict:
    result = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        return {"query_error": result.stderr.strip()}
    values = [value.strip() for value in result.stdout.strip().split(",")]
    apps = subprocess.run(
        [
            "nvidia-smi", "-i", str(index),
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=False,
    )
    compute_apps = []
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3 and fields[0].isdigit():
                compute_apps.append({
                    "pid": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "used_memory_mib": int(fields[2]),
                })
    return {
        "uuid": values[0],
        "utilization_percent": int(values[1]),
        "memory_used_mib": int(values[2]),
        "memory_total_mib": int(values[3]),
        "compute_apps": compute_apps,
    }


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_descends_from(pid: int, ancestor: int) -> bool:
    """Verify Linux process ownership without relying on aggregate GPU use."""
    if pid <= 0 or ancestor <= 0:
        return False
    current = pid
    visited = set()
    while current > 1 and current not in visited:
        if current == ancestor:
            return True
        visited.add(current)
        try:
            lines = Path(f"/proc/{current}/status").read_text().splitlines()
        except OSError:
            return False
        parent = next((line for line in lines if line.startswith("PPid:")), "")
        try:
            current = int(parent.split()[1])
        except (IndexError, ValueError):
            return False
    return current == ancestor


def latest_progress(run_root: Path, exp: str, phase: str) -> dict:
    kind = "action" if phase in {"action_training", "dev_evaluation", "complete"} else "trigger"
    path = run_root / "manifests" / exp / f"train_{kind}.progress.json"
    if path.is_file():
        return read_json(path)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--record-health-check", type=int, choices=(1, 2))
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    global_status = read_json(run_root / "STATUS.json")
    report = {
        "checked_at": utc_now(),
        "run_id": global_status["run_id"],
        "dataset_id": global_status["dataset_id"],
        "check_number": args.record_health_check,
        "experiments": {},
    }
    if report["dataset_id"] != DATASET_ID:
        raise RuntimeError("status file has wrong Dataset v2 ID")

    previous = None
    previous_path = run_root / "manifests" / "health_check_1.json"
    if args.record_health_check == 2:
        if not previous_path.is_file():
            raise RuntimeError("health check 1 must be recorded before check 2")
        previous = read_json(previous_path)

    all_healthy = True
    for config_name in CONFIG_NAMES:
        config = config_from_path(run_root / "configs" / config_name)
        exp = config["experiment_id"]
        status = global_status["experiments"][exp]
        phase = status.get("phase", "not_started")
        pipeline_manifest_path = run_root / "manifests" / exp / "pipeline.json"
        pipeline_manifest = read_json(pipeline_manifest_path) if pipeline_manifest_path.is_file() else {}
        pid_record_path = run_root / "pids" / f"{exp}.pid"
        pid_record = read_json(pid_record_path) if pid_record_path.is_file() else {}
        pid = int(pid_record.get("pid", status.get("pid", 0)) or 0)
        session_alive = command_ok(["tmux", "has-session", "-t", config["session"]])
        progress = latest_progress(run_root, exp, phase)
        log_path = run_root / "logs" / exp / "pipeline.log"
        tail = ""
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-200:])
        errors = sorted(set(match.group(0) for match in ERROR_RE.finditer(tail)))
        latest_loss = progress.get("loss", progress.get("training_loss"))
        finite_loss = isinstance(latest_loss, (int, float)) and math.isfinite(float(latest_loss))
        gpu = gpu_status(config["gpu"])
        compute_pids = [int(app["pid"]) for app in gpu.get("compute_apps", [])]
        owned_gpu_pids = [candidate for candidate in compute_pids if pid_descends_from(candidate, pid)]
        progress_pid = int(progress.get("pid", 0) or 0)
        progress_pid_on_expected_gpu = (
            progress_pid in compute_pids and pid_descends_from(progress_pid, pid)
        )
        manifest_gpu_uuid = pipeline_manifest.get("gpu", {}).get("uuid")
        gpu_uuid_matches_manifest = bool(manifest_gpu_uuid) and manifest_gpu_uuid == gpu.get("uuid")
        active_or_complete = (session_alive and pid_alive(pid)) or phase == "complete"
        manifest_dataset_ok = pipeline_manifest.get("dataset_id") == DATASET_ID
        if phase in {"trigger_training", "action_training"}:
            allocation_verified = progress_pid_on_expected_gpu
        elif phase == "dev_evaluation":
            allocation_verified = bool(owned_gpu_pids)
        elif phase == "complete":
            allocation_verified = True
        else:
            allocation_verified = False
        healthy = (
            active_or_complete
            and manifest_dataset_ok
            and finite_loss
            and not errors
            and allocation_verified
            and (gpu_uuid_matches_manifest or phase == "complete")
        )
        advanced = None
        if previous is not None:
            old = previous["experiments"][exp]
            old_phase = old["phase"]
            advanced = (
                PHASE_ORDER.get(phase, -1) > PHASE_ORDER.get(old_phase, -1)
                or (
                    phase == old_phase
                    and int(progress.get("step", 0)) > int(old.get("step", 0))
                )
            )
            healthy = healthy and advanced
        row = {
            "experiment_id": exp,
            "gpu": config["gpu"],
            "session": config["session"],
            "session_alive": session_alive,
            "pid": pid,
            "pid_alive": pid_alive(pid),
            "progress_pid": progress_pid,
            "phase": phase,
            "step": int(progress.get("step", 0)),
            "max_steps": progress.get("max_steps"),
            "latest_loss": latest_loss,
            "finite_loss": finite_loss,
            "dataset_id": pipeline_manifest.get("dataset_id"),
            "gpu_status": gpu,
            "owned_gpu_pids": owned_gpu_pids,
            "progress_pid_on_expected_gpu": progress_pid_on_expected_gpu,
            "gpu_uuid_matches_manifest": gpu_uuid_matches_manifest,
            "allocation_verified": allocation_verified,
            "log_path": str(log_path),
            "checkpoint_root": str(run_root / "checkpoints" / exp),
            "errors": errors,
            "advanced_since_check_1": advanced,
            "healthy": healthy,
        }
        report["experiments"][exp] = row
        all_healthy = all_healthy and healthy
    report["all_healthy"] = all_healthy
    if args.record_health_check:
        output = run_root / "manifests" / f"health_check_{args.record_health_check}.json"
        write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.record_health_check and not all_healthy:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

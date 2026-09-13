#!/usr/bin/env python3
"""Twice verify local Granite servers, Round8 sessions, and progress read-only."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import launch_round8_local as local_ops
import verify_round8 as common


def _tmux(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def inspect_experiment(
    run_root: Path, slot: local_ops.Slot, *, smoke: int | None = None
) -> dict[str, Any]:
    artifacts = local_ops.local_artifact_root(run_root, smoke)
    expected_rows, _ = local_ops.selection_identity(smoke)
    session = slot.experiment_session_for(smoke)
    session_active = _tmux(session)
    pid_path = artifacts / "pids" / f"{slot.arm}.pid"
    pid = local_ops._read_pid(pid_path)
    alive = local_ops._pid_alive(pid)
    command = local_ops._process_command(pid) if alive and pid is not None else ""
    process_matches = all(
        token in command
        for token in (str(run_root), "run_round8.py", slot.arm, slot.host, local_ops.MODEL)
    )
    progress, progress_error = common._json_file(
        artifacts / "progress" / f"{slot.arm}.json"
    )
    result, result_error = common._json_file(artifacts / "results" / f"{slot.arm}.json")
    launch_binding, launch_binding_error = common._json_file(
        artifacts / "state" / f"{slot.arm}.launch.json"
    )
    expected_launch_binding = local_ops.local_launch_binding(
        run_root, slot, smoke=smoke
    )
    launch_binding_matches = (
        launch_binding is not None and dict(launch_binding) == expected_launch_binding
    )
    progress_state = common._state(progress)
    result_state = common._state(result)
    records_path = artifacts / "records" / f"{slot.arm}.jsonl"
    record_lines = common._line_count(records_path)
    progress_binding_mismatches = (
        local_ops.runner_binding_mismatches(
            run_root, slot, progress.get("binding"), smoke=smoke
        )
        if progress is not None
        else ()
    )
    result_binding_mismatches = (
        local_ops.runner_binding_mismatches(
            run_root, slot, result.get("binding"), smoke=smoke
        )
        if result is not None
        else ()
    )
    progress_count_valid = True
    if progress is not None:
        completed_rows = progress.get("completed_rows")
        progress_count_valid = (
            isinstance(completed_rows, int)
            and not isinstance(completed_rows, bool)
            and completed_rows == record_lines
            and progress.get("target_rows") == expected_rows
        )
    result_integrity_valid = True
    completion_validation_error: str | None = None
    if result is not None:
        try:
            local_ops.validate_completed_artifacts(run_root, slot, smoke=smoke)
        except (OSError, RuntimeError) as error:
            result_integrity_valid = False
            completion_validation_error = str(error)
    log_path = artifacts / "logs" / f"{slot.arm}.log"
    try:
        stat = log_path.stat()
        log_bytes, log_mtime_ns = stat.st_size, stat.st_mtime_ns
    except OSError:
        log_bytes, log_mtime_ns = 0, 0

    if (
        progress_error
        or result_error
        or launch_binding_error
        or not launch_binding_matches
        or progress_binding_mismatches
        or result_binding_mismatches
        or not progress_count_valid
        or not result_integrity_valid
        or (pid_path.exists() and pid is None)
    ):
        status = "ambiguous"
    elif result is not None and (session_active or alive):
        status = "ambiguous"
    elif result_state in {"failed", "error", "aborted"}:
        status = "failed"
    elif result is not None:
        status = "complete"
    elif progress_state in {"complete", "completed", "success", "succeeded"}:
        status = "ambiguous"
    elif progress_state in {"failed", "error", "aborted"} and not session_active and not alive:
        status = "failed"
    elif session_active and alive and process_matches:
        status = "running"
    elif session_active and not alive:
        status = "starting"
    elif alive:
        status = "ambiguous"
    elif any((progress is not None, records_path.exists(), log_bytes, pid_path.exists())):
        status = "stale-partial"
    else:
        status = "absent"
    return {
        "arm": slot.arm,
        "gpu": slot.gpu,
        "host": slot.host,
        "session": session,
        "namespace": artifacts.name,
        "status": status,
        "tmux_session": session_active,
        "pid": pid,
        "process_alive": alive,
        "process_matches_runner": process_matches,
        "progress_state": progress_state,
        "completed_cases": common._number(
            progress,
            (
                "completed_rows",
                "completed_cases",
                "cases_completed",
                "processed",
                "records_written",
                "next_index",
            ),
        ),
        "total_cases": common._number(
            progress, ("target_rows", "total_cases", "cases_total", "total")
        ),
        "record_lines": record_lines,
        "log_bytes": log_bytes,
        "log_mtime_ns": log_mtime_ns,
        "progress_error": progress_error,
        "result_error": result_error,
        "result_present": result is not None,
        "launch_binding_matches": launch_binding_matches,
        "launch_binding_error": launch_binding_error,
        "progress_binding_mismatches": list(progress_binding_mismatches),
        "progress_count_valid": progress_count_valid,
        "result_integrity_valid": result_integrity_valid,
        "completion_validation_error": completion_validation_error,
    }


def inspect_server(
    run_root: Path, slot: local_ops.Slot, models: Path = local_ops.DEFAULT_MODELS
) -> dict[str, Any]:
    probe = local_ops.probe_ollama(slot.port)
    if slot.port == 11434:
        listener_pid: int | None = None
        gpu_visibility: str | None = None
        models_match: bool | None = None
        environment_error: str | None = None
        try:
            listener_pid = local_ops.listening_pid(slot.port)
            environment = local_ops._process_environment(listener_pid)
            gpu_visibility = environment.get("CUDA_VISIBLE_DEVICES")
            observed_models = environment.get("OLLAMA_MODELS")
            models_match = bool(
                observed_models
                and Path(observed_models).resolve() == models.resolve()
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            environment_error = str(error)
        return {
            "port": slot.port,
            "gpu": 0,
            "status": "ready" if probe.ready else "unhealthy",
            "ownership": "preexisting_reused",
            "api_ok": probe.api_ok,
            "model": local_ops.MODEL,
            "digest_exact": probe.digest_ok,
            "observed_digest": probe.observed_digest,
            "tmux_session": None,
            "pid": listener_pid,
            "process_alive": (
                local_ops._pid_alive(listener_pid)
                if listener_pid is not None
                else None
            ),
            "gpu_environment_matches": gpu_visibility == "0",
            "gpu_visibility": gpu_visibility,
            "physical_gpu_isolation_source": "post_smoke_compute_telemetry",
            "models_environment_matches": models_match,
            "error": environment_error or probe.error,
        }
    session_active = _tmux(slot.server_session)
    pid_path = local_ops.local_server_root(run_root) / f"{slot.port}.pid"
    pid = local_ops._read_pid(pid_path)
    alive = local_ops._pid_alive(pid)
    gpu_matches = False
    models_matches = False
    host_matches = False
    environment_error: str | None = None
    if alive and pid is not None:
        try:
            environment = local_ops._process_environment(pid)
            gpu_matches = environment.get("CUDA_VISIBLE_DEVICES") == str(slot.gpu)
            observed_models = environment.get("OLLAMA_MODELS")
            models_matches = bool(
                observed_models
                and Path(observed_models).resolve() == models.resolve()
            )
            host_matches = environment.get("OLLAMA_HOST") in {
                f"127.0.0.1:{slot.port}",
                f"http://127.0.0.1:{slot.port}",
            }
        except RuntimeError as error:
            environment_error = str(error)
    ready = (
        probe.ready
        and session_active
        and alive
        and gpu_matches
        and models_matches
        and host_matches
        and environment_error is None
    )
    return {
        "port": slot.port,
        "gpu": slot.gpu,
        "status": "ready" if ready else "ambiguous",
        "ownership": "round8_managed",
        "api_ok": probe.api_ok,
        "model": local_ops.MODEL,
        "digest_exact": probe.digest_ok,
        "observed_digest": probe.observed_digest,
        "tmux_session": session_active,
        "pid": pid,
        "process_alive": alive,
        "gpu_environment_matches": gpu_matches,
        "models_environment_matches": models_matches,
        "host_environment_matches": host_matches,
        "error": environment_error or probe.error,
    }


def collect_snapshot(
    run_root: Path,
    models: Path = local_ops.DEFAULT_MODELS,
    *,
    smoke: int | None = None,
) -> dict[str, Any]:
    smoke_by_arm = {
        slot.arm: local_ops.smoke_size_for_arm(slot.arm, smoke)
        for slot in local_ops.SLOTS
    }
    try:
        gpu_isolation = local_ops.gpu_isolation_snapshot(run_root)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        gpu_isolation = {
            "verified": False,
            "errors": [f"GPU telemetry unavailable: {error}"],
            "conflicts": [f"GPU telemetry unavailable: {error}"],
            "servers": [],
            "unexpected_compute_apps": [],
        }
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "transport": "local_loopback",
        "data_left_dgx": False,
        "model": local_ops.MODEL,
        "expected_digest": local_ops.MODEL_DIGEST,
        "namespaces": {
            arm: local_ops.local_artifact_root(run_root, selected_smoke).name
            for arm, selected_smoke in smoke_by_arm.items()
        },
        "ollama_models": str(models),
        "gpu_isolation": gpu_isolation,
        "servers": [inspect_server(run_root, slot, models) for slot in local_ops.SLOTS],
        "arms": [
            inspect_experiment(run_root, slot, smoke=smoke_by_arm[slot.arm])
            for slot in local_ops.SLOTS
        ],
    }


def assess(
    snapshots: Sequence[Mapping[str, Any]], expectation: str
) -> tuple[bool, tuple[str, ...]]:
    experiment_ok, experiment_errors = common.assess_snapshots(
        snapshots, expectation, arms=local_ops.ARMS
    )
    errors = list(experiment_errors)
    if len(snapshots) < 2:
        return False, tuple(errors)
    for index, snapshot in enumerate(snapshots):
        servers = snapshot.get("servers")
        if not isinstance(servers, list) or len(servers) != len(local_ops.SLOTS):
            errors.append(
                f"sample {index}: missing {len(local_ops.SLOTS)}-server inventory"
            )
            continue
        for server in servers:
            if not isinstance(server, Mapping) or server.get("status") != "ready":
                errors.append(f"sample {index}: local Ollama server not ready: {server}")
            elif server.get("observed_digest") != local_ops.MODEL_DIGEST:
                errors.append(f"sample {index}: local Ollama digest changed")
        isolation = snapshot.get("gpu_isolation")
        if not isinstance(isolation, Mapping):
            errors.append(f"sample {index}: GPU isolation telemetry missing")
        elif isolation.get("conflicts"):
            errors.append(
                f"sample {index}: GPU isolation conflict: {isolation.get('conflicts')}"
            )
    if expectation == "complete":
        final_isolation = snapshots[-1].get("gpu_isolation")
        if not isinstance(final_isolation, Mapping) or not final_isolation.get("verified"):
            errors.append("final sample: all five loaded GPU mappings are not proven")
    return experiment_ok and not errors, tuple(errors)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ollama-models", type=Path, default=local_ops.DEFAULT_MODELS)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--initial-delay", type=float, default=0.0)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument(
        "--smoke",
        type=int,
        choices=(local_ops.SMOKE_ROWS,),
        help="verify the per-arm smoke gate (2 rows; 5 routed rows for consensus)",
    )
    parser.add_argument(
        "--expect",
        choices=("running", "running-or-complete", "complete"),
        default="running-or-complete",
    )
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if not args.status_only and args.samples < 2:
        parser.error("verification requires at least two samples")
    if args.initial_delay < 0 or args.interval <= 0:
        parser.error("verification timing is invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    models = args.ollama_models.resolve()
    if args.initial_delay:
        time.sleep(args.initial_delay)
    count = 1 if args.status_only else args.samples
    snapshots: list[dict[str, Any]] = []
    for index in range(count):
        snapshot = collect_snapshot(run_root, models, smoke=args.smoke)
        snapshots.append(snapshot)
        print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True), flush=True)
        if index + 1 < count:
            time.sleep(args.interval)
    if args.status_only:
        return 0
    ok, errors = assess(snapshots, args.expect)
    report = {
        "schema_version": "farm_round8_local_launch_verification_v1",
        "verified": ok,
        "expectation": args.expect,
        "observation_count": len(snapshots),
        "namespaces": snapshots[-1].get("namespaces", {}),
        "errors": list(errors),
        "snapshots": snapshots,
        "safety": {
            "loopback_only": True,
            "data_left_dgx": False,
            "processes_modified": False,
            "sessions_killed": False,
            "survives_ssh_or_laptop_disconnect": ok,
        },
    }
    if not args.no_write:
        status = (
            local_ops.local_artifact_root(run_root) / "status"
            if args.smoke is None
            else local_ops.local_model_root(run_root)
            / "status"
            / f"smoke-gate-{args.smoke}"
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        common._atomic_json(status / f"verification-{stamp}.json", report)
        common._atomic_json(status / "verification-latest.json", report)
    print(json.dumps({"verified": ok, "errors": list(errors)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROUND8_LOCAL_VERIFY_FAILED: {error}", file=sys.stderr)
        raise SystemExit(2) from error

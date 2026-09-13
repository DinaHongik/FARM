#!/usr/bin/env python3
"""Observe Round8 tmux/process/progress state twice without changing jobs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence


ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
)


def session_name(arm: str) -> str:
    return f"farm-r8-{arm.replace('_', '-')}"


def _tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        return pid if pid > 1 else None
    except (FileNotFoundError, OSError, ValueError):
        return None


def _process(pid: int | None, run_root: Path, arm: str) -> tuple[bool, bool, str]:
    if pid is None:
        return False, False, ""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False, False, ""
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        command = ""
    belongs = str(run_root) in command and arm in command and "run_round8.py" in command
    return True, belongs, command


def _json_file(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"
    if not isinstance(value, Mapping):
        return None, "top-level JSON value is not an object"
    return value, None


def _number(payload: Mapping[str, Any] | None, keys: Sequence[str]) -> int | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    for container in ("progress", "counts", "metrics"):
        nested = payload.get(container)
        if isinstance(nested, Mapping):
            found = _number(nested, keys)
            if found is not None:
                return found
    return None


def _state(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    for key in ("state", "status", "phase"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.casefold()
    return None


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def inspect_arm(
    run_root: Path,
    arm: str,
    *,
    tmux_probe: Callable[[str], bool] = _tmux_has_session,
    process_probe: Callable[[int | None, Path, str], tuple[bool, bool, str]] = _process,
) -> dict[str, Any]:
    session = session_name(arm)
    session_active = tmux_probe(session)
    pid_path = run_root / "pids" / f"{arm}.pid"
    pid_file_present = pid_path.exists()
    pid = _read_pid(pid_path)
    process_alive, process_matches, command = process_probe(pid, run_root, arm)
    progress, progress_error = _json_file(run_root / "progress" / f"{arm}.json")
    result, result_error = _json_file(run_root / "results" / f"{arm}.json")
    log_path = run_root / "logs" / f"{arm}.log"
    records_path = run_root / "records" / f"{arm}.jsonl"
    try:
        log_size = log_path.stat().st_size
        log_mtime_ns = log_path.stat().st_mtime_ns
    except OSError:
        log_size = 0
        log_mtime_ns = 0

    progress_state = _state(progress)
    result_state = _state(result)
    completed_cases = _number(
        progress,
        (
            "completed_rows",
            "completed_cases",
            "cases_completed",
            "processed",
            "records_written",
            "next_index",
        ),
    )
    total_cases = _number(progress, ("target_rows", "total_cases", "cases_total", "total"))
    if result_error is not None or progress_error is not None or (pid_file_present and pid is None):
        status = "ambiguous"
    elif result is not None and (session_active or process_alive):
        status = "ambiguous"
    elif result_state in {"failed", "error", "aborted"}:
        status = "failed"
    elif result is not None:
        status = "complete"
    elif progress_state in {"complete", "completed", "success", "succeeded"}:
        status = "ambiguous"
    elif progress_state in {"failed", "error", "aborted"} and not session_active and not process_alive:
        status = "failed"
    elif session_active and process_alive and process_matches:
        status = "running"
    elif session_active and not process_alive:
        status = "starting"
    elif process_alive:
        status = "ambiguous"
    elif any((progress is not None, log_size, records_path.exists(), pid is not None)):
        status = "stale-partial"
    else:
        status = "absent"

    # A full command is useful interactively but can contain paths or CLI data.
    # Persist only whether it matched the frozen runner identity.
    return {
        "arm": arm,
        "session": session,
        "status": status,
        "tmux_session": session_active,
        "pid": pid,
        "pid_file_present": pid_file_present,
        "process_alive": process_alive,
        "process_matches_runner": process_matches,
        "progress_state": progress_state,
        "completed_cases": completed_cases,
        "total_cases": total_cases,
        "record_lines": _line_count(records_path),
        "log_bytes": log_size,
        "log_mtime_ns": log_mtime_ns,
        "progress_error": progress_error,
        "result_error": result_error,
        "result_present": result is not None,
        "command_observed": bool(command),
    }


def collect_snapshot(
    run_root: Path,
    *,
    tmux_probe: Callable[[str], bool] = _tmux_has_session,
    process_probe: Callable[[int | None, Path, str], tuple[bool, bool, str]] = _process,
) -> dict[str, Any]:
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "ollama_cloud_inference": True,
        "v100_compute_consumed_by_these_arms": False,
        "arms": [
            inspect_arm(
                run_root,
                arm,
                tmux_probe=tmux_probe,
                process_probe=process_probe,
            )
            for arm in ARMS
        ],
    }


def assess_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    expectation: Literal["running", "running-or-complete", "complete"],
    *,
    arms: Sequence[str] = ARMS,
) -> tuple[bool, tuple[str, ...]]:
    if len(snapshots) < 2:
        return False, ("verification requires at least two observations",)
    errors: list[str] = []
    by_sample = [
        {str(arm["arm"]): arm for arm in snapshot.get("arms", [])}
        for snapshot in snapshots
    ]
    for arm in arms:
        observations = [sample.get(arm) for sample in by_sample]
        if any(item is None for item in observations):
            errors.append(f"{arm}: missing from a verification observation")
            continue
        states = [str(item["status"]) for item in observations if item is not None]
        terminal = states[-1]
        if expectation == "running":
            if terminal != "running" or any(state not in {"starting", "running"} for state in states):
                errors.append(f"{arm}: expected running, observed {states}")
        elif expectation == "complete":
            if terminal != "complete" or any(
                state not in {"starting", "running", "complete"} for state in states
            ):
                errors.append(f"{arm}: expected complete, observed {states}")
        elif terminal not in {"running", "complete"} or any(
            state not in {"starting", "running", "complete"} for state in states
        ):
            errors.append(f"{arm}: expected running-or-complete, observed {states}")

        final_observation = observations[-1]
        if final_observation is not None and terminal == "running":
            record_lines = final_observation.get("record_lines")
            log_bytes = final_observation.get("log_bytes")
            if not (
                isinstance(final_observation.get("completed_cases"), int)
                or (isinstance(record_lines, int) and record_lines > 0)
                or (isinstance(log_bytes, int) and log_bytes > 0)
            ):
                errors.append(f"{arm}: runner is alive but exposes no progress/log evidence")

        for metric in ("completed_cases", "record_lines", "log_bytes"):
            values = [item.get(metric) for item in observations if item is not None]
            known = [int(value) for value in values if isinstance(value, int)]
            if any(later < earlier for earlier, later in zip(known, known[1:])):
                errors.append(f"{arm}: {metric} regressed across observations: {values}")
        for item in observations:
            if item is not None and item["status"] == "running":
                if not (
                    item["tmux_session"]
                    and item["process_alive"]
                    and item["process_matches_runner"]
                ):
                    errors.append(f"{arm}: running identity check failed")
    return not errors, tuple(errors)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--initial-delay", type=float, default=0.0)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument(
        "--expect",
        choices=("running", "running-or-complete", "complete"),
        default="running-or-complete",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="print one read-only snapshot; do not perform the two-sample verification",
    )
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    if not args.status_only and args.samples < 2:
        parser.error("verification requires --samples >= 2")
    if args.initial_delay < 0 or args.interval <= 0:
        parser.error("delays must be non-negative and interval must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    if args.initial_delay:
        time.sleep(args.initial_delay)
    sample_count = 1 if args.status_only else args.samples
    snapshots: list[dict[str, Any]] = []
    for index in range(sample_count):
        snapshot = collect_snapshot(run_root)
        snapshots.append(snapshot)
        print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True), flush=True)
        if index + 1 < sample_count:
            time.sleep(args.interval)

    if args.status_only:
        return 0
    ok, errors = assess_snapshots(snapshots, args.expect)
    report = {
        "schema_version": "farm_round8_launch_verification_v1",
        "verified": ok,
        "expectation": args.expect,
        "observation_count": len(snapshots),
        "errors": list(errors),
        "snapshots": snapshots,
        "safety": {
            "processes_modified": False,
            "sessions_killed": False,
            "survives_ssh_or_laptop_disconnect": ok,
            "ollama_cloud_inference": True,
            "v100_compute_consumed": False,
        },
    }
    if not args.no_write:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        status_root = run_root / "status"
        _atomic_json(status_root / f"verification-{stamp}.json", report)
        _atomic_json(status_root / "verification-latest.json", report)
    print(json.dumps({"verified": ok, "errors": list(errors)}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROUND8_VERIFY_FAILED: {error}", file=sys.stderr)
        raise SystemExit(2) from error

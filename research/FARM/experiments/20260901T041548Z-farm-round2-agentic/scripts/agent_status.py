#!/usr/bin/env python3
"""Inspect and persist progress evidence for detached Ollama comparisons."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import read_json, utc_now, write_json  # noqa: E402

JOBS = (
    ("gpt_oss_120b", "farm-r2-agent-gpt120"),
    ("gemma4_31b", "farm-r2-agent-gemma31"),
)
LLM_ARMS = ("one_shot_plain", "one_shot_schema", "reflect_agent", "role_agent", "tool_agent")
ERROR_RE = re.compile(r"Traceback|secret persistence|digest changed|candidate hash|RuntimeError", re.I)


def session_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def work_health(path: Path) -> dict:
    rows = []
    parse_error = None
    if path.is_file():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            parse_error = type(exc).__name__
    arm_health = {}
    for arm in LLM_ARMS:
        outcomes = [row.get("arms", {}).get(arm, {}) for row in rows]
        arm_health[arm] = {
            "successful_call_rows": sum(
                "error_type" not in item
                and int(item.get("calls", 0)) > 0
                and len(item.get("usage") or []) > 0
                for item in outcomes
            ),
            "protocol_valid_rows": sum(bool(item.get("protocol_valid")) for item in outcomes),
            "total_calls": sum(int(item.get("calls", 0)) for item in outcomes),
            "failures": sum("error_type" in item for item in outcomes),
        }
    successful_calls = bool(rows) and all(
        item["successful_call_rows"] == len(rows) and item["failures"] == 0
        for item in arm_health.values()
    )
    protocol_arms = [arm for arm, item in arm_health.items() if item["protocol_valid_rows"] > 0]
    protocol_exercised = (
        "reflect_agent" in protocol_arms
        and "tool_agent" in protocol_arms
        and len(protocol_arms) >= 3
    )
    return {
        "rows": len(rows), "parse_error": parse_error, "arms": arm_health,
        "all_rows_have_successful_llm_calls": successful_calls,
        "protocol_valid_arms": protocol_arms,
        "minimum_protocols_exercised": protocol_exercised,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--record-health-check", type=int, choices=(1, 2))
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    previous = None
    if args.record_health_check == 2:
        previous_path = run_root / "manifests/agent_health_check_1.json"
        if not previous_path.is_file():
            raise RuntimeError("agent health check 1 is required")
        previous = read_json(previous_path)
    report = {"checked_at": utc_now(), "check_number": args.record_health_check, "jobs": {}}
    all_healthy = True
    for slug, session in JOBS:
        progress_path = run_root / "manifests/agents" / f"{slug}.progress.json"
        output = run_root / "results" / f"agent_{slug}_dev300.json"
        log = run_root / "logs/agents" / f"{slug}.log"
        work = run_root / "results/agent_work" / f"{slug}.jsonl"
        progress = read_json(progress_path) if progress_path.is_file() else {}
        completed = int(progress.get("completed_rows", 0))
        target = int(progress.get("target_rows", 300))
        phase = progress.get("status", "starting")
        alive = session_alive(session)
        finished = phase == "completed" and output.is_file() and completed == target
        tail = ""
        if log.is_file():
            tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-100:])
        errors = sorted(set(match.group(0) for match in ERROR_RE.finditer(tail)))
        calls = work_health(work)
        advanced = None
        if previous is not None:
            old = previous["jobs"][slug]
            advanced = completed > int(old["completed_rows"]) or finished
        healthy = (
            (alive or finished)
            and completed > 0
            and calls["rows"] == completed
            and calls["parse_error"] is None
            and calls["all_rows_have_successful_llm_calls"]
            and calls["minimum_protocols_exercised"]
            and not errors
        )
        if previous is not None:
            healthy = healthy and bool(advanced)
        report["jobs"][slug] = {
            "session": session, "session_alive": alive, "phase": phase,
            "completed_rows": completed, "target_rows": target,
            "advanced_since_check_1": advanced, "output_exists": output.is_file(),
            "errors": errors, "healthy": healthy,
            "llm_call_health": calls,
        }
        all_healthy = all_healthy and healthy
    report["all_healthy"] = all_healthy
    if args.record_health_check:
        write_json(run_root / "manifests" / f"agent_health_check_{args.record_health_check}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.record_health_check and not all_healthy:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

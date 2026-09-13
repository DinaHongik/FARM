#!/usr/bin/env python3
"""All-or-none preflight and detached launch for the four Round8 arms.

Ollama Cloud performs the model inference.  ``CUDA_VISIBLE_DEVICES`` is set to
0..3 only as a stable process-isolation/diagnostic label; these jobs do not load
or run a local model on the V100s.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


RUN_ID = "20260904T074517Z-farm-round8-executable-agent"
ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
)
DEFAULT_PYTHON = Path("/raid/session/aicontents/farm/.venv/bin/python")


def session_name(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unknown Round8 arm: {arm}")
    return f"farm-r8-{arm.replace('_', '-')}"


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _read_pid(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
        pid = int(value)
        return pid if pid > 1 else None
    except (FileNotFoundError, OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def _tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _progress_state(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"malformed progress artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"progress artifact is not an object: {path}")
    for key in ("state", "status", "phase"):
        value = payload.get(key)
        if isinstance(value, str):
            return value.casefold()
    return None


@dataclass(frozen=True)
class PreflightArm:
    arm: str
    session: str
    resumable_partial: bool


def preflight_arm(run_root: Path, arm: str) -> PreflightArm:
    """Reject active, completed, malformed, and identity-ambiguous state."""

    name = session_name(arm)
    arm_result = run_root / "results" / f"{arm}.json"
    arm_progress = run_root / "progress" / f"{arm}.json"
    arm_pid = run_root / "pids" / f"{arm}.pid"
    if _tmux_has_session(name):
        raise RuntimeError(f"active tmux session already exists: {name}")
    if arm_result.exists():
        try:
            json.loads(arm_result.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"ambiguous malformed result for {arm}: {error}") from error
        raise RuntimeError(f"completed result already exists for {arm}; refusing resume")

    pid = _read_pid(arm_pid)
    if arm_pid.exists() and pid is None:
        raise RuntimeError(f"pid artifact for {arm} is malformed: {arm_pid}")
    if pid is not None and _pid_alive(pid):
        command = _process_command(pid)
        if str(run_root) in command and arm in command:
            raise RuntimeError(f"live Round8 process already exists for {arm}: pid={pid}")
        raise RuntimeError(f"pid file for {arm} points to an unrelated live process: pid={pid}")

    state = _progress_state(arm_progress)
    if state in {"complete", "completed", "success", "succeeded"}:
        raise RuntimeError(
            f"progress says {arm} is complete but no final result exists; state is ambiguous"
        )
    partial_paths = (
        arm_progress,
        run_root / "records" / f"{arm}.jsonl",
        run_root / "logs" / f"{arm}.log",
        arm_pid,
    )
    return PreflightArm(
        arm=arm,
        session=name,
        resumable_partial=any(path.exists() for path in partial_paths),
    )


def validate_runtime(run_root: Path, env_file: Path, python: Path) -> None:
    required = (
        run_root / "ROUND8_SCOPE.md",
        run_root / "EXPERIMENT_MATRIX.json",
        run_root / "scripts" / "run_round8.py",
        run_root / "scripts" / "run_round8_job.py",
        run_root / "scripts" / "verify_round8.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing Round8 launch prerequisites: {missing}")
    if run_root.name != RUN_ID:
        raise RuntimeError(f"unexpected Round8 run directory: {run_root}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"shared FARM Python is missing or not executable: {python}")
    if not env_file.is_file():
        raise RuntimeError(f"external environment file does not exist: {env_file}")
    if _inside(env_file, run_root):
        raise RuntimeError(".env must remain outside the Round8 artifact directory")
    dependency_root = run_root / ".deps"
    if not (dependency_root / "dspy" / "__init__.py").is_file():
        raise RuntimeError(
            f"isolated DSPy dependency is missing under {dependency_root}; run dependency setup first"
        )
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required for disconnect-safe detached execution")


def tmux_launch_command(
    *,
    run_root: Path,
    env_file: Path,
    python: Path,
    arm: str,
    slot: int,
    smoke: int | None,
) -> list[str]:
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        session_name(arm),
        "-c",
        str(run_root),
        str(python),
        str(run_root / "scripts" / "run_round8_job.py"),
        "--run-root",
        str(run_root),
        "--env-file",
        str(env_file),
        "--python",
        str(python),
        "--arm",
        arm,
        "--slot",
        str(slot),
        "--resume",
    ]
    if smoke is not None:
        command.extend(("--smoke", str(smoke)))
    return command


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    inferred_root = Path(__file__).resolve().parents[1]
    inferred_env = inferred_root.parents[1] / ".env"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=inferred_root)
    parser.add_argument("--env-file", type=Path, default=inferred_env)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--smoke", type=int, default=None, metavar="N")
    parser.add_argument("--verify-initial-delay", type=float, default=3.0)
    parser.add_argument("--verify-interval", type=float, default=15.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform preflight and print commands without launching sessions",
    )
    args = parser.parse_args(argv)
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be a positive case count")
    if args.verify_initial_delay < 0 or args.verify_interval <= 0:
        parser.error("verification delays must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    env_file = args.env_file.resolve()
    python = args.python.resolve()
    validate_runtime(run_root, env_file, python)

    state_dir = run_root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / "launch.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Round8 launcher is currently in preflight") from error

        arms = [preflight_arm(run_root, arm) for arm in ARMS]
        commands = [
            tmux_launch_command(
                run_root=run_root,
                env_file=env_file,
                python=python,
                arm=arm,
                slot=slot,
                smoke=args.smoke,
            )
            for slot, arm in enumerate(ARMS)
        ]
        for arm_state, command in zip(arms, commands, strict=True):
            disposition = "resume-partial" if arm_state.resumable_partial else "fresh"
            print(
                f"PREFLIGHT arm={arm_state.arm} session={arm_state.session} "
                f"slot={ARMS.index(arm_state.arm)} disposition={disposition}"
            )
            if args.dry_run:
                print("DRY_RUN " + " ".join(command))

        if args.dry_run:
            print("No tmux sessions were launched.")
            return 0

        for arm, command in zip(ARMS, commands, strict=True):
            subprocess.run(command, check=True)
            print(
                f"LAUNCHED arm={arm} session={session_name(arm)} "
                f"CUDA_VISIBLE_DEVICES={ARMS.index(arm)}"
            )

    print(
        "NOTE: Ollama Cloud performs inference; CUDA_VISIBLE_DEVICES values are "
        "process-isolation labels and do not reserve or consume V100 compute."
    )
    verify_command = [
        str(python),
        str(run_root / "scripts" / "verify_round8.py"),
        "--run-root",
        str(run_root),
        "--samples",
        "2",
        "--initial-delay",
        str(args.verify_initial_delay),
        "--interval",
        str(args.verify_interval),
        "--expect",
        "running-or-complete",
    ]
    completed = subprocess.run(verify_command, check=False)
    if completed.returncode != 0:
        print(
            "Verification failed; detached sessions were intentionally left untouched. "
            "Inspect with verify_round8.py before taking further action.",
            file=sys.stderr,
        )
        return completed.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"ROUND8_LAUNCH_REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error

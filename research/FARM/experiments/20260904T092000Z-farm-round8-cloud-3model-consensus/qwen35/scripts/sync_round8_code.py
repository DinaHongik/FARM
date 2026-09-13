#!/usr/bin/env python3
"""Atomically sync an uploaded Round8 code tree without touching artifacts.

Run this on the DGX after uploading into a staging directory.  The destination
may already contain ``.env``, ``.deps``, logs, records, progress, results, and
status; none of those paths are selected or deleted by this tool.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


RUN_ID = "20260904T074517Z-farm-round8-executable-agent"
ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
    "trigger_consensus_executable",
)
MANAGED_SESSIONS = (
    *(f"farm-r8-{arm.replace('_', '-')}" for arm in ARMS),
    *(f"farm-r8-local-{arm.replace('_', '-')}" for arm in ARMS),
    *(
        f"farm-r8-local-smoke-"
        f"{5 if arm == 'trigger_consensus_executable' else 2}-"
        f"{arm.replace('_', '-')}"
        for arm in ARMS
    ),
    *(f"farm-r8-ollama-{port}" for port in (11435, 11436, 11437, 11438)),
)
ROOT_FILES = {
    "Dockerfile",
    "EXPERIMENT_MATRIX.json",
    "LAUNCH_HANDOFF.md",
    "LOCAL_GRANITE_HANDOFF.md",
    "PRIMARY_SOURCE_ARCHITECTURE_RESEARCH.md",
    "PRIOR_FAILURE_AUDIT.md",
    "ROUND8_SCOPE.md",
    "requirements-round8.txt",
}
CODE_DIRECTORIES = {"scripts", "tests", "configs"}
CODE_SUFFIXES = {".py", ".sh", ".json", ".md"}
PROTECTED_NAMES = {
    ".deps",
    ".env",
    ".venv",
    "attempts",
    "logs",
    "local_granite",
    "pids",
    "progress",
    "records",
    "results",
    "state",
    "status",
}


def selected_files(source: Path) -> tuple[Path, ...]:
    selected: list[Path] = []
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_symlink():
            if relative.parts and (
                relative.parts[0] in CODE_DIRECTORIES or relative.name in ROOT_FILES
            ):
                raise RuntimeError(f"refusing symlink in sync input: {relative}")
            continue
        if not path.is_file() or any(part in PROTECTED_NAMES for part in relative.parts):
            continue
        if len(relative.parts) == 1 and relative.name in ROOT_FILES:
            selected.append(relative)
        elif relative.parts[0] in CODE_DIRECTORIES and path.suffix in CODE_SUFFIXES:
            selected.append(relative)
    return tuple(sorted(selected, key=lambda item: item.as_posix()))


def active_sessions() -> tuple[str, ...]:
    active: list[str] = []
    for name in MANAGED_SESSIONS:
        result = subprocess.run(
            ["tmux", "has-session", "-t", f"={name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            active.append(name)
    return tuple(active)


def active_runtime_pids(destination: Path) -> tuple[tuple[Path, int], ...]:
    """Conservatively detect live wrappers/servers even if tmux disappeared."""

    candidates = list((destination / "pids").glob("*.pid"))
    local_root = destination / "local_granite"
    candidates.extend(local_root.glob("*/servers/*.pid"))
    candidates.extend(local_root.glob("*/full/pids/*.pid"))
    candidates.extend(local_root.glob("*/smoke-*/pids/*.pid"))
    active: list[tuple[Path, int]] = []
    for path in sorted(set(candidates)):
        try:
            pid = int(path.read_text(encoding="utf-8").strip())
            if pid <= 1:
                continue
        except (OSError, ValueError):
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            pass
        active.append((path.relative_to(destination), pid))
    return tuple(active)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.sync-tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb", closefd=False) as output_handle:
            while True:
                chunk = input_handle.read(1024 * 1024)
                if not chunk:
                    break
                output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(temporary, source.stat().st_mode & 0o777)
    os.replace(temporary, destination)


def sync_files(source: Path, destination: Path, relative_paths: Iterable[Path]) -> int:
    count = 0
    for relative in relative_paths:
        _atomic_copy(source / relative, destination / relative)
        print(f"SYNCED {relative.as_posix()}")
        count += 1
    return count


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="uploaded staging tree")
    parser.add_argument("destination", type=Path, help="persistent Round8 run root")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.source.resolve()
    destination = args.destination.resolve()
    if source == destination:
        raise RuntimeError("source and destination must differ")
    if source.name != RUN_ID or destination.name != RUN_ID:
        raise RuntimeError("both sync paths must end in the frozen Round8 run ID")
    if not (source / "ROUND8_SCOPE.md").is_file():
        raise RuntimeError("source is not a complete Round8 code tree")
    sessions = active_sessions()
    if sessions:
        raise RuntimeError(f"refusing code sync while Round8 sessions are active: {sessions}")
    processes = active_runtime_pids(destination)
    if processes:
        raise RuntimeError(
            f"refusing code sync while Round8 process PIDs are live: {processes}"
        )
    files = selected_files(source)
    if not files:
        raise RuntimeError("no allowlisted code files found")
    if args.dry_run:
        for relative in files:
            print(f"WOULD_SYNC {relative.as_posix()}")
        print("No files or artifacts were changed.")
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    count = sync_files(source, destination, files)
    print(f"Sync complete: {count} code/documentation files; runtime artifacts preserved.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROUND8_SYNC_REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error

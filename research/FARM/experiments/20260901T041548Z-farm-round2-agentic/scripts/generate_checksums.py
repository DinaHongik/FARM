#!/usr/bin/env python3
"""Snapshot hashes for all non-checkpoint experiment artifacts."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from experiment_lib import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = run_root / "checksums.sha256"
    excluded_parts = {"checkpoints", "__pycache__"}
    paths = []
    for path in run_root.rglob("*"):
        if not path.is_file() or path == output or any(part in excluded_parts for part in path.parts):
            continue
        if path.name.endswith(".lock") or path.name == ".status.lock":
            continue
        paths.append(path)
    lines = [f"{sha256_file(path)}  {path.relative_to(run_root)}" for path in sorted(paths)]
    fd, temporary = tempfile.mkstemp(prefix=".checksums.", dir=str(run_root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"wrote {len(lines)} hashes to {output}")


if __name__ == "__main__":
    main()

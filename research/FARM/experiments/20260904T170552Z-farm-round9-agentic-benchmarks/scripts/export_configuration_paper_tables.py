#!/usr/bin/env python3
"""Emit three strict, aggregate-only FARM configuration table artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.configuration_reporting import (  # noqa: E402
    ConfigurationReportInputError,
    build_configuration_paper_tables,
)


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.stat(path.parent).st_mode & 0o077:
        raise PermissionError("output directory must not be group/world accessible")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-shot-records", type=Path, required=True)
    parser.add_argument("--one-shot-manifest", type=Path, required=True)
    parser.add_argument("--bounded-records", type=Path, required=True)
    parser.add_argument("--bounded-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-records", type=Path, required=True)
    parser.add_argument("--one-shot-output", type=Path, required=True)
    parser.add_argument("--bounded-output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    args = parser.parse_args()
    outputs = {
        "one_shot": args.one_shot_output,
        "bounded": args.bounded_output,
        "paired": args.paired_output,
    }
    try:
        if len(set(outputs.values())) != len(outputs) or any(
            path.exists() or path.is_symlink() for path in outputs.values()
        ):
            raise FileExistsError("all output paths must be distinct and absent")
        tables = build_configuration_paper_tables(
            one_shot_records=args.one_shot_records,
            one_shot_manifest=args.one_shot_manifest,
            bounded_records=args.bounded_records,
            bounded_manifest=args.bounded_manifest,
            endpoint_records=args.endpoint_records,
        )
        for name, path in outputs.items():
            _write_exclusive(path, tables[name])
    except (ConfigurationReportInputError, OSError, ValueError):
        print("configuration paper-table export failed closed", file=sys.stderr)
        return 2
    print(json.dumps({"status": "written", "files": 3}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

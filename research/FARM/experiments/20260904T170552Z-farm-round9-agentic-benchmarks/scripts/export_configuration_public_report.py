#!/usr/bin/env python3
"""Emit a fresh access-controlled aggregate report for FARM configuration arms.

Despite this legacy filename, the output is not a FARM paper-table or public
release artifact. Use the release stager's strict aggregate mode before any
publication.
"""

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
    build_configuration_internal_report,
)


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.stat(path.parent).st_mode & 0o077:
        raise PermissionError("output directory must not be group/world accessible")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--one-shot-records", type=Path, required=True)
    parser.add_argument("--one-shot-manifest", type=Path, required=True)
    parser.add_argument("--bounded-records", type=Path, required=True)
    parser.add_argument("--bounded-manifest", type=Path, required=True)
    parser.add_argument("--endpoint-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_configuration_internal_report(
            one_shot_records=args.one_shot_records,
            one_shot_manifest=args.one_shot_manifest,
            bounded_records=args.bounded_records,
            bounded_manifest=args.bounded_manifest,
            endpoint_records=args.endpoint_records,
        )
        _write_exclusive(args.output, report)
    except (ConfigurationReportInputError, OSError, ValueError):
        print("configuration internal report failed closed", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

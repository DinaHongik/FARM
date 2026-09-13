#!/usr/bin/env python3
"""Emit a privacy-safe paired comparison for two completed FARM arms."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.configuration_comparison import (
    ComparisonInputError,
    compare_configuration_ledgers,
)


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--one-shot-records", type=Path, required=True)
    parser.add_argument("--bounded-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        comparison = compare_configuration_ledgers(
            args.one_shot_records,
            args.bounded_records,
        )
        _write_exclusive(args.output, comparison)
    except (ComparisonInputError, OSError):
        print("configuration comparison failed closed", file=sys.stderr)
        return 2
    print(json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

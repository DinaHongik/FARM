#!/usr/bin/env python3
"""Validate and exclusively write a strict public Yao aggregate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


ROUND9_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND9_ROOT / "src"))

from farm_r9.yao_reporting import (  # noqa: E402
    YaoPublicReportError,
    export_public_yao_aggregate,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="private Yao aggregate.json"
    )
    parser.add_argument(
        "--current-records",
        type=Path,
        help="current arm terminal records; required for a paired aggregate",
    )
    parser.add_argument(
        "--reference-records",
        type=Path,
        help="one-shot terminal records; required for a paired aggregate",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="new public aggregate path"
    )
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise YaoPublicReportError("terminal ledger has a non-object record")
            rows.append(value)
    return rows


def _write_exclusive(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        if (args.current_records is None) != (args.reference_records is None):
            raise YaoPublicReportError(
                "current and reference terminal ledgers must be supplied together"
            )
        with args.input.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        current_records = (
            _read_jsonl(args.current_records)
            if args.current_records is not None
            else None
        )
        reference_records = (
            _read_jsonl(args.reference_records)
            if args.reference_records is not None
            else None
        )
        public = export_public_yao_aggregate(
            source,
            current_records=current_records,
            reference_records=reference_records,
        )
        _write_exclusive(args.output, public)
    except (OSError, json.JSONDecodeError, YaoPublicReportError) as error:
        # Validator messages contain schema paths/categories, never input values.
        print(f"Yao public aggregate export refused: {error}", file=sys.stderr)
        return 2
    print(f"wrote one validated Yao public aggregate to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

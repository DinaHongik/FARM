#!/usr/bin/env python3
"""Export a strict, disclosure-safe JSON aggregate from Round 9 results."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


ROUND9_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND9_ROOT / "src"))

from farm_r9.privacy import (  # noqa: E402
    DataClassification,
    PrivacyViolation,
    export_public_aggregate,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="private aggregate JSON"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="public aggregate JSON"
    )
    parser.add_argument(
        "--classification",
        required=True,
        choices=[item.value for item in DataClassification],
        help="explicit classification of the source aggregate",
    )
    return parser.parse_args(argv)


def _validated_public_payload(
    value: Any,
    *,
    source_classification: DataClassification,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return export_public_aggregate(
            value,
            source_classification=source_classification,
        )
    if isinstance(value, list):
        if not value:
            raise PrivacyViolation("aggregate list must not be empty")
        return [
            export_public_aggregate(
                record,
                source_classification=source_classification,
            )
            for record in value
        ]
    raise PrivacyViolation(
        "input must contain one aggregate object or a list of objects"
    )


def _write_json_atomic_mode_0600(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.stat(path.parent).st_mode & 0o077:
        raise PermissionError("output directory must not be group/world accessible")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path, follow_symlinks=False)
        temporary_path.unlink()
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        with args.input.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        classification = DataClassification(args.classification)
        public_payload = _validated_public_payload(
            source,
            source_classification=classification,
        )
        _write_json_atomic_mode_0600(args.output, public_payload)
    except (OSError, ValueError, PrivacyViolation) as error:
        # Validation errors contain structural paths/field names, never values.
        print(f"public export refused: {error}", file=sys.stderr)
        return 2

    count = len(public_payload) if isinstance(public_payload, list) else 1
    print(f"wrote {count} validated public aggregate record(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only content audit of the four released TARGE recipe splits.

Only the fixed paths in ``RELEASE_FILES`` are opened.  The utility does not
discover, read, or import any FARM development, confirmation, or test split.
Its JSON report is printed to stdout; it never writes to the audited tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


RELEASE_FILES: dict[str, Path] = {
    "train": Path("data/dataset/train_recipe.json"),
    "gold": Path("data/dataset/gold/test_recipe.json"),
    "noisy": Path("data/dataset/noisy/test_recipe.json"),
    "oneshot": Path("data/dataset/gold/test_recipe_one_shot.json"),
}

TupleKey = tuple[str, str]


def _strict_text(record: Mapping[str, Any], field: str, *, location: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location}.{field} must be a non-empty string")
    return value


def _load_split(path: Path, *, name: str) -> tuple[dict[str, Any], Counter[TupleKey]]:
    raw = path.read_bytes()
    try:
        records = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a top-level JSON array")

    tuples: Counter[TupleKey] = Counter()
    for index, record in enumerate(records):
        location = f"{name}[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{location} must be an object")
        input_text = _strict_text(record, "input", location=location)
        output_text = _strict_text(record, "output", location=location)
        tuples[(input_text, output_text)] += 1

    duplicate_groups = sum(count > 1 for count in tuples.values())
    duplicate_rows = len(records) - len(tuples)
    stats = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rows": len(records),
        "unique_input_output_tuples": len(tuples),
        "duplicate_rows_beyond_first": duplicate_rows,
        "duplicate_tuple_groups": duplicate_groups,
        "maximum_tuple_multiplicity": max(tuples.values(), default=0),
    }
    return stats, tuples


def _inside(root: Path, path: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"release path escapes TARGE root: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"release path is not a regular file: {path}")
    return resolved


def audit_release(targe_root: str | Path) -> dict[str, Any]:
    """Return hashes, duplicates, and exact tuple overlaps without writing files."""

    root = Path(targe_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"TARGE root is not a directory: {root}")

    split_stats: dict[str, dict[str, Any]] = {}
    tuple_counts: dict[str, Counter[TupleKey]] = {}
    for name, relative_path in RELEASE_FILES.items():
        path = _inside(root, root / relative_path)
        stats, tuples = _load_split(path, name=name)
        stats["path"] = str(relative_path)
        split_stats[name] = stats
        tuple_counts[name] = tuples

    train_tuples = set(tuple_counts["train"])
    overlap_with_train: dict[str, dict[str, Any]] = {}
    for name in ("gold", "noisy", "oneshot"):
        split_tuples = set(tuple_counts[name])
        overlap = train_tuples & split_tuples
        denominator = len(split_tuples)
        overlap_with_train[name] = {
            "unique_tuple_overlap": len(overlap),
            "split_unique_tuples": denominator,
            "fraction_of_split_unique_tuples": len(overlap) / denominator if denominator else 0.0,
        }

    pairwise: dict[str, dict[str, int]] = {}
    names = tuple(RELEASE_FILES)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            pairwise[f"{left}__{right}"] = {
                "unique_tuple_overlap": len(set(tuple_counts[left]) & set(tuple_counts[right]))
            }

    return {
        "schema_version": 1,
        "audit_kind": "targe_release_exact_input_output_tuple_audit",
        "read_only": True,
        "targe_root": str(root),
        "fixed_release_paths": [str(path) for path in RELEASE_FILES.values()],
        "splits": split_stats,
        "overlap_with_train": overlap_with_train,
        "pairwise_unique_tuple_overlap": pairwise,
    }


def _default_targe_root() -> Path:
    return Path(__file__).resolve().parents[3] / "Targe"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targe-root",
        type=Path,
        default=_default_targe_root(),
        help="root of the released TARGE artifact (default: repository FARM/Targe)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON instead of indented JSON",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = audit_release(args.targe_root)
    except (OSError, ValueError) as error:
        parser.exit(2, f"audit error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

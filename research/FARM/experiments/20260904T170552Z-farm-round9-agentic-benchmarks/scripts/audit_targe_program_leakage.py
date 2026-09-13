#!/usr/bin/env python3
"""Write a versioned, non-destructive exact TARGE input/program leakage audit."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import sha256_file, write_json_atomic, write_jsonl_atomic  # noqa: E402
from farm_r9.targe_leakage_audit import (  # noqa: E402
    NORMALIZATION_VERSION,
    SCHEMA_VERSION,
    classify_frozen_sample,
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=RUN_ROOT.parents[1])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUN_ROOT / "audits" / "targe_exact_input_program_v1",
    )
    parser.add_argument("--expected-n", type=int, default=150)
    parser.add_argument(
        "--splits", nargs="+", choices=("gold", "noisy", "one_shot"),
        default=("gold", "noisy", "one_shot"),
    )
    return parser.parse_args(argv)


def _paths(repo_root: Path, split: str) -> tuple[Path, Path, Path]:
    train = repo_root / "Targe" / "data" / "dataset" / "train_recipe.json"
    base = repo_root / "Targe" / "data" / "dataset"
    test = (
        base / "gold" / "test_recipe_one_shot.json"
        if split == "one_shot"
        else base / split / "test_recipe.json"
    )
    selected = RUN_ROOT / "prepared" / f"targe_{split}" / "cases.jsonl"
    return selected, train, test


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing audit path: {output_root}")
    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "implementation_hashes": {
            "audit_module_sha256": sha256_file(
                RUN_ROOT / "src" / "farm_r9" / "targe_leakage_audit.py"
            ),
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
        "splits": {},
    }
    for split in args.splits:
        benchmark = f"targe_{split}"
        selected, train, test = _paths(repo_root, split)
        records, aggregate = classify_frozen_sample(
            selected_cases_path=selected,
            train_recipe_path=train,
            test_recipe_path=test,
            benchmark=benchmark,
            expected_n=args.expected_n,
        )
        aggregate["implementation_hashes"] = summary["implementation_hashes"]
        destination = output_root / split
        destination.mkdir(mode=0o700)
        records_path = destination / "case_audit.jsonl"
        aggregate_path = destination / "aggregate.json"
        write_jsonl_atomic(records_path, records)
        os.chmod(records_path, 0o600)
        aggregate["records_sha256"] = sha256_file(records_path)
        write_json_atomic(aggregate_path, aggregate)
        os.chmod(aggregate_path, 0o600)
        summary["splits"][split] = {
            "n": aggregate["n"],
            "class_counts": aggregate["class_counts"],
            "class_percentages": aggregate["class_percentages"],
            "aggregate_sha256": sha256_file(aggregate_path),
            "records_sha256": aggregate["records_sha256"],
        }

    summary_path = output_root / "SUMMARY.json"
    write_json_atomic(summary_path, summary)
    os.chmod(summary_path, 0o600)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

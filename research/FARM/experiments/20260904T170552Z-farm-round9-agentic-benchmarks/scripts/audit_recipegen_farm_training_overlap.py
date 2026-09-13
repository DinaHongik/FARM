#!/usr/bin/env python3
"""Run the local-only RecipeGen++/FARM-v2 training-overlap audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import (  # noqa: E402
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from farm_r9.recipegen_farm_overlap import (  # noqa: E402
    audit_frozen_recipegen_sample,
    make_public_aggregate,
)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-root", type=Path, default=RUN_ROOT.parents[1])
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUN_ROOT / "audits" / "recipegen_farm_training_overlap_v2",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("gold", "noisy"),
        default=("gold", "noisy"),
    )
    return parser.parse_args(argv)


def _split_paths(split: str) -> tuple[Path, Path]:
    return (
        RUN_ROOT / "prepared" / f"recipegen_{split}" / "cases.jsonl",
        RUN_ROOT / "manifests" / "samples" / f"recipegen_{split}.json",
    )


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing audit path: {output_root}")
    if len(set(args.splits)) != len(args.splits):
        raise SystemExit("refusing duplicate split targets")

    farm_root = args.farm_root.resolve()
    farm_data_root = farm_root / "data" / "v2"
    processed_csv = farm_root / "RecipeGen" / "dataset" / "processed.csv"
    completed: dict[str, tuple[list[dict[str, object]], dict[str, object]]] = {}
    for split in args.splits:
        cases_path, sample_manifest_path = _split_paths(split)
        completed[split] = audit_frozen_recipegen_sample(
            cases_path=cases_path,
            sample_manifest_path=sample_manifest_path,
            processed_csv=processed_csv,
            split=split,
            farm_data_root=farm_data_root,
        )

    module_path = RUN_ROOT / "src" / "farm_r9" / "recipegen_farm_overlap.py"
    # The public hash names one complete immutable implementation artifact.
    # The wrapper is independently inspectable but is not folded into a
    # synthetic multi-file digest.
    code_artifact_sha256 = sha256_file(module_path)

    output_root.mkdir(parents=True, mode=0o700)
    os.chmod(output_root, 0o700)
    console_summary: dict[str, object] = {"status": "complete", "splits": {}}
    for split, (records, summary) in completed.items():
        destination = output_root / split
        destination.mkdir(mode=0o700)
        os.chmod(destination, 0o700)
        case_audit_path = destination / "case_audit.jsonl"
        aggregate_path = destination / "aggregate_public.json"

        write_jsonl_atomic(case_audit_path, records)
        os.chmod(case_audit_path, 0o600)
        aggregate = make_public_aggregate(
            summary=summary,
            case_audit_sha256=sha256_file(case_audit_path),
            code_artifact_sha256=code_artifact_sha256,
        )
        write_json_atomic(aggregate_path, aggregate)
        os.chmod(aggregate_path, 0o600)
        console_summary["splits"][split] = {
            "n": aggregate["n"],
            "aggregate_artifact_sha256": sha256_file(aggregate_path),
        }

    print(json.dumps(console_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

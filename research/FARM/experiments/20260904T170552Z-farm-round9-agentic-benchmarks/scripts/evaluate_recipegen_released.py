#!/usr/bin/env python3
"""Verify RecipeGen++ predictions with explicit endpoint gates and v3 metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.recipegen_released import (  # noqa: E402
    evaluate_released_predictions,
    write_aggregate_exclusive,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", required=True, type=Path, help="pinned Zenodo results.zip"
    )
    parser.add_argument("--processed-csv", required=True, type=Path)
    parser.add_argument(
        "--cases", required=True, type=Path, help="frozen public cases.jsonl"
    )
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("gold", "noisy"))
    parser.add_argument(
        "--output", required=True, type=Path, help="new aggregate-only JSON path"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    aggregate = evaluate_released_predictions(
        archive_path=args.archive,
        processed_csv=args.processed_csv,
        cases_path=args.cases,
        manifest_path=args.sample_manifest,
        split=args.split,
    )
    write_aggregate_exclusive(args.output, aggregate)
    print(
        json.dumps(
            {
                "benchmark_label": aggregate["benchmark_label"],
                "n": aggregate["n"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

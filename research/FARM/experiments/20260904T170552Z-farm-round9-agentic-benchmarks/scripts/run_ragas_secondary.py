#!/usr/bin/env python3
"""Run deterministic RAGAS-compatible ID diagnostics on exactly 150 cases."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import (
    ordered_ids_sha256,
    read_jsonl,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from farm_r9.privacy import DataClassification, export_public_aggregate
from farm_r9.ragas_secondary import (
    DEFAULT_EXPECTED_N,
    IDMetricRecord,
    public_secondary_aggregate,
    score_id_records,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-label", required=True)
    parser.add_argument(
        "--classification",
        choices=[item.value for item in DataClassification],
        required=True,
    )
    parser.add_argument("--expected-n", type=int, default=DEFAULT_EXPECTED_N)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if args.expected_n != DEFAULT_EXPECTED_N:
        raise SystemExit("Round 9 protocol requires --expected-n 150")
    private_rows_path = args.output_dir / "records_private.jsonl"
    private_aggregate_path = args.output_dir / "aggregate_private.json"
    public_aggregate_path = args.output_dir / "aggregate_public.json"
    existing = [
        path
        for path in (private_rows_path, private_aggregate_path, public_aggregate_path)
        if path.exists()
    ]
    if existing:
        raise SystemExit("refusing to overwrite existing RAGAS-secondary artifacts")

    source_rows = read_jsonl(args.input)
    records = [IDMetricRecord.from_mapping(row) for row in source_rows]
    private_rows, private_aggregate = score_id_records(
        records, expected_n=args.expected_n
    )
    private_aggregate = {
        **private_aggregate,
        "input_sha256": sha256_file(args.input),
        "ordered_case_ids_sha256": ordered_ids_sha256(source_rows),
        "classification": args.classification,
        "role": "secondary_diagnostic",
    }
    candidate_public = public_secondary_aggregate(
        benchmark_label=args.benchmark_label, id_aggregate=private_aggregate
    )
    candidate_public["hashes"] = {
        "aggregate_input_sha256": sha256_file(args.input),
    }
    if DataClassification(args.classification) is DataClassification.CONFIDENTIAL:
        # FARM's release table permits only aligned case counts and uncertainty.
        # Keep macro ID means in the access-controlled aggregate, where their
        # fractional units are explicit; do not invent integer case numerators.
        names = set(candidate_public["raw_numerators"])
        candidate_public = {
            "n": candidate_public["n"],
            "raw_numerators": candidate_public["raw_numerators"],
            "raw_denominators": candidate_public["raw_denominators"],
            "percentages": {name: candidate_public["percentages"][name] for name in names},
            "confidence_intervals": {name: candidate_public["confidence_intervals"][name] for name in names},
            "failure_counts": {},
            "hashes": {"source_artifact_sha256": sha256_file(args.input)},
        }
    public_aggregate = export_public_aggregate(
        candidate_public,
        source_classification=DataClassification(args.classification),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    write_jsonl_atomic(private_rows_path, private_rows)
    write_json_atomic(private_aggregate_path, private_aggregate)
    write_json_atomic(public_aggregate_path, public_aggregate)
    # artifact_io uses mkstemp (0600), but enforce the deep/private boundary
    # even on systems with an unusual umask or an existing permissive directory.
    os.chmod(private_rows_path, 0o600)
    os.chmod(private_aggregate_path, 0o600)
    print(
        json.dumps(
            {
                "n": private_aggregate["n"],
                "aggregate_public": str(public_aggregate_path),
                "records_private": str(private_rows_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

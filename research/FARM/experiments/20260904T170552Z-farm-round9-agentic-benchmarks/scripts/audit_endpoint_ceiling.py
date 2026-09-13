#!/usr/bin/env python3
"""Partition a completed public endpoint run by initial top-k coverage."""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.artifact_io import read_jsonl, sha256_file
from farm_r9.endpoint_runner import _latest_by_case, _rescore_terminal_records, wilson_95
from build_ragas_id_inputs import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cases = read_jsonl(args.candidates)
    if len(cases) != 150 or any(case.get("data_classification") != "public" for case in cases):
        raise SystemExit("audit requires frozen public n=150")
    by_id = {case["case_id"]: case for case in cases}
    records = _latest_by_case(read_jsonl(args.records))
    if set(records) != set(by_id) or any(not record["terminal"] for record in records.values()):
        raise SystemExit("audit requires complete terminal sample")
    records = _rescore_terminal_records(records, by_id)
    coverage = {side: {row["case_id"]: set(row["reference_context_ids"]) <= set(row["retrieved_context_ids"])
                     for row in build(cases, side)} for side in ("trigger", "action", "joint")}
    partition = Counter()
    for case_id, record in records.items():
        if not coverage["joint"][case_id]:
            if record["scores"]["function_joint"]:
                raise SystemExit("correct prediction contradicts fixed-candidate ceiling")
            partition["gold_pair_outside_initial_top10"] += 1
        elif record["prediction"] is None:
            partition["covered_pair_no_valid_prediction"] += 1
        elif record["scores"]["function_joint"]:
            partition["covered_pair_correct"] += 1
        else:
            partition["covered_pair_wrong_selection"] += 1
    report = {"n": 150, "raw_numerators": dict(partition),
              "raw_denominators": {key: 150 for key in partition},
              "percentages": {key: 100 * value / 150 for key, value in partition.items()},
              "confidence_intervals": {key: wilson_95(value, 150) for key, value in partition.items()},
              "candidate_coverage_counts": {side: sum(values.values()) for side, values in coverage.items()},
              "hashes": {"source_artifact_sha256": sha256_file(args.candidates),
                         "artifact_sha256": sha256_file(args.records)}}
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optimize the DSPy configuration module on explicitly public train gold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import ordered_ids_sha256, read_jsonl, sha256_file, write_json_atomic
from farm_r9.dspy_program import (
    AppletConfigurationProgram, DSPyUnavailable, dspy, exact_action_json_metric,
)


PUBLIC_SOURCES = {"recipegen", "interactive_ifttt", "bfcl_v4", "targe"}
INPUTS = (
    "instruction", "grounding_text", "selected_trigger_json",
    "selected_action_json", "observations_json",
)


def _localhost_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise ValueError("DSPy optimization is restricted to a DGX-local 127.0.0.1 endpoint")
    return value.rstrip("/")


def _training_rows(path: Path, source: str) -> list[dict]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("public DSPy train set is empty")
    for row in rows:
        if row.get("data_classification") != "public" or row.get("training_source") != source:
            raise ValueError("every DSPy training record needs matching registered public lineage")
        if row.get("split") != "train" or not isinstance(row.get("gold_action_json"), str):
            raise ValueError("every record needs split=train and supervised gold_action_json")
        if any(not isinstance(row.get(key), str) for key in INPUTS):
            raise ValueError("DSPy training inputs must all be strings")
        try:
            parsed = json.loads(row["gold_action_json"])
        except json.JSONDecodeError as error:
            raise ValueError("gold_action_json must be valid JSON") from error
        if not isinstance(parsed, dict):
            raise ValueError("gold_action_json must encode one object")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-jsonl", type=Path, required=True)
    parser.add_argument("--training-source", choices=sorted(PUBLIC_SOURCES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--max-labeled-demos", type=int, default=8)
    parser.add_argument("--max-bootstrapped-demos", type=int, default=4)
    args = parser.parse_args()
    if dspy is None:
        raise DSPyUnavailable("dspy is required in the pinned DGX environment")
    api_base = _localhost_origin(args.api_base)
    rows = _training_rows(args.train_jsonl, args.training_source)
    heldout = read_jsonl(args.heldout_jsonl)
    train_ids = {row["case_id"] for row in rows}
    heldout_ids = {row["case_id"] for row in heldout}
    overlap = train_ids & heldout_ids
    if overlap:
        raise ValueError("DSPy train and held-out IDs overlap")
    trainset = [
        dspy.Example(**{**{key: row[key] for key in INPUTS}, "action_json": row["gold_action_json"]}).with_inputs(*INPUTS)
        for row in rows
    ]
    lm = dspy.LM(f"openai/{args.model}", api_base=api_base + "/v1", api_key="local-only", temperature=0.0)
    dspy.configure(lm=lm)
    optimizer = dspy.BootstrapFewShot(
        metric=exact_action_json_metric,
        max_bootstrapped_demos=args.max_bootstrapped_demos,
        max_labeled_demos=args.max_labeled_demos,
    )
    compiled = optimizer.compile(AppletConfigurationProgram(), trainset=trainset)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    state_path = args.output_dir / "program_state.json"
    compiled.save(str(state_path), save_program=False)
    manifest = {
        "schema_version": "round9-dspy-optimizer-v1", "status": "optimized",
        "optimizer": "BootstrapFewShot", "optimizer_executed": True,
        "training_source": args.training_source, "training_classification": "public",
        "train_count": len(rows), "training_case_ids_sha256": ordered_ids_sha256(rows),
        "heldout_overlap_count": 0, "program_state_path": state_path.name,
        "program_state_sha256": sha256_file(state_path),
    }
    write_json_atomic(args.output_dir / "optimizer_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


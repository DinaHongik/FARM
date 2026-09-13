#!/usr/bin/env python3
"""Build opaque RecipeGen endpoint-ID inputs for RAGAS-secondary diagnostics."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import read_jsonl, write_jsonl_atomic


def _canonical_gold(gold: dict, side: str) -> str:
    return f"{gold[f'{side}_channel_norm']}::{gold[f'{side}_function_norm']}"


def build(rows: list[dict], side: str) -> list[dict]:
    if len(rows) != 150:
        raise ValueError("Round 9 RAGAS-secondary input requires exactly 150 cases")
    output = []
    for row in rows:
        if row.get("benchmark") not in {"recipegen_gold", "recipegen_noisy"}:
            raise ValueError("ID diagnostics currently support single-gold RecipeGen only")
        private = row["private"]
        gold = private["gold"]
        if side in {"trigger", "action"}:
            retrieved = list(private[side]["alias_map"].values())
            references = [_canonical_gold(gold, side)]
        else:
            retrieved = [
                *("trigger:" + value for value in private["trigger"]["alias_map"].values()),
                *("action:" + value for value in private["action"]["alias_map"].values()),
            ]
            references = [
                "trigger:" + _canonical_gold(gold, "trigger"),
                "action:" + _canonical_gold(gold, "action"),
            ]
        if len(retrieved) != (20 if side == "joint" else 10):
            raise ValueError("candidate artifact does not contain the frozen top-10 per side")
        output.append({
            "case_id": row["case_id"],
            "retrieved_context_ids": retrieved,
            "reference_context_ids": references,
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--side", choices=("trigger", "action", "joint"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("refusing to overwrite an existing RAGAS ID input")
    rows = build(read_jsonl(args.candidates), args.side)
    write_jsonl_atomic(args.output, rows)
    os.chmod(args.output, 0o600)
    print(f"wrote {len(rows)} opaque-ID rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

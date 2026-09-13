#!/usr/bin/env python3
"""Emit an aggregate-only paired comparison of official BFCL arm scores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.bfcl_comparison import (  # noqa: E402
    BFCLComparisonInputError,
    compare_bfcl_arms,
)
from farm_r9.bfcl_runner import (  # noqa: E402
    BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN,
)


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-step-result", required=True, type=Path)
    parser.add_argument("--single-step-score", required=True, type=Path)
    parser.add_argument("--native-agent-result", required=True, type=Path)
    parser.add_argument("--native-agent-score", required=True, type=Path)
    parser.add_argument(
        "--single-step-binding",
        type=Path,
        help="single-step RUN_BINDING.json; defaults to the result tree binding",
    )
    parser.add_argument(
        "--native-agent-binding",
        type=Path,
        help="native-agent RUN_BINDING.json; defaults to the result tree binding",
    )
    parser.add_argument(
        "--native-agent-call-budget",
        type=int,
        choices=(4, BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN),
        default=4,
        help=(
            "declared native-agent per-turn budget: 4 for the original arm, "
            "21 for the BFCL-official-horizon correction"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        comparison = compare_bfcl_arms(
            single_step_result=args.single_step_result,
            single_step_score=args.single_step_score,
            native_agent_result=args.native_agent_result,
            native_agent_score=args.native_agent_score,
            single_step_binding=(
                args.single_step_binding
                or args.single_step_result.parent.parent / "RUN_BINDING.json"
            ),
            native_agent_binding=(
                args.native_agent_binding
                or args.native_agent_result.parent.parent / "RUN_BINDING.json"
            ),
            native_agent_call_budget=args.native_agent_call_budget,
        )
        _write_exclusive(args.output, comparison)
    except (BFCLComparisonInputError, OSError):
        print("BFCL comparison failed closed", file=sys.stderr)
        return 2
    print(json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

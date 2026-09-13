#!/usr/bin/env python3
"""CLI for strict deterministic TARGE-compatible endpoint metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

try:
    from .evaluator import evaluate_payload
except ImportError:  # Direct script execution.
    from evaluator import evaluate_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        help="JSON payload path, or '-' to read JSON from stdin",
    )
    parser.add_argument(
        "--k",
        dest="cutoffs",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="positive Recall/MRR cutoffs (default: 1 3 5)",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def _read_payload(input_path: str):
    try:
        if input_path == "-":
            return json.load(sys.stdin)
        with Path(input_path).open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read input JSON: {error}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = evaluate_payload(_read_payload(args.input), cutoffs=args.cutoffs)
    except ValueError as error:
        parser.exit(2, f"evaluation error: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

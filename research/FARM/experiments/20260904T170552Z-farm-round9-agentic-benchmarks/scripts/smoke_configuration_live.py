#!/usr/bin/env python3
"""One synthetic, non-FARM live gate for the local configuration model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.configuration_agent import run_configuration_agent  # noqa: E402
from farm_r9.ollama_client import OllamaChatClient  # noqa: E402
from farm_r9.privacy import DataClassification, DataSource  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=["single_shot_configurator", "bounded_configuration_agent"],
        default="single_shot_configurator",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("synthetic gate output must be a fresh path")
    args.output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(args.output_dir, 0o700)
    case = {
        "benchmark": "recipegen_gold",
        "case_id": "synthetic-config-gate-v1",
        "input": {"query": "Save recipe posts as notes"},
        "public_evidence": {
            "trigger_candidates": [{
                "alias": "T01", "side": "trigger", "service": "Feed",
                "function": "New item", "fields": [],
                "ingredients": [{"slug": "title", "label": "Title", "value_type": "string"}],
            }],
            "action_candidates": [{
                "alias": "A01", "side": "action", "service": "Notes",
                "function": "Create note", "ingredients": [],
                "fields": [
                    {"slug": "body", "label": "Body", "required": True, "bindable": True,
                     "value_type": "string", "resource_like": False, "auth_like": False},
                    {"slug": "folder", "label": "Folder", "required": False, "bindable": False,
                     "value_type": "string", "resource_like": True, "auth_like": False},
                ],
            }],
        },
        "configuration_evidence": {"clarification_answers": {}},
    }
    client = OllamaChatClient(
        host=args.host,
        api_key=None,
        model=args.model,
        cache_directory=args.output_dir / "cache",
        journal_path=args.output_dir / "journal.jsonl",
        cloud=False,
        limiter=None,
        timeout_seconds=900,
    )
    result = run_configuration_agent(
        client=client,
        case=case,
        trigger_alias="T01",
        action_alias="A01",
        benchmark="recipegen_gold",
        classification=DataClassification.PUBLIC,
        data_source=DataSource.RECIPEGEN,
        arm=args.arm,
        think=False,
    )
    print(json.dumps({
        "synthetic_only": True,
        "arm": args.arm,
        "protocol_failure": result.protocol_failure,
        "metrics": result.metrics(),
        "semantic_calls": len(result.calls),
    }, sort_keys=True))
    return 0 if result.compilation is not None and result.compilation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())

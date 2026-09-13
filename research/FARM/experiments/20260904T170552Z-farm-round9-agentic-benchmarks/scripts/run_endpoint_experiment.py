#!/usr/bin/env python3
"""Run or resume one frozen endpoint experiment without changing its ledger."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import ordered_ids_sha256, read_jsonl, sha256_file
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.endpoint_runner import ARMS, EndpointRun, run_endpoint_experiment
from farm_r9.ollama_client import OllamaChatClient
from farm_r9.privacy import DataClassification, DataSource


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="frozen candidates.jsonl")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--classification", choices=[item.value for item in DataClassification], required=True)
    parser.add_argument("--data-source", choices=[item.value for item in DataSource], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-reference", type=Path, help="optional private records.jsonl from another arm")
    parser.add_argument("--host", help="Ollama host; required for model arms")
    parser.add_argument("--model", help="Ollama model; required for model arms")
    parser.add_argument("--cloud", action="store_true", help="use Ollama Cloud (PUBLIC data only)")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY", help="environment variable, never its value")
    parser.add_argument("--lease-dir", type=Path, default=RUN_ROOT / "derived" / ".ollama-cloud-leases")
    parser.add_argument("--cache-dir", type=Path, help="private Ollama response cache")
    parser.add_argument("--journal", type=Path, help="private credential-free client journal")
    parser.add_argument("--prompt-cost-per-million-usd", type=float)
    parser.add_argument("--completion-cost-per-million-usd", type=float)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    classification = DataClassification(args.classification)
    if args.cloud and classification is not DataClassification.PUBLIC:
        raise SystemExit("refusing Cloud inference for non-PUBLIC data")
    cases = read_jsonl(args.candidates)
    if not cases:
        raise SystemExit("candidate artifact is empty")
    client = None
    model_metadata: dict[str, object] = {}
    protocol_metadata: dict[str, object] = {
        "arm": args.arm, "semantic_calls_max": 0 if args.arm == "retrieval_top1" else (1 if args.arm == "same_model_one_shot" else 2),
        "repairs_max": 0 if args.arm == "same_model_one_shot" else (1 if args.arm == "bounded_tool_agent" else 0),
        "temperature": 0, "seed": 42,
    }
    if args.arm != "retrieval_top1":
        if not args.host or not args.model:
            raise SystemExit("--host and --model are required for model arms")
        api_key = os.environ.get(args.api_key_env) if args.cloud else None
        if args.cloud and not api_key:
            raise SystemExit(f"Cloud run requires a nonempty {args.api_key_env} environment variable")
        cache_dir = args.cache_dir or args.output_dir / "ollama-cache"
        journal = args.journal or args.output_dir / "ollama-journal.jsonl"
        client = OllamaChatClient(
            host=args.host, api_key=api_key, model=args.model, cloud=args.cloud,
            limiter=OllamaCloudLimiter(args.lease_dir) if args.cloud else None,
            cache_directory=cache_dir, journal_path=journal,
        )
        model_metadata = {"name": args.model, "provider": "ollama_cloud" if args.cloud else "ollama_local"}
    run = EndpointRun(
        benchmark=args.benchmark, arm=args.arm, classification=classification,
        data_source=DataSource(args.data_source),
        output_directory=args.output_dir, candidate_artifact_sha256=sha256_file(args.candidates),
        ordered_case_ids_sha256=ordered_ids_sha256(cases), model_metadata=model_metadata,
        protocol_metadata=protocol_metadata,
        prompt_cost_per_million_usd=args.prompt_cost_per_million_usd,
        completion_cost_per_million_usd=args.completion_cost_per_million_usd,
    )
    reference = read_jsonl(args.paired_reference) if args.paired_reference else None
    result = run_endpoint_experiment(cases=cases, run=run, client=client, paired_reference_records=reference)
    print(json.dumps({
        "intended_n": result["intended_n"], "terminal_n": result["terminal_n"], "pending_n": result["pending_n"],
        "aggregate": str(args.output_dir / "aggregate_private.json"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Copy a frozen PUBLIC endpoint ledger and retry only unfinished transport cases.

No prediction, prompt, sample, or model policy is changed. Existing terminal
failures remain failures. Source artifacts are never modified; the destination
must not exist. All new Cloud requests use the existing shared limiter.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.artifact_io import read_json, read_jsonl, sha256_file, ordered_ids_sha256
from farm_r9.endpoint_runner import EndpointRun, run_endpoint_experiment
from farm_r9.ollama_client import OllamaChatClient
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.privacy import DataClassification, DataSource


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--lease-dir", required=True, type=Path)
    parser.add_argument("--host-env", default="OLLAMA_CLOUD_HOST_2")
    parser.add_argument("--key-env", default="OLLAMA_API_KEY_2")
    args = parser.parse_args()
    manifest = read_json(args.source / "manifest.json")
    cases = read_jsonl(args.candidates)
    records = read_jsonl(args.source / "records.jsonl")
    if manifest["data_classification"] != "public":
        raise SystemExit("recovery is restricted to public benchmarks")
    if manifest["arm"] != "same_model_one_shot":
        raise SystemExit("recovery requires the frozen one-shot arm")
    if sha256_file(args.candidates) != manifest["candidate_artifact_sha256"]:
        raise SystemExit("candidate hash mismatch")
    if ordered_ids_sha256(cases) != manifest["ordered_case_ids_sha256"]:
        raise SystemExit("candidate order mismatch")
    expected = {case["case_id"] for case in cases}
    if len(cases) != 150 or len(expected) != 150:
        raise SystemExit("requires frozen n=150")
    latest = {}
    for row in records:
        if row["case_id"] not in expected:
            raise SystemExit("unexpected ledger case")
        previous = latest.get(row["case_id"])
        if previous is None or row["attempt"] > previous["attempt"]:
            latest[row["case_id"]] = row
    if set(latest) != expected:
        raise SystemExit("source must already cover the full sample")
    pending = [row for row in latest.values() if not row.get("terminal")]
    if not pending or any(row.get("failure_code") != "OllamaTransportError" for row in pending):
        raise SystemExit("only unresolved transport failures may be retried")
    source_hashes = {name: sha256_file(args.source / name)
                     for name in ("records.jsonl", "manifest.json")}
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in source_hashes:
        shutil.copy2(args.source / name, args.output / name)
        os.chmod(args.output / name, 0o600)
    with (args.output / "recovery_provenance.json").open("x") as handle:
        json.dump({"source_hashes": source_hashes, "n": 150,
                   "pending_transport_cases": len(pending),
                   "policy": "retry_transport_only_preserve_terminal_predictions"}, handle, indent=2)
    os.chmod(args.output / "recovery_provenance.json", 0o600)
    client = OllamaChatClient(
        host=os.environ[args.host_env], api_key=os.environ[args.key_env],
        model=manifest["model_metadata"]["name"], cloud=True,
        cache_directory=args.output / "ollama-cache",
        journal_path=args.output / "ollama-journal.jsonl",
        limiter=OllamaCloudLimiter(args.lease_dir), timeout_seconds=900,
    )
    run = EndpointRun(
        benchmark=manifest["benchmark"], arm=manifest["arm"],
        classification=DataClassification.PUBLIC, data_source=DataSource(manifest["data_source"]),
        output_directory=args.output,
        candidate_artifact_sha256=manifest["candidate_artifact_sha256"],
        ordered_case_ids_sha256=manifest["ordered_case_ids_sha256"],
        model_metadata=manifest["model_metadata"], protocol_metadata=manifest["protocol_metadata"],
        prompt_cost_per_million_usd=manifest["pricing"]["prompt_cost_per_million_usd"],
        completion_cost_per_million_usd=manifest["pricing"]["completion_cost_per_million_usd"],
    )
    result = run_endpoint_experiment(cases=cases, run=run, client=client)
    if any(sha256_file(args.source / name) != value for name, value in source_hashes.items()):
        raise SystemExit("source ledger changed during recovery")
    print(json.dumps({key: result[key] for key in ("intended_n", "terminal_n", "pending_n")}))


if __name__ == "__main__":
    main()

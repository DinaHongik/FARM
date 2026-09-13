#!/usr/bin/env python3
"""Run/resume one frozen 150-case Yao arm through local or Cloud Ollama."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import ordered_ids_sha256, read_json, read_jsonl, sha256_file, sha256_text
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.ollama_client import OllamaChatClient
from farm_r9.privacy import DataClassification, DataSource, require_cloud_route
from farm_r9.yao_agent import ARMS, load_official_catalog, prompt_sha256, run_experiment, validate_final_cases


FROZEN_CASES_SHA256 = "5543d5e331033147a960392c474f5bb2b1560e2d9a7ca1942e8de8d06aa345ff"
FROZEN_MANIFEST_SHA256 = "c93a98c8d198df037011a0fcc3d3c3f31334b4e6b0741c01b2058e001673ab5e"
FROZEN_CONVERTED_SHA256 = "9cbab0a4711fe7c4e1ae9ab65c5963d60e60c79b59bbe956d75e95b489c3a25d"
FROZEN_ORDERED_IDS_SHA256 = "f0b147aa9b23686a1e9010e3135639ecda2a10edbac00949b5770d79bd1b1388"


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="prepared/interactive_ifttt/cases.jsonl")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True, help="pinned inert Yao JSON conversion")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paired-reference", type=Path, help="other arm's completed records.jsonl")
    parser.add_argument("--host", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cloud", action="store_true")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument("--lease-dir", type=Path, default=RUN_ROOT / "derived" / ".ollama-cloud-leases")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--prompt-cost-per-million-usd", type=float)
    parser.add_argument("--completion-cost-per-million-usd", type=float)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=240.0,
        help="per-provider-attempt timeout; operational only, not a semantic protocol change",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate frozen inputs/routing and require an absent output directory; never contact Ollama",
    )
    return parser.parse_args(argv)


def _validate_endpoint_without_io(host: str, *, cloud: bool) -> None:
    parsed = urlsplit(host)
    if cloud:
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise SystemExit("Cloud Ollama host must be a credential-free HTTPS origin")
        require_cloud_route(
            DataClassification.PUBLIC,
            benchmark_label="interactive_ifttt",
            source=DataSource.INTERACTIVE_IFTTT,
        )
    elif parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
        raise SystemExit("local Ollama host must be http://127.0.0.1:PORT")
    if parsed.query or parsed.fragment:
        raise SystemExit("Ollama host cannot contain query or fragment")


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    if not 30.0 <= args.timeout_seconds <= 1800.0:
        raise SystemExit("--timeout-seconds must be between 30 and 1800")
    cases = read_jsonl(args.cases)
    validate_final_cases(cases)
    sample_manifest = read_json(args.sample_manifest)
    if not isinstance(sample_manifest, dict):
        raise SystemExit("sample manifest must be an object")
    if sample_manifest.get("sample_size") != 150:
        raise SystemExit("sample manifest is not the frozen 150-case protocol")
    expected_strata = {"CI": 28, "VI-1": 18, "VI-2": 32, "VI-3": 10, "VI-4": 62}
    if sample_manifest.get("sample_strata") != expected_strata or sample_manifest.get("quotas") != expected_strata:
        raise SystemExit("sample manifest does not match the frozen ambiguity-stratum quotas")
    if sample_manifest.get("population_size") != 3870:
        raise SystemExit("sample manifest does not identify the official 3,870-case test split")
    if sample_manifest.get("source_commit") != "cd190229b0b6f237fd3534a297d138c2d834168d":
        raise SystemExit("sample manifest repository commit changed")
    if sample_manifest.get("source_artifact_sha256") != "3d47f7047808085c75826d90997f05498fae4747fd3d78e6130e93742806a535":
        raise SystemExit("sample manifest source artifact changed")
    case_sha256 = sha256_file(args.cases)
    manifest_sha256 = sha256_file(args.sample_manifest)
    converted_sha256 = sha256_file(args.converted)
    ordered_sha256 = ordered_ids_sha256(cases)
    if case_sha256 != FROZEN_CASES_SHA256:
        raise SystemExit("case payload differs from the preregistered final sample")
    if manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise SystemExit("sample manifest differs from the preregistered final manifest")
    if converted_sha256 != FROZEN_CONVERTED_SHA256:
        raise SystemExit("converted Yao artifact differs from the preregistered conversion")
    if ordered_sha256 != FROZEN_ORDERED_IDS_SHA256:
        raise SystemExit("case order differs from the preregistered final order")
    if sample_manifest.get("ordered_case_ids_sha256") != ordered_sha256:
        raise SystemExit("case order/hash disagrees with sample manifest")
    if sample_manifest.get("case_payload_sha256") != case_sha256:
        raise SystemExit("case payload hash disagrees with sample manifest")
    converted_metadata = sample_manifest.get("converted_file")
    if not isinstance(converted_metadata, dict) or converted_metadata.get("sha256") != converted_sha256:
        raise SystemExit("converted artifact hash disagrees with sample manifest")
    catalog = load_official_catalog(args.converted)
    api_key = os.environ.get(args.api_key_env) if args.cloud else None
    if args.cloud and not api_key:
        raise SystemExit(f"Cloud run requires nonempty {args.api_key_env}")
    _validate_endpoint_without_io(args.host, cloud=args.cloud)
    if args.preflight_only:
        if args.output_dir.exists():
            raise SystemExit("preflight requires a confirmed-absent output directory")
        print(json.dumps({
            "status": "ready_no_inference_performed",
            "arm": args.arm,
            "model": args.model,
            "n": len(cases),
            "case_sha256": case_sha256,
            "manifest_sha256": manifest_sha256,
            "converted_sha256": converted_sha256,
            "ordered_case_ids_sha256": ordered_sha256,
            "output_directory_absent": True,
            "cloud_route": "public_interactive_ifttt" if args.cloud else "local_loopback",
            "lease_directory": str(args.lease_dir.resolve()) if args.cloud else None,
        }, sort_keys=True))
        return 0
    cache_directory = args.cache_dir or args.output_dir / "ollama-cache"
    journal_path = args.journal or args.output_dir / "ollama-journal.jsonl"
    client = OllamaChatClient(
        host=args.host,
        api_key=api_key,
        model=args.model,
        cache_directory=cache_directory,
        journal_path=journal_path,
        cloud=args.cloud,
        limiter=OllamaCloudLimiter(args.lease_dir) if args.cloud else None,
        timeout_seconds=args.timeout_seconds,
    )
    catalog_hash = sha256_text(canonical_catalog(catalog))
    manifest = {
        "schema_version": "round9-yao-run-manifest-v1",
        "benchmark": "interactive_ifttt",
        "data_classification": "public",
        "data_source": "interactive_ifttt",
        "arm": args.arm,
        "n": 150,
        "sample_sha256": case_sha256,
        "sample_manifest_sha256": manifest_sha256,
        "ordered_case_ids_sha256": ordered_sha256,
        "source_artifact_sha256": sample_manifest.get("source_artifact_sha256"),
        "converted_artifact_sha256": converted_sha256,
        "catalog_sha256": catalog_hash,
        "prompt_sha256": prompt_sha256(args.arm),
        "model_metadata": {
            "name": args.model,
            "provider": "ollama_cloud" if args.cloud else "ollama_local",
        },
        "protocol_metadata": {
            "version": "round9-yao-v1",
            "arm": args.arm,
            "temperature": 0,
            "seed": 42,
            "questions_max": 0 if args.arm == "same_model_one_shot" else 4,
            "semantic_calls_max": 1 if args.arm == "same_model_one_shot" else 5,
            "repairs_max": 0,
            "failure_policy": "terminal_incorrect_no_fallback",
            "answer_policy": "frozen_official_pool_sha256_v1",
        },
        "pricing": {
            "prompt_cost_per_million_usd": args.prompt_cost_per_million_usd,
            "completion_cost_per_million_usd": args.completion_cost_per_million_usd,
        },
    }
    aggregate = run_experiment(
        cases=cases,
        catalog=catalog,
        arm=args.arm,
        client=client,
        output_directory=args.output_dir,
        immutable_manifest=manifest,
        paired_reference_path=args.paired_reference,
    )
    print(json.dumps({
        "n": aggregate["n"],
        "arm": args.arm,
        "joint_percent": aggregate["percentages"]["joint"],
        "questions_mean": aggregate["questions"]["mean_per_case"],
        "output": str(args.output_dir / "aggregate.json"),
    }, sort_keys=True))
    return 0


def canonical_catalog(catalog: dict[str, tuple[str, ...]]) -> str:
    return json.dumps(
        {key: list(catalog[key]) for key in sorted(catalog)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    raise SystemExit(main())

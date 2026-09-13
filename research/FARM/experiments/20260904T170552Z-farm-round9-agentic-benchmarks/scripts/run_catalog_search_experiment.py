#!/usr/bin/env python3
"""Paired public RecipeGen Noisy fixed-context vs search-enabled native agent."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.artifact_io import read_jsonl, read_json, sha256_file, canonical_json, sha256_text
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.ollama_client import OllamaChatClient, OllamaTransportError
from farm_r9.catalog_search_agent import ARMS, SYSTEM, PublicCatalogSearch, execute_search_case, tool_schemas
from farm_r9.endpoint_runner import wilson_95, exact_mcnemar
from farm_r9.paired_statistics import paired_delta_bootstrap
from build_external_candidates import make_catalog
from run_yao_native import exclusive_json


def summarize(records, manifest, path):
    if len(records) != 150 or len({row["case_id"] for row in records}) != 150:
        raise ValueError("final aggregate requires all 150 distinct cases")
    metrics = {key: sum(row["scores"][key] for row in records) for key in records[0]["scores"]}
    return {"n": 150, "raw_numerators": metrics, "raw_denominators": {key: 150 for key in metrics},
            "percentages": {key: 100 * value / 150 for key, value in metrics.items()},
            "confidence_intervals": {key: wilson_95(value, 150) for key, value in metrics.items()},
            "failure_counts": dict(Counter(row["failure_code"] for row in records if row["failure_code"])),
            "initial_joint_coverage": sum(row["initial_joint_coverage"] for row in records),
            "observed_joint_coverage": sum(row["observed_joint_coverage"] for row in records),
            "successful_outside_initial_top10": sum(row["scores"]["function_joint"] and not
                row["initial_joint_coverage"] for row in records),
            "search_calls_total": sum(row["search_calls"] for row in records),
            "usage_totals": {key: sum(row["usage"].get(key, 0) for row in records) for key in
                ("semantic_calls", "prompt_tokens", "completion_tokens", "physical_attempts", "cache_hits",
                 "provider_latency_ms", "queue_wait_ms")},
            "protocol": manifest, "records_sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-gpu", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.expected_gpu):
        raise SystemExit("expected one explicitly allocated GPU")
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 0):
        raise SystemExit("requires one V100")
    root = args.round9_root.resolve()
    candidates_path = root / "derived/recipegen_noisy/candidates.jsonl"
    candidate_manifest = read_json(root / "derived/recipegen_noisy/candidates.manifest.json")
    metadata = root / "sources/recipegen_metadata.csv"
    if sha256_file(candidates_path) != "34ba86f638024c4c9e834fa6020386006ec91f916dc039c92dcd95a867ec3009":
        raise SystemExit("frozen candidate binding mismatch")
    if sha256_file(metadata) != "8013c38bd8aa2b3839d2fba47910caa99c9d03e74502a69f1d4d7d7b3e97fb36":
        raise SystemExit("public catalog binding mismatch")
    cases = read_jsonl(candidates_path)
    if len(cases) != 150 or len({row["case_id"] for row in cases}) != 150:
        raise SystemExit("requires frozen 150-case sample")
    catalog = PublicCatalogSearch(make_catalog(metadata), bi_encoder=candidate_manifest["bi_encoder"],
                                  cross_encoder=candidate_manifest["cross_encoder"])
    if args.smoke:
        # Plumbing only: public metadata query, not a selected final-test case.
        from farm_r9.catalog_search_agent import SearchArguments
        row = catalog.by_side["trigger"][0]
        returned = catalog.search(SearchArguments(side="trigger", query=row["document"][:400], service=row["service"]))
        if not returned or any(item["side"] != "trigger" for item in returned):
            raise SystemExit("catalog search smoke failed")
        print(json.dumps({"status": "catalog_search_smoke_passed", "returned_count": len(returned),
                          "gpu": args.expected_gpu}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)
    manifest = {"protocol": "recipegen-noisy-native-search-v1-exploratory", "n": 150,
        "classification": "public", "model": "deepseek-v4-flash:0731", "temperature": 0, "seed": 42,
        "think": False, "num_predict": 8192, "arms": list(ARMS),
        "search_model_calls_max": 5, "search_tool_calls_max": 6, "fixed_model_calls_max": 1,
        "candidate_sha256": sha256_file(candidates_path), "metadata_sha256": sha256_file(metadata),
        "code_sha256": sha256_file(ROOT / "src/farm_r9/catalog_search_agent.py"),
        "prompt_and_tools_sha256": sha256_text(canonical_json({"system": SYSTEM, "tools": tool_schemas(can_search=True)})),
        "bi_encoder": candidate_manifest["bi_encoder"], "cross_encoder": candidate_manifest["cross_encoder"],
        "search_retrieve_k": 50, "search_return_k": 10, "rank_hidden": True,
        "field_metric": "metadata_field_names_after_endpoint_selection_not_values_or_bindings",
        "budget_matched": False, "novel_architecture_superiority_claim": False}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path) != manifest:
            raise SystemExit("existing search run has a different manifest")
    else:
        exclusive_json(manifest_path, manifest)
    clients, ledgers, completed = {}, {}, {}
    for arm in ARMS:
        out = args.output / arm
        out.mkdir(exist_ok=True, mode=0o700)
        ledgers[arm] = out / "records.jsonl"
        previous = read_jsonl(ledgers[arm]) if ledgers[arm].exists() else []
        completed[arm] = {row["case_id"] for row in previous}
        if len(completed[arm]) != len(previous) or not completed[arm] <= {row["case_id"] for row in cases}:
            raise SystemExit("search ledger has invalid cases")
        clients[arm] = OllamaChatClient(host=os.environ["OLLAMA_CLOUD_HOST_2"], api_key=os.environ["OLLAMA_API_KEY_2"],
            model=manifest["model"], cloud=True, limiter=OllamaCloudLimiter(root / "results/.ollama-cloud-leases-v1"),
            cache_directory=out / "ollama-cache", journal_path=out / "ollama-journal.jsonl", timeout_seconds=900)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for case in cases:
            for arm in ARMS:
                if case["case_id"] in completed[arm]:
                    continue
                for attempt in range(3):
                    try:
                        result = execute_search_case(case=case, arm=arm, catalog=catalog, client=clients[arm])
                        break
                    except OllamaTransportError:
                        if attempt == 2:
                            raise
                        time.sleep(5)
                with ledgers[arm].open("a", encoding="utf-8") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(canonical_json(result) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed[arm].add(case["case_id"])
                print(json.dumps({"arm": arm, "completed": len(completed[arm]), "n": 150}), flush=True)
        rows = {arm: read_jsonl(ledgers[arm]) for arm in ARMS}
        for arm in ARMS:
            output = args.output / arm / "aggregate.json"
            if not output.exists():
                exclusive_json(output, summarize(rows[arm], manifest, ledgers[arm]))
        if not (args.output / "paired_comparison.json").exists():
            baseline = {row["case_id"]: row for row in rows[ARMS[0]]}
            current, reference = [], []
            for row in rows[ARMS[1]]:
                current.append(row["scores"]["function_joint"])
                reference.append(baseline[row["case_id"]]["scores"]["function_joint"])
            rescues = sum(a and not b for a, b in zip(current, reference))
            regressions = sum(b and not a for a, b in zip(current, reference))
            exclusive_json(args.output / "paired_comparison.json", {
                "n": 150, "metric": "function_joint", "rescues": rescues, "regressions": regressions,
                "paired_delta": paired_delta_bootstrap(current, reference, expected_n=150),
                "mcnemar_exact_p": exact_mcnemar(rescues, regressions), "budget_matched": False})


if __name__ == "__main__":
    main()

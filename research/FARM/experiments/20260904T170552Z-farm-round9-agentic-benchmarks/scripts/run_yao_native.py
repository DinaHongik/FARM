#!/usr/bin/env python3
"""Frozen n=150 native-tool Yao controls; public-only, single worker, resumable."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.artifact_io import read_jsonl, read_json, sha256_file, sha256_text, canonical_json
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.ollama_client import OllamaChatClient, OllamaTransportError
from farm_r9.yao_agent import load_official_catalog, validate_final_cases, aggregate_records, COMPONENTS
from farm_r9.yao_native import ARMS, SYSTEM, MAX_OUTPUT_TOKENS, execute_native_case, tools_for
from run_yao_experiment import FROZEN_CASES_SHA256, FROZEN_MANIFEST_SHA256, FROZEN_CONVERTED_SHA256


def exclusive_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="deepseek-v4-flash:0731")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.round9_root.resolve()
    paths = {"cases": root / "prepared/interactive_ifttt/cases.jsonl",
             "sample_manifest": root / "manifests/samples/interactive_ifttt.json",
             "converted": root / "staged/yao-worker-v3-timeout-20260904T182000Z/source/yao-converted.json"}
    for key, expected in (("cases", FROZEN_CASES_SHA256), ("sample_manifest", FROZEN_MANIFEST_SHA256),
                          ("converted", FROZEN_CONVERTED_SHA256)):
        if sha256_file(paths[key]) != expected:
            raise SystemExit("frozen public source hash mismatch: " + key)
    cases = read_jsonl(paths["cases"])
    validate_final_cases(cases)
    catalog = load_official_catalog(paths["converted"])
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)
    if args.smoke:
        # Synthetic plumbing case, constructed from catalog metadata only.
        # No final-test query, gold, or simulator answer enters this request.
        labels = {key: catalog[key][0] for key in COMPONENTS}
        cases = [{"case_id": "synthetic-native-plumbing-v1", "stratum": "synthetic",
                  "input": {"query": "Use exactly these endpoints: " + canonical_json(labels)},
                  "private_gold": labels, "simulator": {}}]
    manifest = {"protocol": "yao-native-v3-exploratory", "n": len(cases), "model": args.model,
                "temperature": 0, "seed": 42, "think": False, "num_predict": MAX_OUTPUT_TOKENS,
                "arms": [ARMS[0]] if args.smoke else list(ARMS),
                "sample_sha256": sha256_file(paths["cases"]) if not args.smoke else None,
                "catalog_sha256": sha256_text(canonical_json(catalog)),
                "prompt_and_tools_sha256": sha256_text(canonical_json({"system": SYSTEM,
                    "tools": tools_for(catalog, can_ask=True)})),
                "code_sha256": sha256_file(ROOT / "src/farm_r9/yao_native.py"),
                "classification": "public", "source": "interactive_ifttt",
                "scope": "endpoint_clarification_not_executable_configuration",
                "adaptive_questions_max": 4, "adaptive_model_calls_max": 5,
                "baseline_model_calls_max": 1, "ask_all_questions": 4}
    path = args.output / "manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise SystemExit("existing native run has a different manifest")
    else:
        exclusive_json(path, manifest)
    with (args.output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for arm in manifest["arms"]:
            out = args.output / arm
            out.mkdir(exist_ok=True, mode=0o700)
            records_path = out / "records.jsonl"
            previous = read_jsonl(records_path) if records_path.exists() else []
            ids = {row["case_id"] for row in previous}
            if len(ids) != len(previous) or not ids <= {case["case_id"] for case in cases}:
                raise SystemExit("native ledger case IDs differ")
            client = OllamaChatClient(host=os.environ["OLLAMA_CLOUD_HOST_2"],
                api_key=os.environ["OLLAMA_API_KEY_2"], model=args.model, cloud=True,
                limiter=OllamaCloudLimiter(root / "results/.ollama-cloud-leases-v1"),
                cache_directory=out / "ollama-cache", journal_path=out / "ollama-journal.jsonl",
                timeout_seconds=900)
            for case in cases:
                if case["case_id"] in ids:
                    continue
                for attempt in range(3):
                    try:
                        row = execute_native_case(case=case, catalog=catalog, arm=arm, client=client)
                        break
                    except OllamaTransportError:
                        if attempt == 2:
                            raise
                        time.sleep(5)
                with records_path.open("a", encoding="utf-8") as handle:
                    os.fchmod(handle.fileno(), 0o600)
                    handle.write(canonical_json(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                ids.add(case["case_id"])
                print(json.dumps({"arm": arm, "completed": len(ids), "n": len(cases)}), flush=True)
                if args.smoke and row["terminal_status"] != "committed":
                    raise SystemExit("native smoke failed: " + str(row["failure_code"]))
            records = read_jsonl(records_path)
            aggregate = aggregate_records(records, intended_n=len(cases))
            aggregate["interaction_turns_total"] = sum(row["interaction_turns"] for row in records)
            aggregate["hashes"] = {"records_sha256": sha256_file(records_path),
                                   "manifest_sha256": sha256_file(path)}
            aggregate["protocol"] = manifest
            if not (out / "aggregate.json").exists():
                exclusive_json(out / "aggregate.json", aggregate)
        if not args.smoke and not (args.output / "paired_comparisons.json").exists():
            adaptive = read_jsonl(args.output / "native_adaptive/records.jsonl")
            comparisons = {}
            for control in ("native_one_shot", "native_ask_all"):
                reference = read_jsonl(args.output / control / "records.jsonl")
                paired = aggregate_records(adaptive, intended_n=150, paired_reference=reference)
                comparisons[control] = {"paired_outcomes": paired["paired_outcomes"],
                                       "reference_questions_total": sum(r["question_count"] for r in reference),
                                       "adaptive_questions_total": sum(r["question_count"] for r in adaptive)}
            exclusive_json(args.output / "paired_comparisons.json", comparisons)


if __name__ == "__main__":
    main()

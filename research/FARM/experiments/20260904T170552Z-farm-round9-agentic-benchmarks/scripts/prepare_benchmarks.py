#!/usr/bin/env python3
"""Freeze Round 9's local public-benchmark samples and provenance."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.adapters.farm_v2 import prepare_farm_v2
from farm_r9.adapters.recipegen import prepare_recipegen
from farm_r9.adapters.targe import prepare_targe
from farm_r9.artifact_io import ordered_ids_sha256, read_json, sha256_file, write_json_atomic, write_jsonl_atomic


def persist(benchmark: str, cases: list[dict], manifest: dict) -> dict:
    if len(cases) != 150:
        raise ValueError(f"{benchmark}: expected exactly 150 cases, got {len(cases)}")
    case_path = RUN_ROOT / "prepared" / benchmark / "cases.jsonl"
    manifest_path = RUN_ROOT / "manifests" / "samples" / f"{benchmark}.json"
    manifest = dict(manifest)
    manifest.update({
        "schema_version": "round9-sample-manifest-v1",
        "case_file": str(case_path.relative_to(RUN_ROOT)),
        "ordered_case_ids_sha256": ordered_ids_sha256(cases),
    })
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in cases)
    import hashlib
    manifest["case_payload_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    if case_path.exists() or manifest_path.exists():
        if not (case_path.exists() and manifest_path.exists()):
            raise RuntimeError(f"{benchmark}: partial frozen sample exists")
        existing = read_json(manifest_path)
        if existing != manifest or sha256_file(case_path) != manifest["case_payload_sha256"]:
            raise RuntimeError(f"{benchmark}: refusing to overwrite a different frozen sample")
        return manifest
    write_jsonl_atomic(case_path, cases)
    write_json_atomic(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=RUN_ROOT.parents[1])
    parser.add_argument("--seed", type=int, default=9052026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    outputs = {}
    cases, manifest = prepare_farm_v2(repo_root / "data" / "v2", size=150, seed=args.seed)
    outputs["farm_v2_test"] = persist("farm_v2_test", cases, manifest)
    for split in ("gold", "noisy"):
        benchmark = f"recipegen_{split}"
        cases, manifest = prepare_recipegen(
            repo_root / "RecipeGen" / "dataset" / "processed.csv", split=split, size=150, seed=args.seed,
        )
        outputs[benchmark] = persist(benchmark, cases, manifest)
    for split in ("gold", "noisy", "one_shot"):
        benchmark = f"targe_{split}"
        cases, manifest = prepare_targe(
            repo_root / "Targe" / "data" / "dataset", split=split, size=150, seed=args.seed,
        )
        outputs[benchmark] = persist(benchmark, cases, manifest)
    write_json_atomic(RUN_ROOT / "DATA_PROVENANCE.json", {
        "run_id": RUN_ROOT.name,
        "sample_size_per_benchmark": 150,
        "samples": outputs,
        "interactive_ifttt": {"status": "pending official artifact conversion"},
    })
    print(json.dumps({key: {"n": value["sample_size"], "hash": value["ordered_case_ids_sha256"]} for key, value in outputs.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

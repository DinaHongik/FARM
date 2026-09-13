#!/usr/bin/env python3
"""Generate one frozen E3 seed ranking for the confidential FARM sample."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import read_json, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"


def assert_gpu(expected: int) -> None:
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible != [str(expected)]:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={expected}, got {visible}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("candidate worker requires exactly one visible CUDA device")
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("candidate worker is pinned to V100 sm_70")


def rank(model_path: Path, queries: list[str], corpus: list[dict], top_k: int) -> list[list[str]]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    config = read_json(model_path / "config.json")
    extra = {"model_kwargs": {"torch_dtype": torch.float32}}
    if config.get("model_type") == "gemma3_text":
        extra["processor_kwargs"] = {"fix_mistral_regex": False}
    model = SentenceTransformer(str(model_path), device="cuda:0", **extra)
    if "query" not in model.prompts or "document" not in model.prompts:
        raise RuntimeError("FARM E3 checkpoint lacks query/document prompts")
    documents = model.encode_document(
        [row["text_schema"] for row in corpus], batch_size=64, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    query_vectors = model.encode_query(
        queries, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )
    if not np.isfinite(documents).all() or not np.isfinite(query_vectors).all():
        raise RuntimeError("non-finite embedding")
    similarities = query_vectors @ documents.T
    output = []
    for row_index in range(len(queries)):
        order = sorted(range(len(corpus)), key=lambda index: (-float(similarities[row_index, index]), corpus[index]["url"]))[:top_k]
        output.append([corpus[index]["url"] for index in order])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--trigger-model", type=Path, required=True)
    parser.add_argument("--action-model", type=Path, required=True)
    parser.add_argument("--trigger-sha256", required=True)
    parser.add_argument("--action-sha256", required=True)
    parser.add_argument("--seed-id", required=True)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assert_gpu(args.expected_gpu)
    data_root = args.data_root.resolve()
    manifest = read_json(data_root / "manifest.json")
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("Dataset-v2 identity mismatch")
    sample_manifest = read_json(RUN_ROOT / "manifests" / "samples" / "farm_v2_test.json")
    cases = read_jsonl(RUN_ROOT / "prepared" / "farm_v2_test" / "cases.jsonl")
    if len(cases) != 150 or sample_manifest["sample_size"] != 150:
        raise RuntimeError("FARM sample must contain exactly 150 cases")
    for path, expected in ((args.trigger_model, args.trigger_sha256), (args.action_model, args.action_sha256)):
        actual = sha256_file(path / "model.safetensors")
        if actual != expected:
            raise RuntimeError(f"checkpoint hash mismatch for {path}")
    corpora = {
        "trigger": read_json(data_root / "corpus" / "triggers.json"),
        "action": read_json(data_root / "corpus" / "actions.json"),
    }
    queries = [case["input"]["query"] for case in cases]
    rankings = {
        "trigger": rank(args.trigger_model, queries, corpora["trigger"], args.top_k),
        "action": rank(args.action_model, queries, corpora["action"], args.top_k),
    }
    rows = [
        {"case_id": case["case_id"], "trigger_ranking": rankings["trigger"][index], "action_ranking": rankings["action"][index]}
        for index, case in enumerate(cases)
    ]
    output = RUN_ROOT / "derived" / "farm_v2_test" / "seeds" / f"{args.seed_id}.jsonl"
    output_manifest = output.with_suffix(".manifest.json")
    if output.exists() or output_manifest.exists():
        raise RuntimeError(f"refusing to overwrite seed artifact: {output}")
    write_jsonl_atomic(output, rows)
    write_json_atomic(output_manifest, {
        "schema_version": "round9-farm-seed-v1", "run_id": RUN_ROOT.name,
        "dataset_id": DATASET_ID, "data_classification": "confidential",
        "sample_manifest_sha256": sha256_file(RUN_ROOT / "manifests" / "samples" / "farm_v2_test.json"),
        "case_ids_sha256": sample_manifest["ordered_case_ids_sha256"],
        "seed_id": args.seed_id, "physical_gpu": args.expected_gpu, "top_k": args.top_k,
        "checkpoints": {
            "trigger": {"path": str(args.trigger_model.resolve()), "sha256": args.trigger_sha256},
            "action": {"path": str(args.action_model.resolve()), "sha256": args.action_sha256},
        },
        "output": str(output.relative_to(RUN_ROOT)), "output_sha256": sha256_file(output), "rows": len(rows),
    })
    print(json.dumps({"seed": args.seed_id, "rows": len(rows), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

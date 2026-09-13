#!/usr/bin/env python3
"""Freeze a deterministic dev-only candidate file for LLM/agent comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    CORPUS_FILE,
    assert_one_visible_gpu,
    assert_split_allowed,
    read_json,
    saved_model_kwargs,
    sha256_file,
    utc_now,
    validate_dataset,
    write_json,
)


def load_model(name_or_path: str, revision: str, torch):
    from sentence_transformers import SentenceTransformer
    path = Path(name_or_path)
    kwargs = saved_model_kwargs(path) if path.is_dir() else {"revision": revision}
    return SentenceTransformer(
        name_or_path,
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.float32},
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--trigger-model", default="google/embeddinggemma-300m")
    parser.add_argument("--action-model", default="google/embeddinggemma-300m")
    parser.add_argument("--model-revision", default="57c266a740f537b4dc058e1b0cda161fd15afa75")
    parser.add_argument("--subset-size", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    physical_gpu = assert_one_visible_gpu(4)
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    manifest = validate_dataset(data_root)
    assert_split_allowed("dev", "eval")

    import numpy as np
    import torch

    groups = read_json(data_root / "splits" / "dev.json")
    groups.sort(key=lambda row: (hashlib.sha256(row["group_id"].encode()).hexdigest(), row["group_id"]))
    groups = groups[: args.subset_size]
    top_by_side = {}
    for kind, model_name in (("trigger", args.trigger_model), ("action", args.action_model)):
        model = load_model(model_name, args.model_revision, torch)
        corpus = read_json(data_root / "corpus" / CORPUS_FILE[kind])
        texts = [row["text_schema"] for row in corpus]
        embeddings = model.encode_document(
            texts, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        queries = model.encode_query(
            [row["query"] for row in groups], batch_size=64,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        if not np.isfinite(embeddings).all() or not np.isfinite(queries).all():
            raise RuntimeError(f"non-finite {kind} candidate embeddings")
        scores = queries @ embeddings.T
        top_rows = []
        for group_index in range(len(groups)):
            order = sorted(
                range(len(corpus)),
                key=lambda index: (-float(scores[group_index, index]), corpus[index]["url"]),
            )[: args.top_k]
            top_rows.append([
                {
                    "url": corpus[index]["url"],
                    "channel": corpus[index]["channel"],
                    "function_name": corpus[index]["function_name"],
                    "text_schema": corpus[index]["text_schema"],
                    "retrieval_score": float(scores[group_index, index]),
                    "retrieval_rank": rank + 1,
                }
                for rank, index in enumerate(order)
            ])
        top_by_side[kind] = top_rows
        del model, embeddings, queries, scores
        torch.cuda.empty_cache()

    rows = []
    for index, group in enumerate(groups):
        rows.append({
            "group_id": group["group_id"],
            "query": group["query"],
            "valid_pairs": group["valid_pairs"],
            "trigger_candidates": top_by_side["trigger"][index],
            "action_candidates": top_by_side["action"][index],
        })
    output = args.output or run_root / "derived_data" / "agent" / f"dev{len(rows)}_candidates.json"
    output = output.resolve()
    write_json(output, rows)
    result = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": manifest["dataset_id"],
        "split": "dev",
        "selection": "lowest SHA-256(group_id), deterministic",
        "rows": len(rows),
        "top_k": args.top_k,
        "trigger_model": args.trigger_model,
        "action_model": args.action_model,
        "base_model_revision": args.model_revision,
        "physical_gpu": int(physical_gpu),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "candidate_ceiling_joint_at_k": sum(
            any(
                pair["trigger_url"] in {x["url"] for x in row["trigger_candidates"]}
                and pair["action_url"] in {x["url"] for x in row["action_candidates"]}
                for pair in row["valid_pairs"]
            )
            for row in rows
        ) / len(rows),
    }
    write_json(output.with_suffix(".manifest.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

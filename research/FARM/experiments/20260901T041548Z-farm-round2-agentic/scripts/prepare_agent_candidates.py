#!/usr/bin/env python3
"""Freeze exact E3 candidate lists for calibration and Ollama comparisons."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    CORPUS_FILE,
    assert_one_visible_gpu,
    read_json,
    saved_model_kwargs,
    sha256_file,
    utc_now,
    validate_dataset,
    write_json,
)


def load_model(path: Path, torch):
    from sentence_transformers import SentenceTransformer
    if not path.is_dir():
        raise RuntimeError(f"missing trained E3 model: {path}")
    model = SentenceTransformer(
        str(path), device="cuda:0", model_kwargs={"torch_dtype": torch.float32},
        **saved_model_kwargs(path),
    )
    if "query" not in model.prompts or "document" not in model.prompts:
        raise RuntimeError("E3 checkpoint lacks query/document prompts")
    return model


def checkpoint_hashes(path: Path) -> dict[str, str]:
    required = ("config.json", "model.safetensors")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"E3 checkpoint lacks files: {missing}")
    return {name: sha256_file(path / name) for name in required}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("reranker_train", "dev"), required=True)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--trigger-model", type=Path, required=True)
    parser.add_argument("--action-model", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-joint-r1", type=float)
    parser.add_argument("--expected-joint-r10", type=float)
    args = parser.parse_args()
    if args.top_k < 2:
        raise ValueError("top-k must be at least two for routing margins")
    if args.max_rows < 0:
        raise ValueError("max-rows cannot be negative")
    physical_gpu = assert_one_visible_gpu(args.expected_gpu)
    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    dataset = validate_dataset(data_root)
    groups = read_json(data_root / "splits" / f"{args.split}.json")
    if args.max_rows:
        groups = groups[: args.max_rows]
    if not groups:
        raise RuntimeError("empty candidate split")

    import numpy as np
    import torch

    models = {"trigger": args.trigger_model.resolve(), "action": args.action_model.resolve()}
    checkpoint_records = {side: checkpoint_hashes(path) for side, path in models.items()}
    top_by_side: dict[str, list[list[dict]]] = {}
    for side in ("trigger", "action"):
        model = load_model(models[side], torch)
        corpus = read_json(data_root / "corpus" / CORPUS_FILE[side])
        documents = [row["text_schema"] for row in corpus]
        doc_embeddings = model.encode_document(
            documents, batch_size=64, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        query_embeddings = model.encode_query(
            [group["query"] for group in groups], batch_size=64,
            normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
        )
        if not np.isfinite(doc_embeddings).all() or not np.isfinite(query_embeddings).all():
            raise RuntimeError(f"non-finite {side} E3 embeddings")
        scores = query_embeddings @ doc_embeddings.T
        rows = []
        for group_index in range(len(groups)):
            order = sorted(
                range(len(corpus)),
                key=lambda index: (-float(scores[group_index, index]), corpus[index]["url"]),
            )[: args.top_k]
            candidates = []
            seen_urls = set()
            for rank, index in enumerate(order, start=1):
                item = corpus[index]
                if item["url"] in seen_urls:
                    raise AssertionError("duplicate candidate URL")
                seen_urls.add(item["url"])
                score = float(scores[group_index, index])
                if not math.isfinite(score):
                    raise RuntimeError("non-finite candidate score")
                candidates.append({
                    "url": item["url"],
                    "channel": item["channel"],
                    "function_name": item["function_name"],
                    "text_plain": item["text_plain"],
                    "text_schema": item["text_schema"],
                    "retrieval_score": score,
                    "retrieval_rank": rank,
                })
            rows.append(candidates)
        top_by_side[side] = rows
        del model, doc_embeddings, query_embeddings, scores
        torch.cuda.empty_cache()

    output_rows = []
    r1_hits, rk_hits = [], []
    for index, group in enumerate(groups):
        triggers = top_by_side["trigger"][index]
        actions = top_by_side["action"][index]
        valid_pairs = {(pair["trigger_url"], pair["action_url"]) for pair in group["valid_pairs"]}
        top_pair = (triggers[0]["url"], actions[0]["url"])
        r1_hits.append(int(top_pair in valid_pairs))
        trigger_ids = {row["url"] for row in triggers}
        action_ids = {row["url"] for row in actions}
        rk_hits.append(int(any(t in trigger_ids and a in action_ids for t, a in valid_pairs)))
        output_rows.append({
            "group_id": group["group_id"],
            "query": group["query"],
            "valid_pairs": group["valid_pairs"],
            "trigger_candidates": triggers,
            "action_candidates": actions,
            "routing_confidence": min(
                triggers[0]["retrieval_score"] - triggers[1]["retrieval_score"],
                actions[0]["retrieval_score"] - actions[1]["retrieval_score"],
            ),
        })
    metrics = {
        "joint_R@1": sum(r1_hits) / len(r1_hits),
        f"joint_R@{args.top_k}": sum(rk_hits) / len(rk_hits),
        "joint_R@1_hits": r1_hits,
        f"joint_R@{args.top_k}_hits": rk_hits,
    }
    if args.expected_joint_r1 is not None and not math.isclose(
        metrics["joint_R@1"], args.expected_joint_r1, abs_tol=1e-12
    ):
        raise RuntimeError(f"E3 joint R@1 reproduction failed: {metrics['joint_R@1']}")
    if args.expected_joint_r10 is not None and not math.isclose(
        metrics[f"joint_R@{args.top_k}"], args.expected_joint_r10, abs_tol=1e-12
    ):
        raise RuntimeError(f"E3 joint R@10 reproduction failed: {metrics[f'joint_R@{args.top_k}']}")

    output = args.output.resolve()
    write_json(output, output_rows)
    manifest = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": dataset["dataset_id"],
        "split": args.split,
        "rows": len(output_rows),
        "top_k": args.top_k,
        "identity": "exact_function_url",
        "pair_policy": "observed valid pairs only",
        "candidate_order": "score descending, exact URL tie-break",
        "physical_gpu": int(physical_gpu),
        "models": {side: str(path) for side, path in models.items()},
        "checkpoint_hashes": checkpoint_records,
        "metrics": metrics,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "metrics"} | {
        "metrics": {key: value for key, value in metrics.items() if not key.endswith("_hits")}
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

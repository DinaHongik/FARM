#!/usr/bin/env python3
"""Build a hash-bound, three-seed RRF candidate lattice from Dataset v2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from experiment_core import DATASET_ID, read_json, sha256_file, write_json


def rrf_fuse(rankings: Iterable[Iterable[str]], *, constant: int = 60) -> list[tuple[str, float, dict[int, int]]]:
    """Fuse exact IDs; output order is independent of input retriever order."""
    if constant <= 0:
        raise ValueError("RRF constant must be positive")
    score: dict[str, float] = {}
    observed: dict[str, list[int]] = {}
    normalized = sorted(tuple(ranking) for ranking in rankings)
    for ranking_index, ranking in enumerate(normalized):
        if len(set(ranking)) != len(ranking):
            raise ValueError("one retriever ranking contains duplicate IDs")
        for rank, identifier in enumerate(ranking, start=1):
            score[identifier] = score.get(identifier, 0.0) + 1.0 / (constant + rank)
            observed.setdefault(identifier, []).append(rank)
    return [
        (identifier, score[identifier], {index: rank for index, rank in enumerate(observed[identifier])})
        for identifier in sorted(score, key=lambda value: (-score[value], value))
    ]


def assert_gpu(expected: int) -> None:
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible != [str(expected)]:
        raise RuntimeError(f"expected only physical GPU {expected}, got {visible}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("candidate generation requires exactly one visible GPU")


def saved_model_kwargs(path: Path) -> dict:
    config = read_json(path / "config.json")
    return {"processor_kwargs": {"fix_mistral_regex": False}} if config.get("model_type") == "gemma3_text" else {}


def validate_data(data_root: Path) -> dict:
    manifest = read_json(data_root / "manifest.json")
    if manifest.get("dataset_id") != DATASET_ID:
        raise RuntimeError("Dataset v2 identity mismatch")
    for relative, metadata in manifest["artifacts"].items():
        path = data_root / relative
        if relative == "splits/test.json":
            if not path.is_file():
                raise RuntimeError("locked test identity file is absent")
            continue
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise RuntimeError(f"Dataset v2 hash mismatch: {relative}")
    return manifest


def rank_with_model(model_path: Path, groups: list[dict], corpus: list[dict], top_k: int) -> list[list[str]]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        str(model_path), device="cuda:0", model_kwargs={"torch_dtype": torch.float32},
        **saved_model_kwargs(model_path),
    )
    if "query" not in model.prompts or "document" not in model.prompts:
        raise RuntimeError("E3 checkpoint lacks query/document prompts")
    documents = model.encode_document(
        [row["text_schema"] for row in corpus], batch_size=64,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )
    queries = model.encode_query(
        [row["query"] for row in groups], batch_size=64,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        raise RuntimeError("non-finite E3 embeddings")
    similarities = queries @ documents.T
    result = []
    for row_index in range(len(groups)):
        order = sorted(
            range(len(corpus)),
            key=lambda index: (-float(similarities[row_index, index]), corpus[index]["url"]),
        )[:top_k]
        result.append([corpus[index]["url"] for index in order])
    del model, documents, queries
    torch.cuda.empty_cache()
    return result


def metrics(rows: list[dict], depth: int) -> dict:
    cutoffs = (1, 5, depth)
    output = {}
    for cutoff in cutoffs:
        trigger, action, joint = 0, 0, 0
        for row in rows:
            valid = {(item["trigger_url"], item["action_url"]) for item in row["valid_pairs"]}
            trigger_ids = {item["url"] for item in row["trigger_candidates"][:cutoff]}
            action_ids = {item["url"] for item in row["action_candidates"][:cutoff]}
            trigger += int(any(t in trigger_ids for t, _ in valid))
            action += int(any(a in action_ids for _, a in valid))
            joint += int(any(t in trigger_ids and a in action_ids for t, a in valid))
        output[f"trigger_R@{cutoff}"] = trigger / len(rows)
        output[f"action_R@{cutoff}"] = action / len(rows)
        output[f"joint_independent_R@{cutoff}"] = joint / len(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--retriever-depth", type=int, default=10)
    parser.add_argument("--fused-depth", type=int, default=10)
    parser.add_argument("--rrf-constant", type=int, default=60)
    args = parser.parse_args()
    if args.fused_depth > args.retriever_depth:
        raise ValueError("fused depth cannot exceed each retriever depth")
    assert_gpu(args.expected_gpu)
    run_root, data_root = args.run_root.resolve(), args.data_root.resolve()
    data_manifest = validate_data(data_root)
    spec = read_json(args.spec.resolve())
    retrievers = spec.get("retrievers") or []
    if len(retrievers) < 2:
        raise ValueError("RRF requires at least two retrievers")

    checkpoint_binding = {}
    for retriever in retrievers:
        seed = str(retriever["id"])
        checkpoint_binding[seed] = {}
        for side in ("trigger", "action"):
            path = Path(retriever[side]).resolve()
            actual = sha256_file(path / "model.safetensors")
            expected = retriever[f"{side}_model_sha256"]
            if actual != expected:
                raise RuntimeError(f"checkpoint hash mismatch: {seed}/{side}")
            checkpoint_binding[seed][side] = {"path": str(path), "model_sha256": actual}

    splits = {name: read_json(data_root / "splits" / f"{name}.json") for name in ("reranker_train", "dev")}
    combined = splits["reranker_train"] + splits["dev"]
    corpora = {
        "trigger": read_json(data_root / "corpus" / "triggers.json"),
        "action": read_json(data_root / "corpus" / "actions.json"),
    }
    corpus_maps = {side: {row["url"]: row for row in rows} for side, rows in corpora.items()}
    rankings: dict[str, list[list[list[str]]]] = {side: [] for side in ("trigger", "action")}
    for retriever in retrievers:
        for side in ("trigger", "action"):
            rankings[side].append(rank_with_model(
                Path(retriever[side]).resolve(), combined, corpora[side], args.retriever_depth,
            ))

    offset = 0
    output_root = run_root / "derived_data" / "function_rrf"
    for split_name, groups in splits.items():
        output_rows = []
        for local_index, group in enumerate(groups):
            combined_index = offset + local_index
            side_values = {}
            union_values = {}
            for side in ("trigger", "action"):
                seed_rankings = [seed_rows[combined_index] for seed_rows in rankings[side]]
                fused = rrf_fuse(seed_rankings, constant=args.rrf_constant)
                candidates = []
                union = []
                for fused_rank, (identifier, score, _) in enumerate(fused, start=1):
                    source = corpus_maps[side][identifier]
                    record = {
                        "url": identifier,
                        "channel": source["channel"],
                        "function_name": source["function_name"],
                        "text_plain": source["text_plain"],
                        "text_schema": source["text_schema"],
                        "retrieval_score": score,
                        "retrieval_rank": fused_rank,
                        "seed_ranks": {
                            str(retrievers[index]["id"]): (
                                seed_rankings[index].index(identifier) + 1 if identifier in seed_rankings[index] else None
                            )
                            for index in range(len(retrievers))
                        },
                    }
                    union.append(record)
                    if fused_rank <= args.fused_depth:
                        candidates.append(record)
                if len({row["url"] for row in candidates}) != len(candidates):
                    raise AssertionError("fused top-k has duplicate URLs")
                side_values[f"{side}_candidates"] = candidates
                union_values[f"{side}_union_candidates"] = union
            output_rows.append({
                "group_id": group["group_id"], "query": group["query"],
                "valid_pairs": group["valid_pairs"], **side_values, **union_values,
                "routing_confidence": min(
                    side_values["trigger_candidates"][0]["retrieval_score"] - side_values["trigger_candidates"][1]["retrieval_score"],
                    side_values["action_candidates"][0]["retrieval_score"] - side_values["action_candidates"][1]["retrieval_score"],
                ),
            })
        offset += len(groups)
        output = output_root / f"{split_name}.json"
        if output.exists():
            raise RuntimeError(f"refusing to overwrite frozen lattice: {output}")
        write_json(output, output_rows)
        manifest = {
            "status": "passed", "dataset_id": DATASET_ID, "split": split_name,
            "rows": len(output_rows), "task_level": "function", "view": "schema-hydrated",
            "retriever_depth": args.retriever_depth, "top_k": args.fused_depth,
            "rrf_constant": args.rrf_constant, "identity": "exact function URL",
            "pair_policy": "observed valid_pairs only", "gold_injection": False,
            "checkpoint_binding": checkpoint_binding,
            "dataset_manifest_sha256": sha256_file(data_root / "manifest.json"),
            "dataset_source_sha256": data_manifest["source"]["sha256"],
            "metrics": metrics(output_rows, args.fused_depth),
            "output": str(output), "output_sha256": sha256_file(output),
        }
        write_json(output.with_suffix(".manifest.json"), manifest)
        print(json.dumps({"split": split_name, "rows": len(output_rows), "metrics": manifest["metrics"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

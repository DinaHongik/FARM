#!/usr/bin/env python3
"""Exact-identity, pair-preserving dev evaluator for E0-E3."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    CORPUS_FILE,
    assert_one_visible_gpu,
    assert_split_allowed,
    config_from_path,
    read_json,
    saved_model_kwargs,
    sha256_file,
    utc_now,
    validate_dataset,
    validate_service_derived,
    write_json,
)


def ndcg_at_k(ranks: list[int], relevant_count: int, cutoff: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 2) for rank in ranks if rank < cutoff)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(relevant_count, cutoff)))
    return dcg / ideal if ideal else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--scope", choices=("full", "smoke"), default="full")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--split", choices=("dev",), default="dev")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    data_root = args.data_root.resolve()
    config = config_from_path(args.config.resolve())
    physical_gpu = assert_one_visible_gpu(config["gpu"])
    assert_split_allowed(args.split, "eval")
    dataset_manifest = validate_dataset(data_root)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    scope_root = run_root if args.scope == "full" else run_root / "smoke"
    if config["level"] == "service":
        validate_service_derived(run_root, data_root)
        groups = read_json(run_root / "derived_data" / "e0_service" / "dev.json")
    else:
        groups = read_json(data_root / "splits" / "dev.json")
    if args.max_rows:
        groups = groups[: args.max_rows]
    if not groups:
        raise RuntimeError("empty dev evaluation")

    per_side: dict[str, dict] = {}
    all_ranks: dict[str, list[dict[str, int]]] = {}
    all_top_ids: dict[str, list[list[str]]] = {}
    side_elapsed: dict[str, float] = {}
    for kind in ("trigger", "action"):
        model_path = scope_root / "checkpoints" / config["experiment_id"] / kind / "final"
        if not model_path.is_dir():
            raise RuntimeError(f"missing {kind} checkpoint: {model_path}")
        model = SentenceTransformer(
            str(model_path),
            device="cuda:0",
            model_kwargs={"torch_dtype": torch.float32},
            **saved_model_kwargs(model_path),
        )
        if config["level"] == "service":
            corpus = read_json(run_root / "derived_data" / "e0_service" / f"corpus_{kind}.json")
            identities = [row["service_id"] for row in corpus]
            documents = [row["text"] for row in corpus]
            gold_key = f"{kind}_service"
        else:
            corpus = read_json(data_root / "corpus" / CORPUS_FILE[kind])
            identities = [row["url"] for row in corpus]
            documents = [row[f"text_{config['view']}"] for row in corpus]
            gold_key = f"{kind}_url"
        started = time.monotonic()
        doc_embeddings = model.encode_document(
            documents,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embeddings = model.encode_query(
            [group["query"] for group in groups],
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if not np.isfinite(doc_embeddings).all() or not np.isfinite(query_embeddings).all():
            raise RuntimeError(f"non-finite {kind} dev embeddings")
        scores = query_embeddings @ doc_embeddings.T
        rank_maps = []
        top_ids = []
        rr_values = []
        ndcg_values = []
        hits = {1: [], 5: [], 10: []}
        id_recall = {1: [], 5: [], 10: []}
        for group_index, group in enumerate(groups):
            # Exact score order, then canonical identity for deterministic ties.
            order = sorted(
                range(len(identities)),
                key=lambda index: (-float(scores[group_index, index]), identities[index]),
            )
            rank_by_id = {identities[index]: rank for rank, index in enumerate(order)}
            valid = sorted({pair[gold_key] for pair in group["valid_pairs"]})
            missing = set(valid) - set(rank_by_id)
            if missing:
                raise AssertionError(f"gold identities absent from {kind} corpus: {sorted(missing)}")
            valid_ranks = [rank_by_id[value] for value in valid]
            rank_maps.append({value: rank_by_id[value] for value in valid})
            top_ids.append([identities[index] for index in order[:10]])
            rr_values.append(1.0 / (min(valid_ranks) + 1))
            ndcg_values.append(ndcg_at_k(valid_ranks, len(valid), 10))
            for cutoff in (1, 5, 10):
                hits[cutoff].append(int(any(rank < cutoff for rank in valid_ranks)))
                id_recall[cutoff].append(sum(rank < cutoff for rank in valid_ranks) / len(valid_ranks))
        all_ranks[kind] = rank_maps
        all_top_ids[kind] = top_ids
        per_side[kind] = {
            **{f"R@{cutoff}": sum(hits[cutoff]) / len(groups) for cutoff in (1, 5, 10)},
            "MRR": sum(rr_values) / len(groups),
            "nDCG@10": sum(ndcg_values) / len(groups),
            "ragas_compatible_IDBasedContextRecall": {
                f"@{cutoff}": sum(id_recall[cutoff]) / len(groups) for cutoff in (1, 5, 10)
            },
            "hit_vectors": {f"R@{cutoff}": hits[cutoff] for cutoff in (1, 5, 10)},
            "candidate_count": len(identities),
        }
        side_elapsed[kind] = time.monotonic() - started
        del model, doc_embeddings, query_embeddings, scores
        torch.cuda.empty_cache()

    joint_hits = {1: [], 5: [], 10: []}
    reciprocal_candidate_depth = []
    per_sample = []
    for index, group in enumerate(groups):
        if config["level"] == "service":
            pairs = [
                (pair["trigger_service"], pair["action_service"])
                for pair in group["valid_pairs"]
            ]
        else:
            pairs = [
                (pair["trigger_url"], pair["action_url"])
                for pair in group["valid_pairs"]
            ]
        pair_ranks = [
            max(all_ranks["trigger"][index][trigger], all_ranks["action"][index][action])
            for trigger, action in pairs
        ]
        best_pair_rank = min(pair_ranks)
        reciprocal_candidate_depth.append(1.0 / (best_pair_rank + 1))
        row_hits = {}
        for cutoff in (1, 5, 10):
            hit = int(any(rank < cutoff for rank in pair_ranks))
            joint_hits[cutoff].append(hit)
            row_hits[f"R@{cutoff}"] = hit
        top_pair = (all_top_ids["trigger"][index][0], all_top_ids["action"][index][0])
        if int(top_pair in set(pairs)) != row_hits["R@1"]:
            raise AssertionError("pair-preserving top-1 mismatch")
        per_sample.append({
            "group_id": group["group_id"],
            "valid_pairs": pairs,
            "top_trigger_ids": all_top_ids["trigger"][index],
            "top_action_ids": all_top_ids["action"][index],
            "gold_trigger_ranks": all_ranks["trigger"][index],
            "gold_action_ranks": all_ranks["action"][index],
            "best_pair_rank_zero_based": best_pair_rank,
            "hits": row_hits,
        })
    metrics = {
        "sides": per_side,
        "joint": {
            **{f"R@{cutoff}": sum(joint_hits[cutoff]) / len(groups) for cutoff in (1, 5, 10)},
            "MeanReciprocalCandidateDepth": sum(reciprocal_candidate_depth) / len(groups),
            "candidate_depth_definition": "1 / (1 + min_valid_pair max(trigger_marginal_rank, action_marginal_rank)); this is not joint-list MRR",
            "hit_vectors": {f"R@{cutoff}": joint_hits[cutoff] for cutoff in (1, 5, 10)},
        },
    }
    for section in (metrics["sides"]["trigger"], metrics["sides"]["action"], metrics["joint"]):
        if not (section["R@1"] <= section["R@5"] <= section["R@10"]):
            raise AssertionError(f"non-monotonic recall: {section}")
    scalar_values = [
        value
        for section in (metrics["sides"]["trigger"], metrics["sides"]["action"], metrics["joint"])
        for key, value in section.items()
        if isinstance(value, (int, float)) and key != "candidate_count"
    ]
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise RuntimeError("non-finite evaluation metric")

    result = {
        "status": "passed",
        "completed_at": utc_now(),
        "dataset_id": dataset_manifest["dataset_id"],
        "experiment_id": config["experiment_id"],
        "scope": args.scope,
        "task_level": config["level"],
        "view": config["view"],
        "identity": "channel_slug" if config["level"] == "service" else "exact_function_url",
        "split": "dev",
        "rows": len(groups),
        "physical_gpu": int(physical_gpu),
        "prompt_policy": "encode_query / encode_document",
        "metric_policy": "exact identity; valid pairs preserved; no Cartesian projection",
        "ragas_package_executed": False,
        "ragas_note": "ID-based recall formula is reported as a compatible supplement; RAGAS is not installed and no LLM judge was used",
        "elapsed_seconds": side_elapsed,
        "metrics": metrics,
        "per_sample": per_sample,
    }
    output_path = scope_root / "results" / f"{config['experiment_id']}_dev.json"
    write_json(output_path, result)
    done_path = scope_root / "manifests" / config["experiment_id"] / "evaluate.done.json"
    write_json(done_path, {key: value for key, value in result.items() if key != "per_sample"} | {
        "result_path": str(output_path), "result_sha256": sha256_file(output_path)
    })
    print(json.dumps({key: value for key, value in result.items() if key != "per_sample"}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

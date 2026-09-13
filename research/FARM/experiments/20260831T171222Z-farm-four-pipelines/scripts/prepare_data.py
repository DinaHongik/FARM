#!/usr/bin/env python3
"""Prepare immutable-run derivatives: E0 services and E3 hard negatives."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiment_lib import (  # noqa: E402
    DATASET_ID,
    CORPUS_FILE,
    assert_one_visible_gpu,
    build_service_view,
    function_pair_path,
    prompt_text,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    validate_dataset,
    validate_training_rows,
    write_json,
)


def build_hard_negatives(
    data_root: Path,
    run_root: Path,
    model_name: str,
    model_revision: str,
    count: int,
    batch_size: int,
) -> dict:
    """Mine by exact corpus identity while excluding every known valid URL/alias."""
    physical_gpu = assert_one_visible_gpu()
    manifest = validate_dataset(data_root)

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(42)
    model = SentenceTransformer(
        model_name, revision=model_revision, device="cuda:0", model_kwargs={"torch_dtype": torch.float32}
    )
    model.max_seq_length = 512
    if "query" not in model.prompts or "document" not in model.prompts:
        raise RuntimeError("Embedding model lacks required query/document prompts")

    destination = run_root / "derived_data" / "e3_hard_negatives"
    destination.mkdir(parents=True, exist_ok=True)
    reranker_groups = read_json(data_root / "splits" / "reranker_train.json")
    results = {}
    for kind in ("trigger", "action"):
        source_path = function_pair_path(data_root, kind, "schema")
        source_rows = read_json(source_path)
        corpus_path = data_root / "corpus" / CORPUS_FILE[kind]
        corpus = read_json(corpus_path)
        corpus_texts = [row["text_schema"] for row in corpus]
        corpus_urls = [row["url"] for row in corpus]

        alias_to_indices: dict[str, set[int]] = {}
        for index, row in enumerate(corpus):
            for url in {row["url"], *(row.get("equivalent_urls") or [])}:
                alias_to_indices.setdefault(url, set()).add(index)

        document_embeddings = model.encode_document(
            corpus_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        anchors = [row["anchor"] for row in source_rows]
        query_embeddings = model.encode_query(
            anchors,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        if not np.isfinite(document_embeddings).all() or not np.isfinite(query_embeddings).all():
            raise RuntimeError(f"non-finite {kind} base embeddings during mining")

        # `reranker_train` is a disjoint audit/calibration set for the mining
        # window, never an encoder-training anchor source. This preserves E2/E3's
        # exact anchor equality while exercising the split assigned to hard-negative
        # and reranker work by the experiment specification.
        audit_queries = model.encode_query(
            [group["query"] for group in reranker_groups],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        audit_scores = audit_queries @ document_embeddings.T
        audit_available = []
        gold_key = f"{kind}_url"
        for group_index, group in enumerate(reranker_groups):
            forbidden = set()
            for url in {pair[gold_key] for pair in group["valid_pairs"]}:
                forbidden.update(alias_to_indices.get(url, set()))
            available = sum(
                index not in forbidden and 0.35 <= float(audit_scores[group_index, index]) <= 0.98
                for index in range(len(corpus))
            )
            audit_available.append(available)

        output_rows = []
        block_size = 256
        for block_start in range(0, len(source_rows), block_size):
            block_stop = min(block_start + block_size, len(source_rows))
            scores = query_embeddings[block_start:block_stop] @ document_embeddings.T
            for local_index, row_index in enumerate(range(block_start, block_stop)):
                row = source_rows[row_index]
                forbidden = set()
                for url in row.get("valid_label_urls") or [row["label_url"]]:
                    forbidden.update(alias_to_indices.get(url, set()))
                # Also exclude byte-identical documents, even if an alias was absent.
                valid_texts = {
                    corpus_texts[index] for index in forbidden
                } | {row["positive"]}
                valid_indices = sorted(forbidden)
                if not valid_indices:
                    raise AssertionError(f"no valid corpus identity for {kind} row {row_index}")
                positive_score = max(float(scores[local_index, index]) for index in valid_indices)
                # Mirror the audited server miner's absolute false-negative guard.
                # A relative margin is unsafe when the base model assigns a low or
                # negative score to a gold; it can leave no candidates and silently
                # reweight the dataset. Identity/alias/text exclusions remain hard.
                ceiling = 0.98
                order = np.argsort(-scores[local_index], kind="stable")
                selected = []
                fallback_count = 0
                for rank, candidate_index in enumerate(order.tolist(), start=1):
                    if candidate_index in forbidden:
                        continue
                    if corpus_texts[candidate_index] in valid_texts:
                        continue
                    if float(scores[local_index, candidate_index]) > ceiling:
                        continue
                    score = float(scores[local_index, candidate_index])
                    selected.append((candidate_index, score, rank))
                    if score < 0.35:
                        fallback_count += 1
                    if len(selected) == count:
                        break
                if len(selected) != count:
                    # Deterministic easy-negative fallback retains every source row.
                    # It should almost never fire with a >1k document corpus, but is
                    # explicit rather than silently dropping head/ambiguous anchors.
                    already = {index for index, _, _ in selected}
                    for candidate_index in sorted(range(len(corpus)), key=lambda i: corpus_urls[i]):
                        if candidate_index in forbidden or candidate_index in already:
                            continue
                        if corpus_texts[candidate_index] in valid_texts:
                            continue
                        score = float(scores[local_index, candidate_index])
                        if score > ceiling:
                            continue
                        selected.append((candidate_index, score, len(corpus)))
                        fallback_count += 1
                        if len(selected) == count:
                            break
                if len(selected) != count:
                    raise RuntimeError(f"could not retain row {row_index} with {count} safe negatives")
                output_rows.append({
                    **row,
                    "negatives": [corpus_texts[index] for index, _, _ in selected],
                    "negative_urls": [corpus_urls[index] for index, _, _ in selected],
                    "negative_scores": [score for _, score, _ in selected],
                    "negative_ranks": [rank for _, _, rank in selected],
                    "best_valid_label_score": positive_score,
                    "negative_similarity_ceiling": ceiling,
                    "negative_fallback_count": fallback_count,
                    "hard_negative_provenance": {
                        "source_split": "encoder_train",
                        "source_pair_artifact": str(source_path.relative_to(data_root)),
                        "corpus_artifact": str(corpus_path.relative_to(data_root)),
                        "model": model_name,
                        "model_revision": model_revision,
                        "model_prompt_policy": "encode_query/encode_document built-in prompts",
                        "selection": "highest cosine candidates <= 0.98 after valid URL, alias, and text exclusion; values below 0.35 retained as deterministic easy fallback so no row is dropped",
                    },
                })
        output_path = destination / f"{kind}.json"
        write_json(output_path, output_rows)
        validation = validate_training_rows(
            {
                "level": "function", "view": "schema", "hard_negatives": count,
            },
            data_root,
            run_root,
            kind,
            output_rows,
        )
        results[kind] = {
            **validation,
            "source_sha256": sha256_file(source_path),
            "corpus_sha256": sha256_file(corpus_path),
            "output_sha256": sha256_file(output_path),
            "output_bytes": output_path.stat().st_size,
            "hard_window_negatives": sum(
                1 for row in output_rows for score in row["negative_scores"] if score >= 0.35
            ),
            "easy_fallback_negatives": sum(row["negative_fallback_count"] for row in output_rows),
            "easy_fallback_rows": sum(row["negative_fallback_count"] > 0 for row in output_rows),
            "reranker_train_window_audit": {
                "rows": len(reranker_groups),
                "min_available": min(audit_available),
                "max_available": max(audit_available),
                "rows_with_at_least_four": sum(value >= count for value in audit_available),
            },
        }
        del document_embeddings, query_embeddings, audit_queries, audit_scores
        torch.cuda.empty_cache()

    result = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": manifest["dataset_id"],
        "source_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "visible_physical_gpu": physical_gpu,
        "model": model_name,
        "model_revision": model_revision,
        "model_prompts": {"query": model.prompts["query"], "document": model.prompts["document"]},
        "negative_count_per_anchor": count,
        "hardness_weighting": None,
        "split_policy": "encoder_train anchors; reranker_train audit only; dev and locked test never loaded",
        "sides": results,
    }
    result["derived_view_hash"] = sha256_json(results)
    write_json(destination / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("service", "hard-negatives", "all"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--model-revision", default="57c266a740f537b4dc058e1b0cda161fd15afa75")
    parser.add_argument("--negative-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.run_root = args.run_root.resolve()
    args.data_root = args.data_root.resolve()
    if args.command in {"service", "all"}:
        result = build_service_view(args.data_root, args.run_root / "derived_data")
        print(json.dumps({"service": result}, indent=2, sort_keys=True), flush=True)
    if args.command in {"hard-negatives", "all"}:
        result = build_hard_negatives(
            args.data_root, args.run_root, args.model, args.model_revision,
            args.negative_count, args.batch_size
        )
        print(json.dumps({"hard_negatives": result}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

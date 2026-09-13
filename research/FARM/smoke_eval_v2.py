"""Small multi-gold retrieval/evaluator smoke for Dataset v2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from farm.dataset_v2 import CORPUS_FILENAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/v2"))
    parser.add_argument("--view", choices=("plain", "schema"), required=True)
    parser.add_argument("--split", choices=("reranker_train", "dev", "test"), default="dev")
    parser.add_argument("--trigger-model", required=True)
    parser.add_argument("--action-model", required=True)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    visible = [value for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, env={visible}, torch={torch.cuda.device_count()}")

    manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    groups = json.loads((args.data_root / f"splits/{args.split}.json").read_text(encoding="utf-8"))[: args.max_rows]
    if not groups:
        raise RuntimeError("evaluation split is empty")
    queries = [group["query"] for group in groups]

    ranks: dict[str, list[dict[str, int]]] = {}
    top_urls: dict[str, list[str]] = {}
    for kind, model_path in (("trigger", args.trigger_model), ("action", args.action_model)):
        corpus = json.loads(
            (args.data_root / "corpus" / CORPUS_FILENAMES[kind]).read_text(encoding="utf-8")
        )
        urls = [row["url"] for row in corpus]
        processor_kwargs = {}
        saved_config = Path(model_path) / "config.json"
        if saved_config.is_file():
            model_type = json.loads(saved_config.read_text(encoding="utf-8")).get("model_type")
            # transformers 4.57.x misclassifies a locally saved Gemma tokenizer as
            # Mistral solely because its vocabulary is large.  Explicit False keeps
            # the byte-identical Gemma pre-tokenizer and avoids applying that patch.
            if model_type == "gemma3_text":
                processor_kwargs["fix_mistral_regex"] = False
        model = SentenceTransformer(
            model_path,
            device="cuda:0",
            model_kwargs={"torch_dtype": torch.float32},
            processor_kwargs=processor_kwargs,
        )
        documents = model.encode(
            [row[f"text_{args.view}"] for row in corpus], batch_size=64,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
        )
        encoded_queries = model.encode(
            queries, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        del model
        torch.cuda.empty_cache()
        if not np.isfinite(documents).all() or not np.isfinite(encoded_queries).all():
            raise RuntimeError(f"non-finite {kind} embeddings")
        scores = encoded_queries @ documents.T
        side_ranks = []
        side_tops = []
        for index, group in enumerate(groups):
            order = sorted(range(len(urls)), key=lambda pos: (-float(scores[index, pos]), urls[pos]))
            rank_by_url = {urls[pos]: rank for rank, pos in enumerate(order)}
            valid = group[f"gold_{kind}_urls"]
            side_ranks.append({url: rank_by_url[url] for url in valid})
            side_tops.append(urls[order[0]])
        ranks[kind] = side_ranks
        top_urls[kind] = side_tops

    metrics = {}
    for cutoff in (1, 5, 10):
        hits = 0
        for index, group in enumerate(groups):
            valid_pairs = {
                (row["trigger_url"], row["action_url"])
                for row in group["valid_pairs"]
            }
            if any(
                ranks["trigger"][index][trigger] < cutoff
                and ranks["action"][index][action] < cutoff
                for trigger, action in valid_pairs
            ):
                hits += 1
        metrics[f"joint_R@{cutoff}"] = hits / len(groups)
    if not (metrics["joint_R@1"] <= metrics["joint_R@5"] <= metrics["joint_R@10"]):
        raise AssertionError(f"non-monotonic retrieval metrics: {metrics}")
    top1_hits = sum(
        (top_urls["trigger"][index], top_urls["action"][index])
        in {(row["trigger_url"], row["action_url"]) for row in group["valid_pairs"]}
        for index, group in enumerate(groups)
    )
    if top1_hits / len(groups) != metrics["joint_R@1"]:
        raise AssertionError("pair-preserving top-1 score disagrees with rank score")

    result = {
        "status": "passed",
        "dataset_id": manifest["dataset_id"],
        "view": args.view,
        "split": args.split,
        "rows": len(groups),
        "visible_gpu": visible[0],
        "trigger_model": args.trigger_model,
        "action_model": args.action_model,
        "metrics": metrics,
        "record_ids": [group["group_id"] for group in groups],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

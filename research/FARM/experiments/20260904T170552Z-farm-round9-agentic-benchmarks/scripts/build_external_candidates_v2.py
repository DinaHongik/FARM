#!/usr/bin/env python3
"""Build public RecipeGen/TARGE candidates with bi-encoder + cross-encoder."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.adapters.common import normalize_label, split_fields
from farm_r9.artifact_io import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from farm_r9.candidate_sets import opaque_candidate_view


def assert_gpu(expected: int) -> None:
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible != [str(expected)]:
        raise RuntimeError(f"expected CUDA_VISIBLE_DEVICES={expected}, got {visible}")
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("external candidate worker requires exactly one V100")


def make_catalog(metadata_path: Path) -> dict[str, list[dict]]:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # RecipeGen metadata contains nine legacy/current descriptions for the same
    # Automatic trigger identities.  They are evidence variants, not distinct
    # gold functions.  Consolidate by the benchmark's channel+function identity
    # and retain every unique description instead of arbitrarily dropping one
    # or making an unscorable duplicate candidate.
    grouped: dict[str, dict[str, list[dict]]] = {"trigger": {}, "action": {}}
    for row in rows:
        full = row["function"]
        event = full.split(".", 1)[1] if "." in full else full
        identifier = f"{normalize_label(row['channel'])}::{normalize_label(full)}"
        variants = grouped[row["split"]].setdefault(identifier, [])
        variants.append(row)
    output: dict[str, list[dict]] = {"trigger": [], "action": []}
    for side, identities in grouped.items():
        for identifier in sorted(identities):
            variants = identities[identifier]
            first = variants[0]
            full = first["function"]
            event = full.split(".", 1)[1] if "." in full else full
            field_variants = {tuple(split_fields(row["field"])) for row in variants}
            if len(field_variants) != 1:
                raise ValueError(f"conflicting fields for external endpoint {identifier}")
            endpoint_descriptions = list(dict.fromkeys(row["desc"] for row in variants if row["desc"]))
            service_descriptions = list(dict.fromkeys(row["channel_desc"] for row in variants if row["channel_desc"]))
            document_parts = [f"service: {first['channel']}", f"function: {event}"]
            document_parts.extend(endpoint_descriptions)
            if first["field"]:
                document_parts.append(f"configuration fields: {first['field']}")
            document_parts.extend(f"service description: {value}" for value in service_descriptions)
            output[side].append({
                "canonical_id": identifier, "side": side,
                "service": first["channel"], "service_id": first["channel"], "function": full,
                "function_event": event, "service_norm": normalize_label(first["channel"]),
                "function_full_norm": normalize_label(full), "function_event_norm": normalize_label(event),
                "description": " | ".join(endpoint_descriptions),
                "field_names": list(next(iter(field_variants))),
                "source_variant_count": len(variants),
                "document": "\n".join(document_parts),
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bi-encoder", required=True, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--cross-encoder", required=True, default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--retrieve-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alias-seed", type=int, default=9052026)
    args = parser.parse_args()
    assert_gpu(args.expected_gpu)
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    benchmarks = ["recipegen_gold", "recipegen_noisy", "targe_gold", "targe_noisy", "targe_one_shot"]
    cases_by_benchmark = {name: read_jsonl(RUN_ROOT / "prepared" / name / "cases.jsonl") for name in benchmarks}
    all_queries = [case["input"]["query"] for name in benchmarks for case in cases_by_benchmark[name]]
    offsets, cursor = {}, 0
    for name in benchmarks:
        offsets[name] = (cursor, cursor + len(cases_by_benchmark[name]))
        cursor += len(cases_by_benchmark[name])
    catalog = make_catalog(args.metadata)
    bi = SentenceTransformer(args.bi_encoder, device="cuda:0", local_files_only=True)
    query_vectors = bi.encode(all_queries, batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    shortlists = {side: [] for side in ("trigger", "action")}
    for side in ("trigger", "action"):
        docs = catalog[side]
        vectors = bi.encode([row["document"] for row in docs], batch_size=128, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        similarities = query_vectors @ vectors.T
        for row_index in range(len(all_queries)):
            order = np.argpartition(-similarities[row_index], min(args.retrieve_k, len(docs)) - 1)[: args.retrieve_k]
            ordered = sorted(order.tolist(), key=lambda index: (-float(similarities[row_index, index]), docs[index]["canonical_id"]))
            shortlists[side].append(ordered)
    del bi, query_vectors
    torch.cuda.empty_cache()
    cross = CrossEncoder(args.cross_encoder, device="cuda:0", local_files_only=True, max_length=512)
    output_by_benchmark = {name: [] for name in benchmarks}
    global_index = 0
    for benchmark in benchmarks:
        for case in cases_by_benchmark[benchmark]:
            public_sides, private_sides = {}, {}
            for side in ("trigger", "action"):
                docs = catalog[side]
                candidate_indices = shortlists[side][global_index]
                pairs = [(case["input"]["query"], docs[index]["document"]) for index in candidate_indices]
                scores = [float(value) for value in cross.predict(pairs, batch_size=64, show_progress_bar=False)]
                order = sorted(range(len(candidate_indices)), key=lambda index: (-scores[index], docs[candidate_indices[index]]["canonical_id"]))[: args.top_k]
                ranked = []
                for rank, local_index in enumerate(order, start=1):
                    source = docs[candidate_indices[local_index]]
                    ranked.append({**source, "rank": rank, "score": scores[local_index]})
                public, aliases = opaque_candidate_view(ranked, case_id=case["case_id"], side=side, seed=args.alias_seed)
                public_sides[side], private_sides[side] = public, {"ranking": [row["canonical_id"] for row in ranked], "alias_map": aliases}
            output_by_benchmark[benchmark].append({
                "schema_version": "round9-candidates-v1", "benchmark": benchmark,
                "data_classification": "public", "case_id": case["case_id"], "input": case["input"],
                "public_evidence": {"trigger_candidates": public_sides["trigger"], "action_candidates": public_sides["action"]},
                "private": {"trigger": private_sides["trigger"], "action": private_sides["action"], "gold": case["private_gold"]},
            })
            global_index += 1
    del cross
    for benchmark, rows in output_by_benchmark.items():
        output = RUN_ROOT / "derived" / benchmark / "candidates.jsonl"
        manifest = output.with_suffix(".manifest.json")
        if output.exists() or manifest.exists():
            raise RuntimeError(f"refusing to overwrite external candidates for {benchmark}")
        write_jsonl_atomic(output, rows)
        write_json_atomic(manifest, {
            "schema_version": "round9-candidates-manifest-v1", "benchmark": benchmark,
            "data_classification": "public", "rows": len(rows), "retrieve_k": args.retrieve_k,
            "top_k": args.top_k, "gold_injection": False, "rank_hidden_from_models": True,
            "bi_encoder": args.bi_encoder, "cross_encoder": args.cross_encoder,
            "metadata_sha256": sha256_file(args.metadata), "output_sha256": sha256_file(output),
        })
    print(json.dumps({name: len(rows) for name, rows in output_by_benchmark.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

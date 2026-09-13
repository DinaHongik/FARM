#!/usr/bin/env python3
"""Frozen FARM E3 seed42 on public RecipeGen, local inference only.

This is an exploratory transfer diagnostic, not a matched-training comparison.
No private corpus, queries, schemas, or labels are read. Learned checkpoint
weights stay on DGX. No LLM or network is used. Existing results are untouched.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.artifact_io import read_json, read_jsonl, sha256_file
from farm_r9.adapters.common import normalize_label
from farm_r9.yao_agent import wilson_interval, exact_mcnemar
from farm_r9.paired_statistics import paired_delta_bootstrap
from build_external_candidates import make_catalog, assert_gpu
from collect_astra_public_results import save_once

CANDIDATES = {
    "recipegen_gold": "1bdaf7f49a48099454790691a5a610ddbc2ee2f49f758db5d153189295455f37",
    "recipegen_noisy": "34ba86f638024c4c9e834fa6020386006ec91f916dc039c92dcd95a867ec3009",
}
METADATA_SHA = "8013c38bd8aa2b3839d2fba47910caa99c9d03e74502a69f1d4d7d7b3e97fb36"
CHECKPOINTS = {
    "trigger": "7b4989e419552d91336331f27c7d14453695ad5ffa12f4059ce9cdf3e409d22c",
    "action": "a6b93ad05fd7ea530adfd3f9f8937f864bb3800d82763355f515b89364322dac",
}


def ordered_ranking(similarities, ids, *, top_k=50):
    import numpy as np
    if len(ids) != len(set(ids)) or similarities.ndim != 2 or similarities.shape[1] != len(ids):
        raise ValueError("invalid catalog similarity shape or duplicate identities")
    if not np.isfinite(similarities).all():
        raise ValueError("non-finite retrieval scores")
    return [[ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(row[i]), ids[i]))[:top_k]]
            for row in similarities]


def score_ranking(gold, rankings, documents, cutoffs=(1, 5, 10, 50)):
    scores = {}
    for k in cutoffs:
        for side in ("trigger", "action"):
            scores[f"{side}_r{k}"] = any(
                normalize_label(documents[i]["service"]) == gold[side + "_channel_norm"]
                and normalize_label(documents[i]["function"]) == gold[side + "_function_norm"]
                for i in rankings[side][:k])
        scores[f"joint_r{k}"] = scores[f"trigger_r{k}"] and scores[f"action_r{k}"]
    return scores


def encode_rankings(model_path, queries, documents, *, require_prompts=True):
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    config = read_json(model_path / "config.json")
    kwargs = {"model_kwargs": {"torch_dtype": torch.float32}, "local_files_only": True}
    if config.get("model_type") == "gemma3_text":
        kwargs["processor_kwargs"] = {"fix_mistral_regex": False}
    model = SentenceTransformer(str(model_path), device="cuda:0", **kwargs)
    if require_prompts and not {"query", "document"} <= set(model.prompts):
        raise ValueError("frozen FARM checkpoint missing asymmetric prompts")
    telemetry = {"max_seq_length": model.max_seq_length, "truncation_count_method": "tokenizer length with saved prompt, before truncation"}
    for label, texts in (("query", queries), ("document", [d["document"] for d in documents])):
        lengths = model.tokenizer([model.prompts.get(label, "") + text for text in texts],
            truncation=False, padding=False, return_length=True)["length"]
        telemetry[label] = {"n": len(texts), "above_max_seq_length": sum(n > model.max_seq_length for n in lengths),
                            "maximum_tokens_before_truncation": max(lengths)}
    options = dict(batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    dv = model.encode_document([d["document"] for d in documents], **options)
    qv = model.encode_query(queries, **options)
    if not np.isfinite(dv).all() or not np.isfinite(qv).all():
        raise ValueError("non-finite embeddings")
    if (np.linalg.norm(dv, axis=1) < 1e-6).any() or (np.linalg.norm(qv, axis=1) < 1e-6).any():
        raise ValueError("zero-norm embeddings")
    ranked = ordered_ranking(qv @ dv.T, [d["canonical_id"] for d in documents])
    del model, dv, qv
    gc.collect()
    torch.cuda.empty_cache()
    return ranked, telemetry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", type=int, default=0)
    args = parser.parse_args()
    assert_gpu(args.expected_gpu)
    import torch
    torch.cuda.set_per_process_memory_fraction(0.35)
    root = args.round9_root.resolve()
    metadata = root / "sources/recipegen_metadata.csv"
    if sha256_file(metadata) != METADATA_SHA:
        raise ValueError("public metadata binding mismatch")
    catalog = make_catalog(metadata)
    docs = {r["canonical_id"]: r for side in catalog.values() for r in side}
    cases = {}
    for name, digest in CANDIDATES.items():
        path = root / "derived" / name / "candidates.jsonl"
        if sha256_file(path) != digest:
            raise ValueError("public sample binding mismatch")
        cases[name] = read_jsonl(path)
        if len(cases[name]) != 150 or len({r["case_id"] for r in cases[name]}) != 150:
            raise ValueError("requires frozen 150-case public sample")
        if any(r.get("data_classification") != "public" for r in cases[name]):
            raise ValueError("nonpublic sample refused")
    seed_manifest = read_json(root / "derived/farm_v2_test/seeds/seed42.manifest.json")
    models = {side: Path(seed_manifest["checkpoints"][side]["path"]) for side in CHECKPOINTS}
    for side, path in models.items():
        if sha256_file(path / "model.safetensors") != CHECKPOINTS[side]:
            raise ValueError("frozen model checkpoint binding mismatch")
    generic_manifest = read_json(root / "derived/recipegen_gold/candidates.manifest.json")
    generic_model = Path(generic_manifest["bi_encoder"])
    if generic_model.name != "1110a243fdf4706b3f48f1d95db1a4f5529b4d41":
        raise ValueError("generic MiniLM snapshot differs")
    inventory = {label: {str(file.relative_to(path)): sha256_file(file)
                        for file in sorted(path.rglob("*")) if file.is_file()}
                 for label, path in {**models, "generic_minilm": generic_model}.items()}
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {"protocol": "farm-e3-seed42-public-transfer-v1-exploratory", "n_per_benchmark": 150,
        "candidate_sha256": CANDIDATES, "metadata_sha256": METADATA_SHA,
        "checkpoint_sha256": CHECKPOINTS, "full_checkpoint_inventory": inventory,
        "source_sha256": sha256_file(Path(__file__)),
        "dtype": "float32", "batch_size": 16, "gpu": args.expected_gpu,
        "catalog_text": "identical public document strings to generic MiniLM candidate builder",
        "prompting": "checkpoint encode_query / encode_document prompts",
        "private_training_overlap": "known overlaps; use existing overlap audit; not clean unseen generalization",
        "claim_boundary": "retrieval only; no binding, execution, equal-training architecture or SOTA claim",
        "cloud_calls": 0, "private_input_records_read": 0, "cutoffs": [1, 5, 10, 50],
        "generic_references": ["fresh MiniLM-only dense retrieval", "original generic bi-encoder plus cross-encoder"],
        "training_and_capacity_matched": False,
        "joint_recall_definition": "both endpoints in independent top-k lists, not a top-k pair ranking"}
    save_once(args.output / "manifest.json", manifest)
    ordered_cases = [case for name in CANDIDATES for case in cases[name]]
    rankings, generic_rankings, encoding_telemetry = {}, {}, {}
    for side in ("trigger", "action"):
        rankings[side], encoding_telemetry["farm_" + side] = encode_rankings(models[side], [c["input"]["query"] for c in ordered_cases], catalog[side])
        print(json.dumps({"side": side, "queries_encoded": len(ordered_cases), "catalog_size": len(catalog[side])}), flush=True)
    for side in ("trigger", "action"):
        generic_rankings[side], encoding_telemetry["generic_" + side] = encode_rankings(generic_model,
            [c["input"]["query"] for c in ordered_cases], catalog[side], require_prompts=False)
    save_once(args.output / "encoding_telemetry.json", encoding_telemetry)
    offset = 0
    for name in CANDIDATES:
        out = args.output / name
        out.mkdir(mode=0o700)
        rows = []
        for case in cases[name]:
            ranked = {side: rankings[side][offset] for side in rankings}
            reference = {side: case["private"][side]["ranking"] for side in rankings}
            generic_dense = {side: generic_rankings[side][offset] for side in rankings}
            rows.append({"case_id": case["case_id"], "terminal": True, "rankings": ranked,
                "scores": score_ranking(case["private"]["gold"], ranked, docs),
                "generic_scores": score_ranking(case["private"]["gold"], reference, docs, (1, 5, 10)),
                "generic_dense_rankings": generic_dense,
                "generic_dense_scores": score_ranking(case["private"]["gold"], generic_dense, docs)})
            offset += 1
        ledger = out / "records.jsonl"
        with ledger.open("x") as handle:
            os.fchmod(handle.fileno(), 0o600)
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        counts = {k: sum(r["scores"][k] for r in rows) for k in rows[0]["scores"]}
        paired = {}
        for reference_name, reference_field in (("generic_dense", "generic_dense_scores"), ("generic_plus_ce", "generic_scores")):
            paired[reference_name] = {}
            for metric in ("joint_r1", "joint_r5", "joint_r10"):
                x, y = [r["scores"][metric] for r in rows], [r[reference_field][metric] for r in rows]
                rescues, regressions = sum(a and not b for a, b in zip(x, y)), sum(b and not a for a, b in zip(x, y))
                paired[reference_name][metric] = {"generic_correct": sum(y), "farm_correct": sum(x),
                    "rescues": rescues, "regressions": regressions,
                    "paired_delta": paired_delta_bootstrap(x, y, expected_n=150),
                    "mcnemar_exact_two_sided_p": exact_mcnemar(rescues, regressions)}
        save_once(out / "aggregate.json", {"n": 150, "raw_numerators": counts,
            "raw_denominators": {k: 150 for k in counts},
            "percentages": {k: 100 * v / 150 for k, v in counts.items()},
            "wilson_95_percent": {k: wilson_interval(v, 150) for k, v in counts.items()},
            "generic_dense_raw_numerators": {k: sum(r["generic_dense_scores"][k] for r in rows) for k in rows[0]["generic_dense_scores"]},
            "paired_against_generic_references": paired, "records_sha256": sha256_file(ledger),
            "manifest_sha256": sha256_file(args.output / "manifest.json"),
            "multiplicity": "unadjusted exploratory diagnostics, not confirmatory claims"})
        print(json.dumps({"benchmark": name, "n": 150, "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()

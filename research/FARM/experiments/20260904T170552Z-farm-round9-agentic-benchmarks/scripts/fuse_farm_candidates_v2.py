#!/usr/bin/env python3
"""Fuse three frozen FARM seed rankings and build rank-hidden top-10 views."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import read_json, read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from farm_r9.candidate_sets import independent_recall, opaque_candidate_view, reciprocal_rank_fusion


def alias_map(corpus: list[dict]) -> dict[str, dict]:
    output = {}
    for row in corpus:
        for identifier in [row["url"], *(row.get("equivalent_urls") or [])]:
            output[identifier] = row
    return output


def visible_endpoint(row: dict, *, side: str, canonical_id: str, rank: int, score: float) -> dict:
    fields = [
        {
            "slug": field["slug"], "label": field.get("label") or field["slug"],
            "required": bool(field.get("required")), "bindable": bool(field.get("bindable")),
            "value_type": field.get("type") or field.get("value_type") or "any",
            "resource_like": not bool(field.get("bindable")) and bool(field.get("required")),
            "auth_like": False,
        }
        for field in row.get("input_fields", [])
    ]
    ingredients = [
        {"slug": item["slug"], "label": item.get("label") or item["slug"], "value_type": item.get("type") or "any"}
        for item in row.get("ingredients", [])
    ]
    return {
        "canonical_id": canonical_id, "rank": rank, "score": score,
        "side": side, "service": row.get("channel_display") or row["channel"],
        # Keep the canonical channel identity for scoring.  The display label is
        # useful model evidence but is not guaranteed to normalize to the slug
        # used by Dataset-v2 gold records (for example IFTTT Notifications).
        "service_id": row["channel"],
        "function": row["function_name"], "description": row.get("description") or "",
        "fields": fields, "ingredients": ingredients,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", default=["seed42", "seed1337", "seed2025"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alias-seed", type=int, default=9052026)
    args = parser.parse_args()
    cases = read_jsonl(RUN_ROOT / "prepared" / "farm_v2_test" / "cases.jsonl")
    seed_rows = []
    seed_manifests = {}
    for seed in args.seeds:
        path = RUN_ROOT / "derived" / "farm_v2_test" / "seeds" / f"{seed}.jsonl"
        manifest_path = path.with_suffix(".manifest.json")
        rows, manifest = read_jsonl(path), read_json(manifest_path)
        if [row["case_id"] for row in rows] != [case["case_id"] for case in cases]:
            raise RuntimeError(f"case order mismatch for {seed}")
        if sha256_file(path) != manifest["output_sha256"]:
            raise RuntimeError(f"seed artifact hash mismatch for {seed}")
        seed_rows.append(rows)
        seed_manifests[seed] = {"manifest_sha256": sha256_file(manifest_path), "output_sha256": manifest["output_sha256"]}
    trigger_map = alias_map(read_json(args.data_root / "corpus" / "triggers.json"))
    action_map = alias_map(read_json(args.data_root / "corpus" / "actions.json"))
    output, score_rows = [], []
    for index, case in enumerate(cases):
        rankings, public_sets, private_maps = {}, {}, {}
        for side, corpus_map in (("trigger", trigger_map), ("action", action_map)):
            fused = reciprocal_rank_fusion([rows[index][f"{side}_ranking"] for rows in seed_rows])
            ranking = [identifier for identifier, _ in fused]
            rankings[side] = ranking
            hydrated = [visible_endpoint(corpus_map[identifier], side=side, canonical_id=identifier, rank=rank, score=score) for rank, (identifier, score) in enumerate(fused[:args.top_k], start=1)]
            public, aliases = opaque_candidate_view(hydrated, case_id=case["case_id"], side=side, seed=args.alias_seed)
            public_sets[side] = public
            private_maps[side] = aliases
        output.append({
            "schema_version": "round9-candidates-v1", "benchmark": "farm_v2_test",
            "data_classification": "confidential", "case_id": case["case_id"],
            "input": case["input"], "public_evidence": {
                "trigger_candidates": public_sets["trigger"], "action_candidates": public_sets["action"],
            },
            "private": {
                "trigger_ranking": rankings["trigger"], "action_ranking": rankings["action"],
                "trigger_alias_map": private_maps["trigger"], "action_alias_map": private_maps["action"],
                "gold": case["private_gold"],
            },
        })
        score_rows.append({
            "trigger_ranking": rankings["trigger"], "action_ranking": rankings["action"],
            "gold_trigger_ids": case["private_gold"]["trigger_ids"],
            "gold_action_ids": case["private_gold"]["action_ids"],
            "gold_pairs": [
                {"trigger_id": pair["trigger_url"], "action_id": pair["action_url"]}
                for pair in case["private_gold"]["valid_pairs"]
            ],
        })
    path = RUN_ROOT / "derived" / "farm_v2_test" / "candidates.jsonl"
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise RuntimeError("refusing to overwrite frozen fused candidates")
    write_jsonl_atomic(path, output)
    write_json_atomic(manifest_path, {
        "schema_version": "round9-candidates-manifest-v1", "benchmark": "farm_v2_test",
        "data_classification": "confidential", "rows": len(output), "top_k": args.top_k,
        "gold_injection": False, "rank_hidden_from_models": True, "seed_artifacts": seed_manifests,
        "metrics": independent_recall(score_rows), "output_sha256": sha256_file(path),
    })
    print(json.dumps(read_json(manifest_path)["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

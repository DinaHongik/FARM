#!/usr/bin/env python3
"""Build a direct service-name candidate lattice from the validated E0 encoders."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_rrf_lattice import assert_gpu, rank_with_model, validate_data
from experiment_core import DATASET_ID, read_json, sha256_file, write_json


def service_corpora(data_root: Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    function_by_url = {}
    output = {}
    for side in ("trigger", "action"):
        functions = read_json(data_root / "corpus" / f"{side}s.json")
        grouped: dict[str, list[dict]] = defaultdict(list)
        for function in functions:
            function_by_url[function["url"]] = function
            grouped[function["channel"]].append(function)
        rows = []
        for service_id, members in sorted(grouped.items()):
            displays = [str(item.get("channel_display") or "").strip() for item in members]
            counts = Counter(value for value in displays if value)
            display = (
                sorted(counts, key=lambda value: (-counts[value], value.casefold(), value))[0]
                if counts else service_id.replace("_", " ")
            )
            rows.append({
                "url": service_id, "service_id": service_id, "channel": service_id,
                "function_name": display, "service_name": display,
                "text_plain": display, "text_schema": display,
            })
        output[side] = rows
    return output, function_by_url


def project_groups(groups: list[dict], function_by_url: dict[str, dict]) -> list[dict]:
    rows = []
    for group in groups:
        valid = sorted({
            (
                function_by_url[pair["trigger_url"]]["channel"],
                function_by_url[pair["action_url"]]["channel"],
            )
            for pair in group["valid_pairs"]
        })
        rows.append({
            "group_id": group["group_id"], "query": group["query"],
            "valid_pairs": [
                {"trigger_url": trigger, "action_url": action}
                for trigger, action in valid
            ],
        })
    return rows


def score(rows: list[dict], depth: int) -> dict:
    result = {}
    for cutoff in (1, 5, depth):
        hit = 0
        for row in rows:
            trigger = {item["url"] for item in row["trigger_candidates"][:cutoff]}
            action = {item["url"] for item in row["action_candidates"][:cutoff]}
            hit += int(any(
                pair["trigger_url"] in trigger and pair["action_url"] in action
                for pair in row["valid_pairs"]
            ))
        result[f"joint_service_R@{cutoff}"] = hit / len(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--expected-gpu", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    assert_gpu(args.expected_gpu)
    run_root, data_root = args.run_root.resolve(), args.data_root.resolve()
    validate_data(data_root)
    spec = read_json(args.spec.resolve())
    checkpoints = {}
    for side in ("trigger", "action"):
        path = Path(spec[side]).resolve()
        actual = sha256_file(path / "model.safetensors")
        if actual != spec[f"{side}_model_sha256"]:
            raise RuntimeError(f"E0 {side} checkpoint hash mismatch")
        checkpoints[side] = {"path": str(path), "model_sha256": actual}

    corpora, function_by_url = service_corpora(data_root)
    output_root = run_root / "derived_data" / "service"
    output_root.mkdir(parents=True, exist_ok=True)
    for side in ("trigger", "action"):
        service_path = output_root / f"corpus_{side}.json"
        write_json(service_path, [{
            "service_id": row["service_id"], "text": row["service_name"],
            "source_function_urls": sorted(
                url for url, function in function_by_url.items()
                if function["channel"] == row["service_id"] and function["kind"] == side
            ),
        } for row in corpora[side]])

    for split in ("reranker_train", "dev"):
        groups = project_groups(read_json(data_root / "splits" / f"{split}.json"), function_by_url)
        rankings = {
            side: rank_with_model(Path(checkpoints[side]["path"]), groups, corpora[side], args.top_k)
            for side in ("trigger", "action")
        }
        maps = {side: {row["url"]: row for row in corpora[side]} for side in corpora}
        rows = []
        for index, group in enumerate(groups):
            side_candidates = {}
            for side in ("trigger", "action"):
                candidates = []
                for rank, identifier in enumerate(rankings[side][index], start=1):
                    source = maps[side][identifier]
                    candidates.append({
                        **source, "retrieval_rank": rank,
                        "retrieval_score": float(args.top_k - rank + 1),
                    })
                side_candidates[f"{side}_candidates"] = candidates
            rows.append({
                **group, **side_candidates,
                "routing_confidence": min(
                    side_candidates["trigger_candidates"][0]["retrieval_score"] - side_candidates["trigger_candidates"][1]["retrieval_score"],
                    side_candidates["action_candidates"][0]["retrieval_score"] - side_candidates["action_candidates"][1]["retrieval_score"],
                ),
            })
        output = output_root / f"{split}.json"
        if output.exists():
            raise RuntimeError(f"refusing to overwrite service lattice: {output}")
        write_json(output, rows)
        manifest = {
            "status": "passed", "dataset_id": DATASET_ID, "split": split,
            "rows": len(rows), "task_level": "service", "view": "service",
            "top_k": args.top_k, "identity": "exact service/channel slug",
            "pair_policy": "projection of observed valid function pairs",
            "gold_injection": False, "checkpoints": checkpoints,
            "metrics": score(rows, args.top_k), "output": str(output),
            "output_sha256": sha256_file(output),
        }
        write_json(output.with_suffix(".manifest.json"), manifest)
        print(json.dumps({"split": split, "metrics": manifest["metrics"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

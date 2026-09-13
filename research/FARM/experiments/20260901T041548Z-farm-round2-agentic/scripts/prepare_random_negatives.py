#!/usr/bin/env python3
"""Build deterministic identity-safe random-negative controls from Dataset v2."""
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
    DATASET_ID,
    function_pair_path,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    validate_dataset,
    write_json,
)


def stable_order(seed: int, side: str, row: dict, url: str) -> str:
    key = "|".join((str(seed), side, row["group_id"], row["label_url"], url))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def build_side(data_root: Path, output_root: Path, side: str, count: int, seed: int) -> dict:
    source_path = function_pair_path(data_root, side, "schema")
    corpus_path = data_root / "corpus" / CORPUS_FILE[side]
    source = read_json(source_path)
    corpus = read_json(corpus_path)
    corpus_by_url = {row["url"]: row for row in corpus}

    alias_to_urls: dict[str, set[str]] = {}
    for item in corpus:
        identities = {item["url"], *(item.get("equivalent_urls") or [])}
        for identity in identities:
            alias_to_urls.setdefault(identity, set()).add(item["url"])

    output = []
    all_urls = sorted(corpus_by_url)
    for row in source:
        valid = set(row.get("valid_label_urls") or [row["label_url"]])
        forbidden: set[str] = set()
        for identity in valid:
            forbidden.update(alias_to_urls.get(identity, set()))
        valid_texts = {row["positive"]} | {
            corpus_by_url[url]["text_schema"] for url in forbidden if url in corpus_by_url
        }
        candidates = [
            url for url in all_urls
            if url not in forbidden and corpus_by_url[url]["text_schema"] not in valid_texts
        ]
        candidates.sort(key=lambda url: (stable_order(seed, side, row, url), url))
        selected_urls: list[str] = []
        selected_texts: set[str] = set()
        for url in candidates:
            text = corpus_by_url[url]["text_schema"]
            if text in selected_texts:
                continue
            selected_urls.append(url)
            selected_texts.add(text)
            if len(selected_urls) == count:
                break
        if len(selected_urls) != count:
            raise RuntimeError(f"could not select {count} safe random negatives for {side}")
        output.append({
            **row,
            "negatives": [corpus_by_url[url]["text_schema"] for url in selected_urls],
            "negative_urls": selected_urls,
            "negative_strategy": "sha256_uniform_without_replacement",
            "negative_seed": seed,
        })

    output_path = output_root / f"{side}.json"
    write_json(output_path, output)
    return {
        "rows": len(output),
        "unique_labels": len({row["label_url"] for row in output}),
        "negative_count": len(output) * count,
        "source_sha256": sha256_file(source_path),
        "corpus_sha256": sha256_file(corpus_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    dataset = validate_dataset(data_root)
    output_root = run_root / "derived_data" / "random4"
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite prepared random negatives: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    sides = {
        side: build_side(data_root, output_root, side, args.count, args.seed)
        for side in ("trigger", "action")
    }
    manifest = {
        "status": "passed",
        "created_at": utc_now(),
        "dataset_id": DATASET_ID,
        "source_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "negative_source": "random4",
        "negative_count_per_anchor": args.count,
        "seed": args.seed,
        "selection": "per-row SHA-256 order; identity/alias/byte-equal positives excluded",
        "split_policy": "encoder_train only; reranker_train/dev/locked test never loaded",
        "sides": sides,
    }
    manifest["derived_view_hash"] = sha256_json({"sides": sides, "selection": manifest["selection"]})
    write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

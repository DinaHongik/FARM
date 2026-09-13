#!/usr/bin/env python3
"""Record complete file-level hashes for every reused model tree."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiment_core import read_json, sha256_file, write_json


def tree_record(path: Path) -> dict:
    path = path.resolve()
    files = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        files.append({
            "relative_path": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    if not files:
        raise RuntimeError(f"source model tree is empty: {path}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(path),
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0-spec", type=Path, required=True)
    parser.add_argument("--e3-spec", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite source-tree binding")
    e0 = read_json(args.e0_spec)
    e3 = read_json(args.e3_spec)
    sources = {"base_cross_encoder": args.base_model}
    for side in ("trigger", "action"):
        sources[f"e0_{side}"] = Path(e0[side])
    for retriever in e3["retrievers"]:
        for side in ("trigger", "action"):
            sources[f"e3_{retriever['id']}_{side}"] = Path(retriever[side])
    write_json(args.output, {
        "status": "passed",
        "identity": "sha256 over every regular file, then canonical file-list SHA-256",
        "sources": {label: tree_record(path) for label, path in sorted(sources.items())},
    })


if __name__ == "__main__":
    main()

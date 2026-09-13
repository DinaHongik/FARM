from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from farm_r9.adapters.common import normalize_label, split_fields
from farm_r9.artifact_io import sha256_file, sha256_text
from farm_r9.sampling import stratified_sample


def parse_target(target: str) -> dict[str, Any]:
    parts = target.split(" <sep> ")
    if len(parts) != 6:
        raise ValueError(f"RecipeGen field target must have six components, got {len(parts)}")
    trigger_channel, trigger_function, trigger_fields, action_channel, action_function, action_fields = parts
    return {
        "trigger_channel": trigger_channel.strip(),
        "trigger_function": trigger_function.strip(),
        "trigger_fields": split_fields(trigger_fields),
        "action_channel": action_channel.strip(),
        "action_function": action_function.strip(),
        "action_fields": split_fields(action_fields),
    }


def prepare_recipegen(
    processed_path: Path,
    *,
    split: str,
    size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in {"gold", "noisy"}:
        raise ValueError("RecipeGen evaluation split must be gold or noisy")
    with processed_path.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    field_rows = [row for row in all_rows if row["split"] == split and row["granularity"] == "field"]
    prepared = []
    for source_index, row in enumerate(field_rows):
        gold = parse_target(row["target"])
        identifier = sha256_text(f"{split}\x1f{source_index}\x1f{row['source']}\x1f{row['target']}")[:20]
        trigger_has_fields = bool(gold["trigger_fields"])
        action_has_fields = bool(gold["action_fields"])
        stratum = f"trigger_fields={int(trigger_has_fields)}|action_fields={int(action_has_fields)}"
        prepared.append({
            "case_id": f"recipegen:{split}:{identifier}",
            "source_index": source_index,
            "source": row["source"],
            "gold": gold,
            "stratum": stratum,
        })
    benchmark = f"recipegen_{split}"
    selected, sample_manifest = stratified_sample(
        prepared,
        size=size,
        seed=seed,
        benchmark=benchmark,
        id_of=lambda row: row["case_id"],
        stratum_of=lambda row: row["stratum"],
    )
    cases = []
    for row in selected:
        gold = row["gold"]
        cases.append({
            "schema_version": "round9-case-v1",
            "benchmark": benchmark,
            "case_id": row["case_id"],
            "stratum": row["stratum"],
            "input": {"query": row["source"]},
            "private_gold": {
                **gold,
                "trigger_channel_norm": normalize_label(gold["trigger_channel"]),
                "trigger_function_norm": normalize_label(gold["trigger_function"]),
                "action_channel_norm": normalize_label(gold["action_channel"]),
                "action_function_norm": normalize_label(gold["action_function"]),
                "trigger_fields_norm": [normalize_label(value) for value in gold["trigger_fields"]],
                "action_fields_norm": [normalize_label(value) for value in gold["action_fields"]],
            },
            "audit": {"source_index_within_split_field_track": row["source_index"]},
        })
    sample_manifest.update({
        "source_files": {str(processed_path.name): sha256_file(processed_path)},
        "tracks": ["channel", "function", "field-name"],
        "field_semantics": "names only; no field values or ingredient bindings",
    })
    return cases, sample_manifest

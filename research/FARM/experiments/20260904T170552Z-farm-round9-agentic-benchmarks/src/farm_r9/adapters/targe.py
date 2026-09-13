from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from farm_r9.adapters.common import normalize_label
from farm_r9.artifact_io import read_json, sha256_file, sha256_text
from farm_r9.sampling import stratified_sample


_TRIGGER = re.compile(r"^TRIGGER SERVICE:\s*(.*?),\s*TRIGGER EVENT:\s*(.*?)\s*$", re.IGNORECASE)
_ACTION = re.compile(r"^ACTION SERVICE:\s*(.*?),\s*ACTION EVENT:\s*(.*?)\s*$", re.IGNORECASE)


def parse_side(value: str, side: str) -> tuple[str, str]:
    match = (_TRIGGER if side == "trigger" else _ACTION).match(value)
    if not match:
        raise ValueError(f"cannot parse TARGE {side} output")
    return match.group(1).strip(), match.group(2).strip()


def _paths(data_root: Path, split: str) -> tuple[Path, Path, Path]:
    if split == "one_shot":
        base, suffix = data_root / "gold", "_one_shot"
    elif split in {"gold", "noisy"}:
        base, suffix = data_root / split, ""
    else:
        raise ValueError("TARGE split must be gold, noisy, or one_shot")
    return (
        base / f"test_recipe{suffix}.json",
        base / f"test_trigger{suffix}.json",
        base / f"test_action{suffix}.json",
    )


def _train_pairs(data_root: Path) -> set[tuple[str, str]]:
    trigger_rows = read_json(data_root / "train_trigger.json")
    action_rows = read_json(data_root / "train_action.json")
    if len(trigger_rows) != len(action_rows):
        raise ValueError("TARGE train trigger/action lengths differ")
    result = set()
    for trigger, action in zip(trigger_rows, action_rows):
        tc, tf = parse_side(trigger["output"], "trigger")
        ac, af = parse_side(action["output"], "action")
        result.add((f"{normalize_label(tc)}::{normalize_label(tf)}", f"{normalize_label(ac)}::{normalize_label(af)}"))
    return result


def prepare_targe(data_root: Path, *, split: str, size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    recipe_path, trigger_path, action_path = _paths(data_root, split)
    recipe_rows, trigger_rows, action_rows = map(read_json, (recipe_path, trigger_path, action_path))
    if not (len(recipe_rows) == len(trigger_rows) == len(action_rows)):
        raise ValueError("TARGE companion split lengths differ")
    train_pairs = _train_pairs(data_root)
    population = []
    for index, (recipe, trigger, action) in enumerate(zip(recipe_rows, trigger_rows, action_rows)):
        if not (recipe["input"] == trigger["input"] == action["input"]):
            raise ValueError(f"TARGE companion inputs differ at row {index}")
        tc, tf = parse_side(trigger["output"], "trigger")
        ac, af = parse_side(action["output"], "action")
        pair_key = (f"{normalize_label(tc)}::{normalize_label(tf)}", f"{normalize_label(ac)}::{normalize_label(af)}")
        overlap = pair_key in train_pairs
        identifier = sha256_text(f"{split}\x1f{index}\x1f{recipe['input']}\x1f{trigger['output']}\x1f{action['output']}")[:20]
        population.append({
            "case_id": f"targe:{split}:{identifier}", "index": index, "query": recipe["input"],
            "gold": (tc, tf, ac, af), "train_pair_overlap": overlap,
            "stratum": f"train_pair_overlap={int(overlap)}",
        })
    benchmark = f"targe_{split}"
    selected, sample_manifest = stratified_sample(
        population, size=size, seed=seed, benchmark=benchmark,
        id_of=lambda row: row["case_id"], stratum_of=lambda row: row["stratum"],
    )
    cases = []
    for row in selected:
        tc, tf, ac, af = row["gold"]
        cases.append({
            "schema_version": "round9-case-v1", "benchmark": benchmark,
            "case_id": row["case_id"], "stratum": row["stratum"],
            "input": {"query": row["query"]},
            "private_gold": {
                "trigger_channel": tc, "trigger_function": tf,
                "action_channel": ac, "action_function": af,
                "trigger_channel_norm": normalize_label(tc), "trigger_function_norm": normalize_label(tf),
                "action_channel_norm": normalize_label(ac), "action_function_norm": normalize_label(af),
            },
            "audit": {"source_index": row["index"], "train_pair_overlap": row["train_pair_overlap"]},
        })
    sample_manifest.update({
        "source_files": {path.name: sha256_file(path) for path in (recipe_path, trigger_path, action_path)},
        "train_pair_overlap_population": sum(row["train_pair_overlap"] for row in population),
        "train_pair_overlap_sample": sum(row["train_pair_overlap"] for row in selected),
        "validity_warning": "secondary reproduction only; released test pairs substantially overlap train pairs",
    })
    return cases, sample_manifest

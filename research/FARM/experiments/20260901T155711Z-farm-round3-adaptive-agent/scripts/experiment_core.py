#!/usr/bin/env python3
"""Stable data and metric interface for FARM round-three experiments."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
ALLOWED_SPLITS = frozenset({"reranker_train", "dev"})


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_artifact(
    payload_path: Path,
    manifest_path: Path,
    *,
    expected_dataset_id: str = DATASET_ID,
) -> tuple[list[dict], dict]:
    """Load a frozen candidate artifact after validating identity and split.

    The manifest is deliberately checked before the payload is opened. This is
    the safety seam that keeps repeated development experiments away from the
    locked test bytes.
    """
    manifest = read_json(manifest_path)
    split = manifest.get("split")
    if split == "test":
        raise ValueError("locked test candidate artifacts are prohibited")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"candidate split must be one of {sorted(ALLOWED_SPLITS)}")
    if manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError("candidate dataset identity mismatch")
    expected_hash = manifest.get("output_sha256")
    if not isinstance(expected_hash, str) or sha256_file(payload_path) != expected_hash:
        raise ValueError("candidate payload hash mismatch")
    rows = read_json(payload_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate payload must be a non-empty list")
    return rows, manifest


def pair_id(trigger_id: str, action_id: str) -> str:
    return f"{trigger_id} || {action_id}"


def serialize_pair_document(trigger_text: str, action_text: str, *, per_side_chars: int) -> str:
    """Give trigger and action equal bounded space before tokenizer truncation."""
    if per_side_chars < 32:
        raise ValueError("per-side serialization budget is too small")
    return (
        "TRIGGER:\n" + trigger_text.strip()[:per_side_chars]
        + "\nACTION:\n" + action_text.strip()[:per_side_chars]
    )


def _candidate_text(candidate: dict, view: str, side: str) -> str:
    if view not in {"plain", "schema", "service"}:
        raise ValueError(f"unsupported text view: {view}")
    if view == "service":
        value = candidate.get("service_name") or candidate.get("channel_display") or candidate.get("channel")
    else:
        value = candidate.get(f"text_{view}")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate lacks {view} text")
    return f"{side.upper()}: {value.strip()}"


def make_pair_training_group(row: dict, *, view: str, depth: int) -> dict:
    """Create one exact, multi-gold listwise pair-ranking group."""
    if depth < 2:
        raise ValueError("pair depth must be at least two")
    triggers = row["trigger_candidates"][:depth]
    actions = row["action_candidates"][:depth]
    valid = {
        (item["trigger_url"], item["action_url"])
        for item in row.get("valid_pairs", [])
    }
    documents = []
    for trigger in triggers:
        for action in actions:
            trigger_url = trigger["url"]
            action_url = action["url"]
            documents.append({
                "pair_id": pair_id(trigger_url, action_url),
                "text": (
                    _candidate_text(trigger, view, "trigger")
                    + "\n"
                    + _candidate_text(action, view, "action")
                ),
                "trigger_url": trigger_url,
                "action_url": action_url,
                "label": float((trigger_url, action_url) in valid),
            })
    return {"group_id": row["group_id"], "query": row["query"], "documents": documents}


def _hit(rank: int | None, cutoff: int) -> bool:
    return isinstance(rank, int) and 1 <= rank <= cutoff


def independent_rank_metrics(rows: list[dict], *, cutoffs: Iterable[int] = (1, 5, 10)) -> dict:
    """Score two separately ranked sides without calling it pair-list recall."""
    if not rows:
        raise ValueError("cannot score an empty row set")
    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] < 1:
        raise ValueError("cutoffs must be positive")
    result: dict[str, float] = {}
    for cutoff in ks:
        trigger_hits = [_hit(row.get("trigger_rank"), cutoff) for row in rows]
        action_hits = [_hit(row.get("action_rank"), cutoff) for row in rows]
        result[f"trigger_R@{cutoff}"] = sum(trigger_hits) / len(rows)
        result[f"action_R@{cutoff}"] = sum(action_hits) / len(rows)
        result[f"joint_independent_R@{cutoff}"] = sum(
            _hit(row.get("joint_rank"), cutoff)
            if "joint_rank" in row else trigger and action
            for row, trigger, action in zip(rows, trigger_hits, action_hits)
        ) / len(rows)
    maximum = ks[-1]
    for side in ("trigger", "action"):
        result[f"{side}_MRR@{maximum}"] = sum(
            1.0 / rank if _hit(rank, maximum) else 0.0
            for rank in (row.get(f"{side}_rank") for row in rows)
        ) / len(rows)
    return result


def pair_rank_metrics(rows: list[dict], *, cutoffs: Iterable[int] = (1, 5, 10)) -> dict:
    """Score a genuinely ordered list of composite trigger-action pair IDs."""
    if not rows:
        raise ValueError("cannot score an empty row set")
    ks = tuple(sorted(set(int(k) for k in cutoffs)))
    if not ks or ks[0] < 1:
        raise ValueError("cutoffs must be positive")
    result = {
        f"pair_R@{cutoff}": sum(_hit(row.get("pair_rank"), cutoff) for row in rows) / len(rows)
        for cutoff in ks
    }
    maximum = ks[-1]
    result[f"pair_MRR@{maximum}"] = sum(
        1.0 / rank if _hit(rank, maximum) else 0.0
        for rank in (row.get("pair_rank") for row in rows)
    ) / len(rows)
    return result

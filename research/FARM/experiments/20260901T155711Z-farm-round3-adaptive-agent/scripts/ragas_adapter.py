#!/usr/bin/env python3
"""Thin Ragas 0.4.3 compatibility adapter with native parity checks."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Iterable

from experiment_core import DATASET_ID, load_candidate_artifact, write_json


PINNED_RAGAS_VERSION = "0.4.3"


def validated_ids(values: Iterable[str], *, label: str) -> list[str]:
    result = list(values)
    if not result:
        raise ValueError(f"{label} IDs must be non-empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise TypeError(f"{label} IDs must be non-empty strings")
    return result


def native_id_recall(retrieved_ids: Iterable[str], reference_ids: Iterable[str]) -> float:
    retrieved = set(validated_ids(retrieved_ids, label="retrieved"))
    reference = set(validated_ids(reference_ids, label="reference"))
    return len(retrieved & reference) / len(reference)


def native_id_precision(retrieved_ids: Iterable[str], reference_ids: Iterable[str]) -> float:
    retrieved = set(validated_ids(retrieved_ids, label="retrieved"))
    reference = set(validated_ids(reference_ids, label="reference"))
    return len(retrieved & reference) / len(retrieved)


async def official_id_scores(retrieved: list[str], reference: list[str]) -> tuple[float, float]:
    """Execute the tagged package's legacy ID metrics directly."""
    from ragas import SingleTurnSample
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    sample = SingleTurnSample(
        retrieved_context_ids=validated_ids(retrieved, label="retrieved"),
        reference_context_ids=validated_ids(reference, label="reference"),
    )
    recall = await IDBasedContextRecall().single_turn_ascore(sample)
    precision = await IDBasedContextPrecision().single_turn_ascore(sample)
    values = float(recall), float(precision)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Ragas returned a non-finite ID score")
    return values


async def evaluate_candidates(rows: list[dict], cutoffs: tuple[int, ...]) -> dict:
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("Ragas cutoffs must be positive")
    version = importlib.metadata.version("ragas")
    if version != PINNED_RAGAS_VERSION:
        raise RuntimeError(f"ragas version drift: {version} != {PINNED_RAGAS_VERSION}")
    results = []
    for row in rows:
        gold_trigger = sorted({pair["trigger_url"] for pair in row["valid_pairs"]})
        gold_action = sorted({pair["action_url"] for pair in row["valid_pairs"]})
        record = {"group_id": row["group_id"], "cutoffs": {}}
        for cutoff in cutoffs:
            trigger = [item["url"] for item in row["trigger_candidates"][:cutoff]]
            action = [item["url"] for item in row["action_candidates"][:cutoff]]
            tr, tp = await official_id_scores(trigger, gold_trigger)
            ar, ap = await official_id_scores(action, gold_action)
            native = {
                "trigger_recall": native_id_recall(trigger, gold_trigger),
                "trigger_precision": native_id_precision(trigger, gold_trigger),
                "action_recall": native_id_recall(action, gold_action),
                "action_precision": native_id_precision(action, gold_action),
            }
            official = {
                "trigger_recall": tr, "trigger_precision": tp,
                "action_recall": ar, "action_precision": ap,
            }
            if any(abs(native[key] - official[key]) > 1e-12 for key in native):
                raise AssertionError(f"Ragas/native parity failure for {row['group_id']} at k={cutoff}")
            record["cutoffs"][str(cutoff)] = official
        results.append(record)
    return {
        "dataset_id": DATASET_ID,
        "ragas_package_executed": True,
        "ragas_version": version,
        "metric_import_path": "ragas.metrics.IDBasedContextRecall/Precision",
        "rows": len(results),
        "row_scores": results,
        "headline_policy": "exact URL R@k/MRR remain primary; ID metrics are audited parity checks",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoffs", default="1,5,10")
    args = parser.parse_args()
    cutoffs = tuple(int(item) for item in args.cutoffs.split(","))
    rows, _ = load_candidate_artifact(args.candidates, args.manifest)
    write_json(args.output, asyncio.run(evaluate_candidates(rows, cutoffs)))
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()

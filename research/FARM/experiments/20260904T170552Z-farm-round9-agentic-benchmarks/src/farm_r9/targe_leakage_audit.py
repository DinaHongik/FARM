"""Exact input/program leakage audit for frozen Round 9 TARGE samples.

This audit is intentionally independent of endpoint-pair overlap.  A test row is
classified against the released TARGE training recipes by the joint identity of
its normalized natural-language input and normalized output program.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from farm_r9.artifact_io import (
    ordered_ids_sha256,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)


SCHEMA_VERSION = "round9-targe-program-leakage-audit-v1"
NORMALIZATION_VERSION = "unicode-nfc-collapse-whitespace-preserve-case-punctuation-v1"
CLASSES = (
    "exact_input_program_seen",
    "input_seen_program_unseen",
    "input_unseen",
)


def normalize_exact_text(value: str) -> str:
    """Normalize representation noise without relaxing lexical identity.

    Unicode is normalized to NFC and every whitespace run becomes one ASCII
    space.  Case, punctuation, accents, underscores, and all other characters
    are preserved.  This is therefore stricter than semantic or fuzzy matching.
    """
    if not isinstance(value, str):
        raise TypeError("TARGE input and output values must be strings")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _recipe_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} is not an object")
        if not isinstance(row.get("input"), str) or not isinstance(row.get("output"), str):
            raise ValueError(f"{path}: row {index} lacks string input/output")
    return rows


def _gold_program(case: Mapping[str, Any]) -> str:
    gold = case.get("private_gold")
    if not isinstance(gold, Mapping):
        raise ValueError("frozen TARGE case lacks private_gold")
    keys = (
        "trigger_channel",
        "trigger_function",
        "action_channel",
        "action_function",
    )
    if any(not isinstance(gold.get(key), str) or not gold[key] for key in keys):
        raise ValueError("frozen TARGE case has malformed endpoint gold")
    return (
        f"IF {gold['trigger_channel']} {gold['trigger_function']} "
        f"THEN {gold['action_channel']} {gold['action_function']}"
    )


def classify_frozen_sample(
    *,
    selected_cases_path: Path,
    train_recipe_path: Path,
    test_recipe_path: Path,
    benchmark: str,
    expected_n: int = 150,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify every selected case into one mutually exclusive leakage class."""
    if expected_n <= 0:
        raise ValueError("expected_n must be positive")
    selected = read_jsonl(selected_cases_path)
    if len(selected) != expected_n:
        raise ValueError(f"expected {expected_n} frozen cases, found {len(selected)}")
    if len({row.get("case_id") for row in selected}) != len(selected):
        raise ValueError("frozen TARGE case IDs must be unique")

    train_rows = _recipe_rows(train_recipe_path)
    test_rows = _recipe_rows(test_recipe_path)
    train_inputs: set[str] = set()
    train_pairs: set[tuple[str, str]] = set()
    for row in train_rows:
        input_norm = normalize_exact_text(row["input"])
        output_norm = normalize_exact_text(row["output"])
        train_inputs.add(input_norm)
        train_pairs.add((input_norm, output_norm))

    records: list[dict[str, Any]] = []
    for case in selected:
        if case.get("benchmark") != benchmark:
            raise ValueError("frozen TARGE case benchmark mismatch")
        audit = case.get("audit")
        request = case.get("input")
        if not isinstance(audit, Mapping) or not isinstance(audit.get("source_index"), int):
            raise ValueError("frozen TARGE case lacks an integer source index")
        if not isinstance(request, Mapping) or not isinstance(request.get("query"), str):
            raise ValueError("frozen TARGE case lacks a string query")
        source_index = audit["source_index"]
        if source_index < 0 or source_index >= len(test_rows):
            raise ValueError("frozen TARGE source index is outside the official test split")
        source = test_rows[source_index]
        input_norm = normalize_exact_text(source["input"])
        output_norm = normalize_exact_text(source["output"])
        if normalize_exact_text(request["query"]) != input_norm:
            raise ValueError("frozen query differs from its official TARGE source row")
        if normalize_exact_text(_gold_program(case)) != output_norm:
            raise ValueError("frozen endpoint gold differs from its official output program")

        pair = (input_norm, output_norm)
        if pair in train_pairs:
            leakage_class = CLASSES[0]
        elif input_norm in train_inputs:
            leakage_class = CLASSES[1]
        else:
            leakage_class = CLASSES[2]
        prior_endpoint_overlap = audit.get("train_pair_overlap")
        if not isinstance(prior_endpoint_overlap, bool):
            raise ValueError("frozen case lacks the prior endpoint-pair overlap flag")
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": benchmark,
                "case_id": case["case_id"],
                "source_index": source_index,
                "leakage_class": leakage_class,
                "exact_input_program_seen": leakage_class == CLASSES[0],
                "input_seen": leakage_class != CLASSES[2],
                "prior_endpoint_pair_overlap": prior_endpoint_overlap,
                "input_sha256": sha256_text(input_norm),
                "output_program_sha256": sha256_text(output_norm),
                "input_program_sha256": sha256_text(input_norm + "\x1f" + output_norm),
            }
        )

    counts = Counter(row["leakage_class"] for row in records)
    cross = Counter(
        f"endpoint_overlap={int(row['prior_endpoint_pair_overlap'])}|{row['leakage_class']}"
        for row in records
    )
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark,
        "n": len(records),
        "normalization": {
            "version": NORMALIZATION_VERSION,
            "unicode": "NFC",
            "whitespace": "strip and collapse every Unicode whitespace run to one ASCII space",
            "case": "preserved",
            "punctuation": "preserved",
            "comparison": "exact normalized string identity",
        },
        "class_counts": {name: counts[name] for name in CLASSES},
        "class_percentages": {
            name: 100.0 * counts[name] / len(records) for name in CLASSES
        },
        "prior_endpoint_pair_overlap_count": sum(
            row["prior_endpoint_pair_overlap"] for row in records
        ),
        "prior_endpoint_pair_overlap_percentage": 100.0
        * sum(row["prior_endpoint_pair_overlap"] for row in records)
        / len(records),
        "endpoint_overlap_by_exact_leakage_class": dict(sorted(cross.items())),
        "ordered_case_ids_sha256": ordered_ids_sha256(records),
        "source_hashes": {
            "selected_cases_sha256": sha256_file(selected_cases_path),
            "train_recipe_sha256": sha256_file(train_recipe_path),
            "test_recipe_sha256": sha256_file(test_recipe_path),
        },
        "limitations": [
            "This is exact normalized text/program membership, not semantic near-duplicate detection.",
            "Endpoint-pair overlap is reported only to expose why it is not a leakage substitute.",
            "The frozen 150-case sample was originally stratified by endpoint-pair overlap, not by this exact leakage class.",
            "TARGE is a secondary audited reproduction; these counts do not establish clean generalization.",
        ],
    }
    return records, aggregate


__all__ = [
    "CLASSES",
    "NORMALIZATION_VERSION",
    "SCHEMA_VERSION",
    "classify_frozen_sample",
    "normalize_exact_text",
]

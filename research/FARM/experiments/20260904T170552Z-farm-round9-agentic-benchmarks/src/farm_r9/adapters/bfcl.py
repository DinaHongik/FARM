"""Pinned BFCL V4 missing-parameter benchmark preparation."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Any

from farm_r9.artifact_io import read_jsonl, sha256_file
from farm_r9.sampling import stratified_sample


BFCL_REPOSITORY_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
BFCL_TEST_SHA256 = "f0c66dda3795f5f53e3e1c0cc8ba0246b6761c8f58bdba8317203bf451ab8838"
BFCL_POSSIBLE_ANSWER_SHA256 = "59c442901779e2c31c33abcd566d032e03736e5ad8069de2fe05489873046ecf"
BFCL_BENCHMARK = "bfcl_v4_multi_turn_miss_param"
BFCL_POPULATION_SIZE = 200
BFCL_SAMPLE_SIZE = 150
BFCL_SAMPLE_QUOTAS = {
    "missing_turn_indices=0": 61,
    "missing_turn_indices=1": 38,
    "missing_turn_indices=1+4+5": 1,
    "missing_turn_indices=1+5": 1,
    "missing_turn_indices=2": 25,
    "missing_turn_indices=3": 18,
    "missing_turn_indices=4": 4,
    "missing_turn_indices=5": 2,
}
_LOG_CREDENTIAL_KEY = re.compile(
    r"(?:password|passwd|api[_-]?key|authorization|secret|access[_-]?token)",
    re.IGNORECASE,
)


def redact_for_log(value: Any) -> Any:
    """Return a log-safe copy while leaving BFCL evaluator input untouched."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _LOG_CREDENTIAL_KEY.search(str(key)) else redact_for_log(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_for_log(child) for child in value]
    if isinstance(value, tuple):
        return tuple(redact_for_log(child) for child in value)
    return value


def _verify_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path.name}: SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _missing_turn_stratum(ground_truth: list[Any]) -> str:
    indices = [str(index) for index, calls in enumerate(ground_truth) if calls == []]
    if not indices:
        raise ValueError("BFCL missing-parameter case has no empty ground-truth turn")
    return f"missing_turn_indices={'+'.join(indices)}"


def _profile_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def prepare_bfcl_missing_parameter(
    test_path: Path,
    possible_answer_path: Path,
    *,
    size: int = 150,
    seed: int = 9052026,
    expected_test_sha256: str = BFCL_TEST_SHA256,
    expected_possible_answer_sha256: str = BFCL_POSSIBLE_ANSWER_SHA256,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return aligned, upstream-compatible rows for a frozen partial evaluation.

    The returned prompt and answer dictionaries are not wrapped or stripped: the
    pinned BFCL evaluator consumes those upstream fields directly.  Callers must
    never place the raw ``initial_config`` in logs; use :func:`redact_for_log`
    when a diagnostic view is unavoidable.
    """
    if size != BFCL_SAMPLE_SIZE:
        raise ValueError(f"BFCL partial protocol requires exactly {BFCL_SAMPLE_SIZE} cases")
    test_hash = _verify_hash(test_path, expected_test_sha256)
    answer_hash = _verify_hash(possible_answer_path, expected_possible_answer_sha256)
    test_rows = read_jsonl(test_path)
    answer_rows = read_jsonl(possible_answer_path)
    if len(test_rows) != BFCL_POPULATION_SIZE or len(answer_rows) != BFCL_POPULATION_SIZE:
        raise ValueError(
            f"BFCL official category must contain {BFCL_POPULATION_SIZE} aligned rows"
        )

    test_ids = [row.get("id") for row in test_rows]
    answer_ids = [row.get("id") for row in answer_rows]
    if test_ids != answer_ids:
        raise ValueError("BFCL prompt and possible-answer IDs are not aligned in upstream order")
    if len(set(test_ids)) != BFCL_POPULATION_SIZE or not all(
        isinstance(case_id, str) and case_id.startswith("multi_turn_miss_param_")
        for case_id in test_ids
    ):
        raise ValueError("BFCL IDs must be unique multi_turn_miss_param identifiers")

    population: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for source_index, (test_row, answer_row) in enumerate(zip(test_rows, answer_rows)):
        questions = test_row.get("question")
        ground_truth = answer_row.get("ground_truth")
        involved_classes = test_row.get("involved_classes")
        if not isinstance(questions, list) or not isinstance(ground_truth, list):
            raise ValueError(f"BFCL row {source_index} has invalid turn containers")
        if len(questions) != len(ground_truth):
            raise ValueError(f"BFCL row {source_index} has mismatched prompt/answer turns")
        if not isinstance(involved_classes, list) or not all(
            isinstance(value, str) and value for value in involved_classes
        ):
            raise ValueError(f"BFCL row {source_index} has invalid involved_classes")
        stratum = _missing_turn_stratum(ground_truth)
        audit = {
            "turn_count": len(questions),
            "involved_class_profile": "+".join(sorted(involved_classes)),
        }
        audit_rows.append(audit)
        population.append({
            "case_id": test_row["id"],
            "stratum": stratum,
            "source_index": source_index,
            "test": test_row,
            "answer": answer_row,
            **audit,
        })

    observed_strata = Counter(row["stratum"] for row in population)
    if set(observed_strata) != set(BFCL_SAMPLE_QUOTAS):
        raise ValueError(f"unexpected BFCL missing-turn strata: {dict(observed_strata)}")
    selected, sample_manifest = stratified_sample(
        population,
        size=size,
        seed=seed,
        benchmark=BFCL_BENCHMARK,
        id_of=lambda row: str(row["case_id"]),
        stratum_of=lambda row: str(row["stratum"]),
        quotas=BFCL_SAMPLE_QUOTAS,
    )
    selected_tests = [row["test"] for row in selected]
    selected_answers = [row["answer"] for row in selected]
    selected_audit = [
        {
            "turn_count": row["turn_count"],
            "involved_class_profile": row["involved_class_profile"],
        }
        for row in selected
    ]
    sample_manifest.update({
        "source_commit": BFCL_REPOSITORY_COMMIT,
        "source_files": {
            test_path.name: test_hash,
            f"possible_answer/{possible_answer_path.name}": answer_hash,
        },
        "evaluation_scope": "partial:150_of_200",
        "official_evaluator_compatible": True,
        "stratum_definition": "indices of turns whose official ground_truth call list is empty",
        "population_turn_counts": _profile_counts(audit_rows, "turn_count"),
        "sample_turn_counts": _profile_counts(selected_audit, "turn_count"),
        "population_involved_class_profiles": _profile_counts(
            audit_rows, "involved_class_profile"
        ),
        "sample_involved_class_profiles": _profile_counts(
            selected_audit, "involved_class_profile"
        ),
        "log_policy": "redact credential-valued fields; evaluator source remains unchanged",
    })
    return selected_tests, selected_answers, sample_manifest

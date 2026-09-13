from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from farm_r9.privacy import DataClassification, PrivacyViolation
from farm_r9.ragas_secondary import (
    IDMetricRecord,
    JudgeRoute,
    PreviewRecord,
    RagasSecondaryError,
    public_secondary_aggregate,
    require_ragas_043,
    score_id_record,
    score_id_records,
    score_preview_faithfulness,
)


RUN_ROOT = Path(__file__).resolve().parents[1]


def _records(n: int = 150) -> list[IDMetricRecord]:
    return [
        IDMetricRecord(f"case-{index}", ("a", "a", "b"), ("a", "c"))
        for index in range(n)
    ]


def test_id_metrics_use_unique_string_ids_like_ragas_043() -> None:
    row = score_id_record(IDMetricRecord("case", ("a", "a", "b"), ("a", "c")))
    assert row["intersection_n"] == 1
    assert row["retrieved_unique_n"] == 2
    assert row["id_context_precision"] == pytest.approx(0.5)
    assert row["id_context_recall"] == pytest.approx(0.5)


def test_id_aggregate_discloses_n_and_perfect_case_numerators() -> None:
    _, aggregate = score_id_records(_records())
    assert aggregate["n"] == 150
    assert aggregate["macro_id_context_precision"] == pytest.approx(0.5)
    assert aggregate["precision_perfect_n"] == 0
    assert aggregate["confidence_intervals_95"]["macro_id_context_precision"] == {
        "low": 0.5,
        "high": 0.5,
    }
    public = public_secondary_aggregate(
        benchmark_label="recipegen_gold", id_aggregate=aggregate
    )
    assert public["n"] == 150
    assert public["raw_numerators"]["id_both_perfect"] == 0
    assert public["raw_denominators"] == {
        "id_precision_perfect": 150,
        "id_recall_perfect": 150,
        "id_both_perfect": 150,
    }
    assert public["confidence_intervals"]["macro_id_context_precision"] == {
        "low": 50.0,
        "high": 50.0,
        "level": 95,
    }
    assert not any("case" in key for key in public)


def test_exact_150_and_nonempty_sets_are_protocol_gates() -> None:
    with pytest.raises(RagasSecondaryError, match="CASE_COUNT_MISMATCH"):
        score_id_records(_records(149))
    with pytest.raises(RagasSecondaryError, match="EMPTY_RETRIEVED_IDS"):
        score_id_record(IDMetricRecord("case", (), ("gold",)))


def test_version_mismatch_has_stable_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.4.2")
    with pytest.raises(RagasSecondaryError) as caught:
        require_ragas_043()
    assert caught.value.code == "RAGAS_VERSION_MISMATCH"


def test_confidential_cloud_judge_is_denied_before_version_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def should_not_run() -> str:
        raise AssertionError("version check must not run")

    monkeypatch.setattr("farm_r9.ragas_secondary.require_ragas_043", should_not_run)
    rows = [PreviewRecord(f"c{i}", "q", "preview", ("context",)) for i in range(150)]
    with pytest.raises(PrivacyViolation, match="local judge"):
        asyncio.run(
            score_preview_faithfulness(
                rows,
                evaluator_llm=object(),
                classification=DataClassification.CONFIDENTIAL,
                judge_route=JudgeRoute.CLOUD,
            )
        )


def test_faithfulness_counts_invalid_scores_as_protocol_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("farm_r9.ragas_secondary.require_ragas_043", lambda: "0.4.3")
    rows = [PreviewRecord(f"c{i}", "q", "preview", ("context",)) for i in range(150)]

    async def fake(record: PreviewRecord) -> float:
        return float("nan") if record.case_id == "c0" else 0.75

    private, aggregate = asyncio.run(
        score_preview_faithfulness(
            rows,
            evaluator_llm=object(),
            classification=DataClassification.PUBLIC,
            judge_route=JudgeRoute.CLOUD,
            metric_factory=lambda _: fake,
        )
    )
    assert private[0]["status"] == "protocol_failure"
    assert aggregate["faithfulness_scored_n"] == 149
    assert aggregate["faithfulness_protocol_failure_n"] == 1
    assert aggregate["faithfulness_mean"] == pytest.approx(0.75)
    _, id_aggregate = score_id_records(_records())
    public = public_secondary_aggregate(
        benchmark_label="recipegen_gold",
        id_aggregate=id_aggregate,
        faithfulness_aggregate=aggregate,
    )
    assert public["raw_numerators"]["faithfulness_scored"] == 149
    assert public["raw_denominators"]["faithfulness_scored"] == 150
    assert public["failure_counts"]["faithfulness_protocol_failure"] == 1
    assert public["percentages"]["faithfulness_mean_scored_cases"] == 75.0


def test_cli_writes_private_artifacts_mode_0600_and_public_is_aggregate_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "case_id": f"case-{index}",
                    "retrieved_context_ids": ["a", "b"],
                    "reference_context_ids": ["a"],
                }
            )
            + "\n"
            for index in range(150)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_ROOT / "scripts" / "run_ragas_secondary.py"),
            "--input",
            str(source),
            "--output-dir",
            str(output),
            "--benchmark-label",
            "private_farm_aggregate",
            "--classification",
            "confidential",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.stat(output / "records_private.jsonl").st_mode & 0o777 == 0o600
    assert os.stat(output / "aggregate_private.json").st_mode & 0o777 == 0o600
    public = json.loads((output / "aggregate_public.json").read_text(encoding="utf-8"))
    assert public["n"] == 150
    assert set(public) == {"n", "raw_numerators", "raw_denominators", "percentages",
                           "confidence_intervals", "failure_counts", "hashes"}
    assert set(public["percentages"]) == set(public["raw_numerators"])
    assert set(public["hashes"]) == {"source_artifact_sha256"}
    serialized = json.dumps(public).casefold()
    assert "case-" not in serialized
    assert "retrieved_context_ids" not in serialized

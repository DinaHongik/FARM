from __future__ import annotations

import copy
import json
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.yao_agent import COMPONENTS, aggregate_records  # noqa: E402
from farm_r9.yao_reporting import (  # noqa: E402
    YaoPublicReportError,
    export_public_yao_aggregate,
)


STRATA = (("CI", 28), ("VI-1", 18), ("VI-2", 32), ("VI-3", 10), ("VI-4", 62))


def _records(*, reference: bool = False) -> list[dict]:
    records: list[dict] = []
    index = 0
    for stratum, count in STRATA:
        for _ in range(count):
            current_correct = index < 100
            reference_correct = index < 80 or 100 <= index < 110
            correct = reference_correct if reference else current_correct
            question_count = index % 5
            questions = list(COMPONENTS[:question_count])
            records.append(
                {
                    "schema_version": "round9-yao-case-result-v1",
                    "case_id": f"yao:{index}",
                    "stratum": stratum,
                    "arm": "same_model_one_shot"
                    if reference
                    else "bounded_clarification_agent",
                    "terminal": True,
                    "terminal_status": "committed",
                    "failure_code": None,
                    "scores": {
                        **{component: correct for component in COMPONENTS},
                        "joint": correct,
                    },
                    "questions": questions,
                    "question_count": question_count,
                    "ask_policy": {
                        "true_positive": question_count,
                        "false_positive": 0,
                        "false_negative": 0,
                        "true_negative": 4 - question_count,
                    },
                    "usage": {
                        "semantic_calls": question_count + 1,
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "physical_attempts": question_count + 1,
                        "provider_latency_ms": 20.0,
                        "queue_wait_ms": 2.0,
                        "strict_json_outputs": question_count + 1,
                        "cache_hits": 0,
                    },
                }
            )
            index += 1
    return records


def _aggregate(*, paired: bool = True) -> dict:
    aggregate = aggregate_records(
        _records(),
        intended_n=150,
        paired_reference=_records(reference=True) if paired else None,
    )
    aggregate.update(
        {
            "benchmark_label": "interactive_ifttt",
            "arm": "bounded_clarification_agent",
            "hashes": {
                "sample_sha256": "5543d5e331033147a960392c474f5bb2b1560e2d9a7ca1942e8de8d06aa345ff",
                "catalog_sha256": "b" * 64,
                "prompt_sha256": "c" * 64,
            },
            "model_metadata": {
                "name": "deepseek-v4-flash:0731",
                "provider": "ollama_cloud",
            },
            "protocol_metadata": {
                "version": "round9-yao-v1",
                "arm": "bounded_clarification_agent",
                "temperature": 0,
                "seed": 42,
                "questions_max": 4,
                "semantic_calls_max": 5,
                "repairs_max": 0,
                "failure_policy": "terminal_incorrect_no_fallback",
                "answer_policy": "frozen_official_pool_sha256_v1",
            },
            "cost": {
                "currency": "USD",
                "prompt_cost_per_million": 0.2,
                "completion_cost_per_million": 0.4,
                "estimated_total": (1500 * 0.2 + 750 * 0.4) / 1_000_000,
            },
        }
    )
    return aggregate


def test_public_report_is_complete_paired_and_aggregate_only() -> None:
    public = export_public_yao_aggregate(
        _aggregate(),
        current_records=_records(),
        reference_records=_records(reference=True),
    )

    assert public["n"] == 150
    assert public["evaluation_scope"] == "partial:150_of_3870"
    assert public["raw_numerators"]["joint"] == 100
    assert public["raw_denominators"]["joint"] == 150
    assert public["failure_denominator"] == 150
    assert set(public["raw_numerators"]) == {*COMPONENTS, "joint"}
    assert public["confidence_intervals"]["joint"]["method"] == "wilson_95"
    assert {key: value["n"] for key, value in public["by_stratum"].items()} == dict(
        STRATA
    )
    assert public["questions"]["total"] == 300
    assert public["questions"]["ask_policy_confusion"]["true_positive"] == 300
    assert public["usage"]["totals"]["semantic_calls"] == 450
    assert public["usage"]["totals"]["total_tokens"] == 2250
    assert public["usage"]["totals"]["provider_latency_ms"] == 3000.0
    assert public["cost"]["available"] is True
    pair = public["paired_outcomes"]["metrics"]["joint"]
    assert pair["rescue_count"] == 20
    assert pair["regression_count"] == 10
    assert pair["delta_percentage_points"] == 100 / 15
    assert pair["delta_confidence_interval_95"] == {
        "method": "stratified_paired_case_bootstrap_percentile",
        "confidence_level": 0.95,
        "resamples": 10000,
        "seed": 9052026,
        "rng": "python_random_mt19937_v2",
        "quantile_method": "hyndman_fan_type_7",
        "experimental_unit": "paired_case",
        "stratified": True,
        "stratification_variable": "frozen_ambiguity_stratum",
        "stratum_counts": dict(STRATA),
        "observed_delta_percentage_points": 6.666667,
        "low_percentage_points": 0.666667,
        "high_percentage_points": 13.333333,
    }
    assert 0 <= pair["mcnemar_exact_two_sided_p"] <= 1
    assert abs(pair["mcnemar_exact_two_sided_p"] - 0.09873714670538902) < 1e-15
    assert (
        public["paired_outcomes"]["multiplicity_policy"]["confirmatory_metric"]
        == "joint"
    )
    reversed_public = export_public_yao_aggregate(
        _aggregate(),
        current_records=list(reversed(_records())),
        reference_records=_records(reference=True),
    )
    assert (
        reversed_public["paired_outcomes"]["metrics"]["joint"][
            "delta_confidence_interval_95"
        ]
        == pair["delta_confidence_interval_95"]
    )

    serialized = json.dumps(public, sort_keys=True)
    for forbidden in (
        "yao:0",
        "case_id",
        "records",
        "events",
        "prediction",
        "transcript",
        "official_answer",
        "Save new email attachments",
    ):
        assert forbidden not in serialized


def test_exporter_rejects_unknown_fields_instead_of_silently_dropping() -> None:
    for path in ("top", "nested"):
        source = copy.deepcopy(_aggregate())
        if path == "top":
            source["events"] = [{"query": "must not escape"}]
        else:
            source["questions"]["case_ids"] = ["yao:secret"]
        try:
            export_public_yao_aggregate(
                source,
                current_records=_records(),
                reference_records=_records(reference=True),
            )
        except YaoPublicReportError:
            pass
        else:
            raise AssertionError("content-bearing field was not rejected")


def test_exporter_recomputes_metrics_usage_cost_and_paired_statistics() -> None:
    mutations = (
        lambda source: source["percentages"].__setitem__("joint", 99.0),
        lambda source: source["by_stratum"]["CI"]["raw_numerators"].__setitem__(
            "joint", 0
        ),
        lambda source: source["usage"]["means_per_case"].__setitem__(
            "semantic_calls", 99.0
        ),
        lambda source: source["cost"].__setitem__("estimated_total", 99.0),
        lambda source: source["paired_outcomes"]["joint"].__setitem__(
            "current_only", 21
        ),
        lambda source: source["paired_outcomes"]["joint"].__setitem__(
            "mcnemar_exact_two_sided_p", 0.5
        ),
    )
    for mutate in mutations:
        source = copy.deepcopy(_aggregate())
        mutate(source)
        try:
            export_public_yao_aggregate(
                source,
                current_records=_records(),
                reference_records=_records(reference=True),
            )
        except YaoPublicReportError:
            pass
        else:
            raise AssertionError("arithmetically inconsistent aggregate was accepted")


def test_private_aggregator_rejects_score_coercion_and_inconsistent_joint() -> None:
    records = _records()
    records[0]["scores"]["joint"] = "false"
    try:
        aggregate_records(records, intended_n=150)
    except ValueError as error:
        assert "JSON booleans" in str(error)
    else:
        raise AssertionError("truthy string score was counted as correct")

    records = _records()
    records[0]["scores"]["joint"] = False
    try:
        aggregate_records(records, intended_n=150)
    except ValueError as error:
        assert "inconsistent joint" in str(error)
    else:
        raise AssertionError("joint score inconsistent with components was accepted")


def test_paired_export_requires_and_revalidates_both_terminal_ledgers() -> None:
    source = _aggregate()
    current = _records()
    reference = _records(reference=True)
    for kwargs in ({}, {"current_records": current}, {"reference_records": reference}):
        try:
            export_public_yao_aggregate(source, **kwargs)
        except YaoPublicReportError:
            pass
        else:
            raise AssertionError(
                "paired export accepted missing terminal-ledger evidence"
            )

    malformed_ledgers = []
    duplicate = copy.deepcopy(current)
    duplicate[-1]["case_id"] = duplicate[0]["case_id"]
    malformed_ledgers.append(duplicate)
    wrong_arm = copy.deepcopy(current)
    wrong_arm[0]["arm"] = "same_model_one_shot"
    malformed_ledgers.append(wrong_arm)
    wrong_score = copy.deepcopy(current)
    wrong_score[0]["scores"] = {metric: False for metric in (*COMPONENTS, "joint")}
    malformed_ledgers.append(wrong_score)
    wrong_stratum = copy.deepcopy(current)
    wrong_stratum[0]["stratum"] = "VI-1"
    malformed_ledgers.append(wrong_stratum)
    protocol_failure_scored_correct = copy.deepcopy(current)
    protocol_failure_scored_correct[0]["terminal_status"] = "protocol_failure"
    protocol_failure_scored_correct[0]["failure_code"] = "no_commit_within_budget"
    malformed_ledgers.append(protocol_failure_scored_correct)
    for malformed in malformed_ledgers:
        try:
            export_public_yao_aggregate(
                source,
                current_records=malformed,
                reference_records=reference,
            )
        except YaoPublicReportError as error:
            assert "yao:0" not in str(error)
        else:
            raise AssertionError("paired export accepted a malformed terminal ledger")


def test_paired_rescue_direction_is_fixed_to_agent_over_one_shot() -> None:
    source = _aggregate()
    source["arm"] = "same_model_one_shot"
    source["protocol_metadata"].update(
        {
            "arm": "same_model_one_shot",
            "questions_max": 0,
            "semantic_calls_max": 1,
        }
    )
    try:
        export_public_yao_aggregate(
            source,
            current_records=_records(),
            reference_records=_records(reference=True),
        )
    except YaoPublicReportError:
        pass
    else:
        raise AssertionError(
            "reverse-direction paired comparison was labelled as rescue"
        )


def test_exporter_enforces_arm_budgets_and_protocol_failure_denominator() -> None:
    one_shot = _aggregate(paired=False)
    one_shot["arm"] = "same_model_one_shot"
    one_shot["protocol_metadata"].update(
        {
            "arm": "same_model_one_shot",
            "questions_max": 0,
            "semantic_calls_max": 1,
        }
    )
    try:
        export_public_yao_aggregate(one_shot)
    except YaoPublicReportError as error:
        assert "questions exceeds" in str(error)
    else:
        raise AssertionError("one-shot export accepted clarification questions")

    source = _aggregate(paired=False)
    source["usage"]["totals"]["semantic_calls"] = 751
    source["usage"]["means_per_case"]["semantic_calls"] = 751 / 150
    try:
        export_public_yao_aggregate(source)
    except YaoPublicReportError as error:
        assert "semantic_calls exceeds" in str(error)
    else:
        raise AssertionError("bounded export accepted calls beyond the frozen budget")

    source = _aggregate(paired=False)
    source["terminal_status_counts"] = {"committed": 99, "protocol_failure": 51}
    source["failure_counts"] = {"no_commit_within_budget": 51}
    try:
        export_public_yao_aggregate(source)
    except YaoPublicReportError as error:
        assert "protocol failure as correct" in str(error)
    else:
        raise AssertionError("export accepted more correct cases than committed cases")


def test_cli_exclusively_writes_mode_0600(tmp_path: Path) -> None:
    source = tmp_path / "private-aggregate.json"
    current_records = tmp_path / "current-records.jsonl"
    reference_records = tmp_path / "reference-records.jsonl"
    output = tmp_path / "public" / "aggregate.json"
    source.write_text(json.dumps(_aggregate()), encoding="utf-8")
    current_records.write_text(
        "".join(json.dumps(record) + "\n" for record in _records()),
        encoding="utf-8",
    )
    reference_records.write_text(
        "".join(json.dumps(record) + "\n" for record in _records(reference=True)),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "export_yao_public_aggregate.py"),
        "--input",
        str(source),
        "--current-records",
        str(current_records),
        "--reference-records",
        str(reference_records),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8"))["n"] == 150

    original = output.read_bytes()
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    assert second.returncode == 2
    assert output.read_bytes() == original

import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/collect_astra_public_results.py"
SPEC = importlib.util.spec_from_file_location("astra_collection", PATH)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def fixture():
    rows = [{"case_id": str(i), "terminal": True, "stratum": "CI",
        "scores": {"joint": i < 30}, "usage": {"semantic_calls": 1},
        "failure_code": None, "question_count": 0} for i in range(150)]
    expected = {row["case_id"]: {"stratum": "CI"} for row in rows}
    aggregate = {"n": 150, "raw_numerators": {"joint": 30}, "percentages": {"joint": 20.0}}
    return rows, expected, aggregate


def test_complete_summary_is_aggregate_only():
    rows, expected, aggregate = fixture()
    report = collector.summarize(rows, aggregate, expected, kind="yao")
    assert report["raw_numerators"]["joint"] == 30
    assert report["usage_means_per_case"]["semantic_calls"] == 1
    assert "case_id" not in json.dumps(report)
    assert "scores" not in json.dumps(report)


@pytest.mark.parametrize("corruption", ["short", "duplicate", "nonterminal", "foreign", "aggregate", "boolean", "stratum"])
def test_collection_rejects_incomplete_or_inconsistent_evidence(corruption):
    rows, expected, aggregate = fixture()
    if corruption == "short":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1]["case_id"] = rows[0]["case_id"]
    elif corruption == "foreign":
        rows[-1]["case_id"] = "foreign"
    elif corruption == "nonterminal":
        rows[-1]["terminal"] = False
    elif corruption == "aggregate":
        aggregate["raw_numerators"]["joint"] = 31
    elif corruption == "boolean":
        rows[0]["scores"]["joint"] = 1
    else:
        rows[0]["stratum"] = "VI-4"
    with pytest.raises(ValueError):
        collector.summarize(rows, aggregate, expected, kind="yao")


def test_missing_aggregate_is_pending_not_an_interim_score(tmp_path):
    rows, expected, _ = fixture()
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert collector.load_complete(tmp_path, expected, kind="yao") is None


def test_paired_delta_includes_regressions_and_stratified_uncertainty():
    reference, _, _ = fixture()
    current = json.loads(json.dumps(reference))
    current[0]["scores"]["joint"] = False
    current[30]["scores"]["joint"] = True
    current[31]["scores"]["joint"] = True
    result = collector.compare_rows(current, reference, kind="yao")
    assert result["rescues"] == 2 and result["regressions"] == 1
    assert result["paired_delta"]["stratified"] is True
    assert result["paired_delta"]["observed_delta_percentage_points"] == pytest.approx(2 / 3, abs=1e-6)


def test_completed_report_is_never_overwritten(tmp_path):
    path = tmp_path / "report.json"
    collector.save_once(path, {"n": 150})
    collector.save_once(path, {"n": 150})
    with pytest.raises(ValueError):
        collector.save_once(path, {"n": 149})
    assert json.loads(path.read_text()) == {"n": 150}
    assert path.stat().st_mode & 0o777 == 0o600

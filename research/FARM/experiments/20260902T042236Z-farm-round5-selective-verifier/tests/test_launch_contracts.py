from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_matrix_keeps_gpu_four_in_reserve() -> None:
    matrix = json.loads((ROOT / "EXPERIMENT_MATRIX.json").read_text())
    assert [row["gpu"] for row in matrix["gpu_experiments"]] == [0, 1, 2, 3]
    assert len({row["id"] for row in matrix["gpu_experiments"]}) == 4


def test_new_agent_work_does_not_touch_sealed_slice() -> None:
    matrix = json.loads((ROOT / "EXPERIMENT_MATRIX.json").read_text())
    exploratory = matrix["splits"]["agent_exploratory"]
    sealed = matrix["splits"]["future_confirmation_sealed"]
    assert (exploratory["slice_start"], exploratory["slice_stop"]) == (0, 300)
    assert (sealed["slice_start"], sealed["slice_stop"]) == (600, 900)
    assert all(arm["scope"] == 300 for arm in matrix["agent_arms"])


def test_agent_protocol_is_bounded_and_conservative() -> None:
    protocol = json.loads((ROOT / "EXPERIMENT_MATRIX.json").read_text())["agent_protocol"]
    assert protocol["max_logical_calls_per_case"] == 3
    assert protocol["max_provider_attempts_per_logical_call"] == 2
    assert protocol["safe_fallback"] == "retrieval_top1"
    assert protocol["proposer_view"] == "plain"
    assert protocol["verifier_view"] == "schema"


def test_launchers_use_detached_sessions_and_expected_interpreter() -> None:
    for name in ("launch_gpu_jobs.sh", "launch_agent_jobs.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "tmux" in text
        assert "new-session" in text
    for name in ("run_gpu_job.sh", "run_agent_job.sh"):
        text = (ROOT / "scripts" / name).read_text()
        assert "/raid/session/aicontents/farm/.venv/bin/python" in text


def test_gpu_launcher_fails_closed_on_occupancy() -> None:
    text = (ROOT / "scripts" / "launch_gpu_jobs.sh").read_text()
    assert "--query-compute-apps=gpu_uuid,pid" in text
    assert "memory_mib > 64" in text
    assert "utilization > 0" in text
    assert "nothing launched" in text


def test_agent_job_passes_all_frozen_inputs_and_exact_scope() -> None:
    text = (ROOT / "scripts" / "run_agent_job.sh").read_text()
    for argument in (
        "--candidates",
        "--candidate-manifest",
        "--pair-plain-results",
        "--pair-schema-results",
        "--data-root",
        "--slice-start 0",
        "--slice-stop 300",
    ):
        assert argument in text


def test_agent_launcher_requires_a_valid_one_case_smoke() -> None:
    text = (ROOT / "scripts" / "launch_agent_jobs.sh").read_text()
    for gate in (
        'value.get("status") != "completed"',
        'binding.get("mode") != "smoke_consumed_dev"',
        'get("target_rows") != 1',
        'protocol.get("valid_rows") != 1',
    ):
        assert gate in text

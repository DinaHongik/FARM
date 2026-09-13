from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_four_gpu_and_four_agent_arms_are_distinct():
    matrix = json.loads((ROOT / "EXPERIMENT_MATRIX.json").read_text())
    gpu = matrix["gpu_experiments"]
    agent = matrix["agent_arms"]
    assert [item["gpu"] for item in gpu] == [0, 1, 2, 3]
    assert len({item["id"] for item in gpu}) == 4
    assert len({item["id"] for item in agent}) == 4
    assert all(item["scope"] == 231 for item in agent)


def test_confirmation_is_reserved_and_test_is_locked():
    matrix = json.loads((ROOT / "EXPERIMENT_MATRIX.json").read_text())
    assert matrix["splits"]["agent_confirmation_reserved"]["status"].startswith("reserved")
    assert matrix["locked_test_policy"].startswith("never")

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("public_transfer", SCRIPTS / "audit_farm_public_transfer.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_ties_have_stable_identity_order():
    assert module.ordered_ranking(np.array([[1., 1., 0.]]), ["b", "a", "c"]) == [["a", "b", "c"]]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_cannot_be_scored_perfect(value):
    with pytest.raises(ValueError, match="non-finite"):
        module.ordered_ranking(np.array([[value, .5]]), ["a", "b"])


def test_duplicate_identity_rejected():
    with pytest.raises(ValueError):
        module.ordered_ranking(np.array([[1., .5]]), ["a", "a"])


def test_matching_requires_correct_service_and_function():
    docs = {"a": {"service": "s", "function": "f"}, "b": {"service": "wrong", "function": "f"},
            "c": {"service": "a", "function": "do"}}
    gold = {"trigger_channel_norm": "s", "trigger_function_norm": "f",
            "action_channel_norm": "a", "action_function_norm": "do"}
    result = module.score_ranking(gold, {"trigger": ["b", "a"], "action": ["c"]}, docs, (1, 5))
    assert result["joint_r1"] is False
    assert result["joint_r5"] is True
    assert result["trigger_r1"] is False
    assert result["action_r1"] is True

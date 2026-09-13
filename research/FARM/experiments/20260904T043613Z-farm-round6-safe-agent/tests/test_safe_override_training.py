from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_safe_override import challenger_pairs, training_examples, validate_config  # noqa: E402


def config():
    return json.loads((ROOT / "configs/function_plain_safe.json").read_text())


def candidate(side, index):
    return {
        "url": f"{side}://{index}", "channel": f"service-{index}",
        "function_name": f"fn-{index}", "retrieval_rank": index + 1,
        "retrieval_score": 10 - index, "text_plain": f"plain {index}",
        "text_schema": f"schema {index}",
    }


def rows():
    base = {
        "group_id": "recoverable", "query": "q",
        "trigger_candidates": [candidate("t", i) for i in range(10)],
        "action_candidates": [candidate("a", i) for i in range(10)],
        "valid_pairs": [{"trigger_url": "t://1", "action_url": "a://1"}],
        "trigger_union_candidates": [], "action_union_candidates": [],
    }
    keep = {
        "group_id": "keep", "query": "q2",
        "trigger_candidates": [candidate("t2", i) for i in range(10)],
        "action_candidates": [candidate("a2", i) for i in range(10)],
        "valid_pairs": [{"trigger_url": "t2://0", "action_url": "a2://0"}],
        "trigger_union_candidates": [], "action_union_candidates": [],
    }
    return [base, keep]


def test_configs_are_valid_and_training_only():
    for path in sorted((ROOT / "configs").glob("*.json")):
        assert validate_config(json.loads(path.read_text()))["dataset_id"]


def test_dev_path_is_rejected():
    value = config()
    value["train_candidates"] = "/tmp/dev.json"
    try:
        validate_config(value)
    except ValueError as error:
        assert "reranker_train" in str(error)
    else:
        raise AssertionError("development labels entered training")


def test_training_labels_only_strict_recoveries_positive():
    columns, summary = training_examples(rows(), config())
    assert 0 in columns["label"] and 1 in columns["label"]
    assert summary["rows_with_recoverable_challenger"] == 1
    assert summary["gold_injection_at_inference"] is False


def test_hard_order_keeps_complete_top10_lattice():
    value = config() | {"candidate_policy": "endpoint_hard"}
    pairs = challenger_pairs(rows()[0], value)
    assert len(pairs) == 99
    assert len({(p[0]["url"], p[1]["url"]) for p in pairs}) == 99

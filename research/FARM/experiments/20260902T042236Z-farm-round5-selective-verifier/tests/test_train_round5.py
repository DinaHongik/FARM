from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("train_round5", ROOT / "scripts" / "train_round5.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def candidate(side: str, rank: int, channel: str | None = None) -> dict:
    channel = channel or f"{side}_service_{rank}"
    url = f"https://ifttt.com/{channel}/{side}s/{side}_{rank}"
    return {
        "url": url,
        "channel": channel,
        "function_name": f"{side} function {rank}",
        "text_plain": f"channel: {channel} | function: {side} function {rank}\nDescription {rank}.",
        "text_schema": f"channel: {channel} | function: {side} function {rank}\nDescription {rank}.\nFields: value.",
        "retrieval_rank": rank,
        "retrieval_score": 1.0 / (60 + rank),
    }


def row(gold_trigger_rank: int = 2, gold_action_rank: int = 2, *, same_channels: bool = True) -> dict:
    triggers = [candidate("trigger", rank) for rank in range(1, 11)]
    actions = [candidate("action", rank) for rank in range(1, 11)]
    if same_channels:
        triggers[1]["channel"] = "gold_trigger_service"
        triggers[2]["channel"] = "gold_trigger_service"
        actions[1]["channel"] = "gold_action_service"
        actions[2]["channel"] = "gold_action_service"
    if gold_trigger_rank > 10:
        trigger_url = "https://ifttt.com/outside/triggers/outside"
    else:
        trigger_url = triggers[gold_trigger_rank - 1]["url"]
    if gold_action_rank > 10:
        action_url = "https://ifttt.com/outside/actions/outside"
    else:
        action_url = actions[gold_action_rank - 1]["url"]
    return {
        "group_id": f"q_{gold_trigger_rank}_{gold_action_rank}",
        "query": "When a precise event occurs, perform the precise consequence",
        "trigger_candidates": triggers,
        "action_candidates": actions,
        "valid_pairs": [{"trigger_url": trigger_url, "action_url": action_url}],
    }


def load_config(name: str) -> dict:
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


class Round5TrainerContractTests(unittest.TestCase):
    def test_all_four_configs_validate_and_bind_unique_gpus(self) -> None:
        names = (
            "pair_guard_plain.json",
            "pair_guard_actionhard.json",
            "endpoint_split_actionguard.json",
            "semantic_depth_router.json",
        )
        configs = [MODULE.validate_config(load_config(name)) for name in names]
        self.assertEqual({config["gpu"] for config in configs}, {0, 1, 2, 3})
        self.assertEqual(
            {config["trainer_kind"] for config in configs}, {"pair", "endpoint", "depth"}
        )
        self.assertTrue(all("reranker_train" in config["train_candidates"] for config in configs))

    def test_config_rejects_development_and_test_candidate_paths(self) -> None:
        for basename in ("dev.json", "test.json", "validation.json"):
            config = load_config("pair_guard_plain.json")
            config["train_candidates"] = f"/tmp/{basename}"
            with self.assertRaises(ValueError):
                MODULE.validate_config(config)

    def test_hash_holdout_is_deterministic_and_disjoint(self) -> None:
        group_ids = [f"q_{index:04d}" for index in range(1000)]
        folds = [
            {group_id for group_id in group_ids if MODULE.stable_holdout(group_id, 5, fold)}
            for fold in range(5)
        ]
        self.assertEqual(set().union(*folds), set(group_ids))
        self.assertTrue(all(folds[left].isdisjoint(folds[right]) for left in range(5) for right in range(left + 1, 5)))
        self.assertEqual(folds[0], {group_id for group_id in group_ids if MODULE.stable_holdout(group_id, 5, 0)})

    def test_balanced_pair_policy_keeps_wrong_baseline_and_never_gold(self) -> None:
        sample = row(2, 2)
        valid = MODULE.valid_pair_set(sample)
        selected, counts = MODULE.select_pair_negatives(
            sample,
            policy="balanced_guard",
            quotas={"action_confounder": 4, "trigger_confounder": 4, "same_service_wrong_function": 4},
            limit=20,
        )
        ids = [MODULE.pair_identifier(pair) for pair in selected]
        baseline = (sample["trigger_candidates"][0]["url"], sample["action_candidates"][0]["url"])
        self.assertEqual(ids[0], baseline)
        self.assertTrue(all(identity not in valid for identity in ids))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(counts["action_confounder"], 0)
        self.assertGreater(counts["trigger_confounder"], 0)

    def test_action_guard_prioritizes_action_errors(self) -> None:
        sample = row(2, 2)
        selected, counts = MODULE.select_pair_negatives(
            sample,
            policy="action_guard",
            quotas={
                "action_confounder": 9,
                "action_wrong": 8,
                "trigger_confounder": 3,
                "same_service_wrong_function": 3,
            },
            limit=24,
        )
        gold_action = sample["action_candidates"][1]["url"]
        wrong_action_count = sum(pair[1]["url"] != gold_action for pair in selected)
        self.assertGreaterEqual(wrong_action_count, 16)
        self.assertEqual(counts["action_confounder"], 9)
        self.assertEqual(len(selected), 24)

    def test_pair_assembly_does_not_inject_unretrieved_gold(self) -> None:
        config = load_config("pair_guard_plain.json")
        columns, stats = MODULE.make_pair_training_columns([row(11, 11)], config)
        self.assertEqual(columns, {"query": [], "docs": [], "labels": []})
        self.assertEqual(stats["excluded_without_positive"], 1)
        self.assertFalse(stats["gold_injection"])

    def test_retention_groups_are_duplicated_only_when_top1_is_gold(self) -> None:
        config = load_config("pair_guard_plain.json")
        correct_columns, correct_stats = MODULE.make_pair_training_columns([row(1, 1)], config)
        wrong_columns, wrong_stats = MODULE.make_pair_training_columns([row(2, 2)], config)
        self.assertEqual(len(correct_columns["query"]), 2)
        self.assertEqual(correct_stats["retention_duplicates"], 1)
        self.assertEqual(len(wrong_columns["query"]), 1)
        self.assertEqual(wrong_stats["retention_duplicates"], 0)

    def test_action_endpoint_mining_prefers_same_service_wrong_function(self) -> None:
        sample = row(2, 2)
        negatives = MODULE.select_endpoint_negatives(sample, "action", 3)
        self.assertEqual(negatives[0]["retrieval_rank"], 1)  # wrong baseline is always guarded
        self.assertEqual(negatives[1]["channel"], "gold_action_service")
        gold_action = sample["action_candidates"][1]["url"]
        self.assertTrue(all(item["url"] != gold_action for item in negatives))

    def test_depth_bucket_uses_joint_pair_rank_not_independent_side_labels(self) -> None:
        self.assertEqual(MODULE.depth_bucket(row(1, 1)), 0)
        self.assertEqual(MODULE.depth_bucket(row(2, 5)), 1)
        self.assertEqual(MODULE.depth_bucket(row(7, 3)), 2)
        self.assertEqual(MODULE.depth_bucket(row(11, 2)), 3)

    def test_trainer_source_hash_is_bound_before_checkpointing(self) -> None:
        source = (ROOT / "scripts" / "train_round5.py").read_text(encoding="utf-8")
        self.assertIn('"trainer_sha256": sha256_file(Path(__file__).resolve())', source)


if __name__ == "__main__":
    unittest.main()

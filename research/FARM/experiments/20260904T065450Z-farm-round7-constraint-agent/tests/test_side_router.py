from __future__ import annotations

import copy
import json
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_side_router import (  # noqa: E402
    CANDIDATE_SHA256,
    CLASS_NAMES,
    CUBLAS_WORKSPACE_CONFIG,
    DATASET_ID,
    DETERMINISM_CONTRACT_VERSION,
    MANIFEST_SHA256,
    TRAIN_SPLIT_SHA256,
    assemble_training_examples,
    balanced_class_weights,
    configure_pre_torch_environment,
    configure_torch_determinism,
    evaluate_predictions,
    family_fold,
    inference_text,
    join_semantic_families,
    partition_family_fold,
    side_label,
    validate_config,
)


class _FakeCudaAttentionBackend:
    def __init__(self) -> None:
        self.flash = True
        self.memory_efficient = True
        self.cudnn = True
        self.math = False

    def enable_flash_sdp(self, enabled: bool) -> None:
        self.flash = enabled

    def flash_sdp_enabled(self) -> bool:
        return self.flash

    def enable_mem_efficient_sdp(self, enabled: bool) -> None:
        self.memory_efficient = enabled

    def mem_efficient_sdp_enabled(self) -> bool:
        return self.memory_efficient

    def enable_cudnn_sdp(self, enabled: bool) -> None:
        self.cudnn = enabled

    def cudnn_sdp_enabled(self) -> bool:
        return self.cudnn

    def enable_math_sdp(self, enabled: bool) -> None:
        self.math = enabled

    def math_sdp_enabled(self) -> bool:
        return self.math


class _FakeCuda:
    def __init__(self) -> None:
        self.seed = None

    def manual_seed_all(self, seed: int) -> None:
        self.seed = seed


class _FakeCudnn:
    benchmark = True
    deterministic = False


class _FakeTorch:
    __version__ = "fake-torch"

    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.backends = type(
            "FakeBackends", (),
            {"cuda": _FakeCudaAttentionBackend(), "cudnn": _FakeCudnn()},
        )()
        self.version = type("FakeVersion", (), {"cuda": "fake-cuda"})()
        self.seed = None
        self.deterministic = False
        self.warn_only = True

    def manual_seed(self, seed: int) -> None:
        self.seed = seed

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.deterministic = enabled
        self.warn_only = warn_only

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only


def endpoint(side: str, index: int) -> dict:
    return {
        "url": f"https://ifttt.com/service_{side}/{side}s/function_{index}",
        "channel": f"service_{side}",
        "function_name": f"{side} function {index}",
        "retrieval_rank": index + 1,
        "retrieval_score": 1.0 / (index + 1),
        "text_plain": f"public plain {side} evidence {index}",
        "text_schema": f"public schema {side} evidence {index}",
    }


def row(group_id: str = "g1", valid_pairs: list[dict] | None = None) -> dict:
    triggers = [endpoint("trigger", index) for index in range(10)]
    actions = [endpoint("action", index) for index in range(10)]
    return {
        "group_id": group_id,
        "query": "When an event occurs, perform the requested action",
        "trigger_candidates": triggers,
        "action_candidates": actions,
        "valid_pairs": valid_pairs or [
            {"trigger_url": triggers[0]["url"], "action_url": actions[0]["url"]}
        ],
    }


def pair(trigger_index: int, action_index: int) -> dict:
    return {
        "trigger_url": endpoint("trigger", trigger_index)["url"],
        "action_url": endpoint("action", action_index)["url"],
    }


def config(fold: int = 0) -> dict:
    return {
        "run_id": "20260904T065450Z-farm-round7-constraint-agent",
        "experiment_id": f"side_router_fold{fold}",
        "gpu": fold,
        "dataset_id": DATASET_ID,
        "base_model": "/models/bge",
        "base_model_revision": "revision",
        "train_candidates": "/raid/farm/experiments/round3/function_rrf/reranker_train.json",
        "train_candidates_sha256": CANDIDATE_SHA256,
        "train_manifest": "/raid/farm/experiments/round3/function_rrf/reranker_train.manifest.json",
        "train_manifest_sha256": MANIFEST_SHA256,
        "train_split": "/raid/farm/data/v2/splits/reranker_train.json",
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "expected_rows": 1145,
        "fold_modulus": 4,
        "holdout_fold": fold,
        "family_key": "semantic_family_id",
        "class_names": list(CLASS_NAMES),
        "num_labels": 4,
        "class_balance": "inverse_frequency_cross_entropy",
        "evidence_views": ["plain", "schema"],
        "chars_per_view": 700,
        "route_threshold": 0.5,
        "calibration_bins": 5,
        "routing_thresholds": [0.5, 0.7, 0.9],
        "epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.00002,
        "warmup_ratio": 0.1,
        "max_length": 128,
        "prediction_batch_size": 8,
        "seed": 42,
        "fp16": False,
    }


class SideRouterTests(unittest.TestCase):
    def test_pre_torch_environment_contract_is_dependency_free_and_fail_closed(self):
        environment: dict[str, str] = {}
        declaration = configure_pre_torch_environment(environment)
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertEqual(declaration["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_WORKSPACE_CONFIG)
        self.assertEqual(declaration["contract_version"], DETERMINISM_CONTRACT_VERSION)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], CUBLAS_WORKSPACE_CONFIG)

        with self.assertRaisesRegex(RuntimeError, "conflicts with the pinned"):
            configure_pre_torch_environment({"CUBLAS_WORKSPACE_CONFIG": ":16:8"})

    def test_strict_torch_contract_disables_optimized_sdp_without_ml_dependencies(self):
        fake = _FakeTorch()
        settings = configure_torch_determinism(fake, 1729)

        self.assertEqual(fake.seed, 1729)
        self.assertEqual(fake.cuda.seed, 1729)
        self.assertTrue(fake.deterministic)
        self.assertFalse(fake.warn_only)
        self.assertFalse(fake.backends.cudnn.benchmark)
        self.assertTrue(fake.backends.cudnn.deterministic)
        self.assertFalse(fake.backends.cuda.flash)
        self.assertFalse(fake.backends.cuda.memory_efficient)
        self.assertFalse(fake.backends.cuda.cudnn)
        self.assertTrue(fake.backends.cuda.math)
        self.assertTrue(settings["strict_deterministic_algorithms"])
        self.assertFalse(settings["deterministic_algorithms_warn_only"])
        self.assertEqual(
            settings["scaled_dot_product_attention"],
            {
                "flash_enabled": False,
                "memory_efficient_enabled": False,
                "cudnn_enabled": False,
                "math_enabled": True,
                "required_policy": "math_only_when_backend_controls_are_available",
            },
        )
        self.assertEqual(
            settings["transformer_training"]["determinism_provider"],
            DETERMINISM_CONTRACT_VERSION,
        )
        self.assertFalse(
            settings["transformer_training"]["transformers_full_determinism_flag"]
        )
        self.assertEqual(settings["transformer_training"]["dataloader_num_workers"], 0)

    def test_side_label_covers_four_states(self):
        cases = [
            ([pair(0, 0)], "KEEP_PAIR"),
            ([pair(1, 0)], "CHANGE_TRIGGER"),
            ([pair(0, 1)], "CHANGE_ACTION"),
            ([pair(1, 1)], "CHANGE_BOTH"),
        ]
        for valid, expected in cases:
            with self.subTest(expected=expected):
                label, state = side_label(row(valid_pairs=valid))
                self.assertEqual(CLASS_NAMES[label], expected)
                self.assertFalse(state["individually_correct_but_invalid_pair"])

    def test_individually_correct_but_invalid_pair_maps_to_change_both_and_is_counted(self):
        rare = row(valid_pairs=[pair(0, 1), pair(1, 0)])
        label, state = side_label(rare)
        self.assertEqual(CLASS_NAMES[label], "CHANGE_BOTH")
        self.assertTrue(state["trigger_correct"])
        self.assertTrue(state["action_correct"])
        self.assertFalse(state["pair_correct"])
        self.assertTrue(state["individually_correct_but_invalid_pair"])

        training_rows = [
            row("keep", [pair(0, 0)]),
            row("trigger", [pair(1, 0)]),
            row("action", [pair(0, 1)]),
            rare | {"group_id": "rare"},
        ]
        columns, assembly = assemble_training_examples(training_rows, config())
        self.assertEqual(len(columns["label"]), 4)
        self.assertEqual(assembly["individually_correct_but_invalid_pair_count"], 1)
        self.assertEqual(assembly["class_counts"], {name: 1 for name in CLASS_NAMES})

    def test_inference_text_is_gold_independent_and_uses_both_public_views(self):
        source = row(valid_pairs=[pair(0, 0)])
        changed_gold = copy.deepcopy(source)
        changed_gold["valid_pairs"] = [pair(9, 9)]
        first = inference_text(source, 700)
        second = inference_text(changed_gold, 700)
        self.assertEqual(first, second)
        self.assertIn("public plain trigger", first[1])
        self.assertIn("public schema trigger", first[1])
        self.assertIn("public plain action", first[1])
        self.assertIn("public schema action", first[1])
        self.assertNotIn(pair(0, 0)["trigger_url"], first[0] + first[1])

    def test_family_join_requires_exact_ids_and_records_explicit_fallback(self):
        candidates = [row("g1"), row("g2")]
        split = [
            {"group_id": "g1", "semantic_family_id": "family-a"},
            {"group_id": "g2", "semantic_family_id": ""},
        ]
        joined, stats = join_semantic_families(candidates, split, expected_rows=2)
        self.assertEqual(joined[0]["_semantic_family_id"], "family-a")
        self.assertEqual(joined[1]["_semantic_family_id"], "g2")
        self.assertEqual(joined[1]["_family_source"], "group_id_fallback")
        self.assertEqual(stats["group_id_fallback_count"], 1)

        with self.assertRaisesRegex(ValueError, "group IDs differ"):
            join_semantic_families(
                candidates,
                [split[0], {"group_id": "g3", "semantic_family_id": "x"}],
                expected_rows=2,
            )

    def test_family_hash_never_splits_a_family(self):
        candidates = [row("g1"), row("g2"), row("g3"), row("g4")]
        split = [
            {"group_id": "g1", "semantic_family_id": "same-family"},
            {"group_id": "g2", "semantic_family_id": "same-family"},
            {"group_id": "g3", "semantic_family_id": "other-family"},
            {"group_id": "g4", "semantic_family_id": "third-family"},
        ]
        joined, _ = join_semantic_families(candidates, split, expected_rows=4)
        heldout_fold = family_fold("same-family")
        training, heldout, stats = partition_family_fold(joined, heldout_fold)
        heldout_ids = {item["group_id"] for item in heldout}
        self.assertTrue({"g1", "g2"} <= heldout_ids)
        self.assertFalse({"g1", "g2"} & {item["group_id"] for item in training})
        self.assertEqual(stats["family_overlap"], 0)

    def test_inverse_frequency_weights_are_deterministic(self):
        counts = {0: 8, 1: 4, 2: 2, 3: 1}
        expected = [15 / 32, 15 / 16, 15 / 8, 15 / 4]
        self.assertEqual(balanced_class_weights(counts), expected)
        self.assertEqual(balanced_class_weights(dict(reversed(list(counts.items())))), expected)

    def test_metrics_include_confusion_macro_f1_calibration_and_side_routing(self):
        rows = [
            row("keep", [pair(0, 0)]),
            row("trigger", [pair(1, 0)]),
            row("action", [pair(0, 1)]),
            row("both", [pair(1, 1)]),
        ]
        for index, item in enumerate(rows):
            item["_semantic_family_id"] = f"family-{index}"
            item["_family_source"] = "semantic_family_id"
        probabilities = [
            [0.9, 0.04, 0.03, 0.03],
            [0.1, 0.8, 0.05, 0.05],
            [0.1, 0.05, 0.8, 0.05],
            [0.1, 0.05, 0.05, 0.8],
        ]
        metrics, details = evaluate_predictions(rows, probabilities, config())
        self.assertEqual(len(details), 4)
        self.assertEqual(metrics["operational_four_class"]["accuracy"], 1.0)
        self.assertEqual(metrics["operational_four_class"]["macro_f1"], 1.0)
        self.assertEqual(
            metrics["operational_four_class"]["confusion_rows_gold_columns_predicted"],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        self.assertIn("calibration", metrics["operational_four_class"])
        self.assertEqual(metrics["routing"]["f1"], 1.0)
        self.assertEqual(metrics["side_routing"]["trigger"]["f1"], 1.0)
        self.assertEqual(metrics["side_routing"]["action"]["f1"], 1.0)

    def test_config_rejects_dev_test_paths_and_wrong_fold_binding(self):
        self.assertEqual(validate_config(config())["holdout_fold"], 0)
        bad_dev = config()
        bad_dev["train_split"] = "/raid/farm/data/dev/reranker_train.json"
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_config(bad_dev)
        bad_test = config()
        bad_test["train_candidates"] = "/raid/farm/data/test/reranker_train.json"
        with self.assertRaisesRegex(ValueError, "prohibited"):
            validate_config(bad_test)
        bad_gpu = config(2)
        bad_gpu["gpu"] = 3
        with self.assertRaisesRegex(ValueError, "same-numbered"):
            validate_config(bad_gpu)

    def test_configs_are_valid_and_cover_each_fold_once(self):
        loaded = [json.loads((ROOT / "configs" / f"router_fold{fold}.json").read_text()) for fold in range(4)]
        validated = [validate_config(item) for item in loaded]
        self.assertEqual([item["holdout_fold"] for item in validated], [0, 1, 2, 3])
        self.assertEqual([item["gpu"] for item in validated], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

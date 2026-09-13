from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RUN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RUN_ROOT.parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "v2"
SCRIPT_ROOT = RUN_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from experiment_lib import (  # noqa: E402
    DATASET_ID,
    assert_fresh_pipeline_output,
    assert_legal_transition,
    assert_new_or_resumable_output,
    assert_one_visible_gpu,
    assert_split_allowed,
    finite_number,
    project_service_group,
    read_json,
    redact_text,
    sha256_file,
    validate_dataset,
)
from ollama_baseline import parse_json_pair  # noqa: E402
from evaluate import ndcg_at_k  # noqa: E402
from status import pid_descends_from  # noqa: E402
from train_one import latest_complete_checkpoint  # noqa: E402


class DatasetAndViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = read_json(DATA_ROOT / "manifest.json")
        cls.service_root = RUN_ROOT / "derived_data" / "e0_service"

    def test_01_required_dataset_id(self):
        self.assertEqual(self.manifest["dataset_id"], DATASET_ID)

    def test_02_every_dataset_artifact_hash(self):
        validated = validate_dataset(DATA_ROOT, full_hash_check=True)
        self.assertEqual(validated["dataset_id"], DATASET_ID)

    def test_03_locked_test_prohibited(self):
        with self.assertRaises(ValueError):
            assert_split_allowed("test", "eval")
        with self.assertRaises(ValueError):
            assert_split_allowed("test", "train")

    def test_04_split_isolation(self):
        assert_split_allowed("encoder_train", "train")
        assert_split_allowed("dev", "eval")
        with self.assertRaises(ValueError):
            assert_split_allowed("reranker_train", "train")

    def test_05_service_view_manifest(self):
        manifest = read_json(self.service_root / "manifest.json")
        self.assertEqual(manifest["dataset_id"], DATASET_ID)
        self.assertEqual(manifest["counts"]["trigger_services"], 740)
        self.assertEqual(manifest["counts"]["action_services"], 584)
        self.assertEqual(manifest["counts"]["trigger_train_rows"], 8107)
        self.assertEqual(manifest["counts"]["action_train_rows"], 8264)
        for name, metadata in manifest["artifacts"].items():
            self.assertEqual(sha256_file(self.service_root / name), metadata["sha256"])

    def test_06_service_documents_are_display_name_only(self):
        corpus_functions = {
            kind: read_json(DATA_ROOT / "corpus" / f"{kind}s.json")
            for kind in ("trigger", "action")
        }
        for kind in ("trigger", "action"):
            canonical = {}
            for row in corpus_functions[kind]:
                canonical.setdefault(row["channel"], set()).add(row["channel_display"])
            for row in read_json(self.service_root / f"corpus_{kind}.json"):
                self.assertIn(row["text"], canonical[row["service_id"]])
                self.assertNotIn("\n", row["text"])

    def test_07_pair_projection_does_not_invent_cartesian_pairs(self):
        by_url = {
            "t1": {"channel": "T1"}, "t2": {"channel": "T2"},
            "a1": {"channel": "A1"}, "a2": {"channel": "A2"},
        }
        group = {"group_id": "g", "valid_pairs": [
            {"trigger_url": "t1", "action_url": "a1"},
            {"trigger_url": "t2", "action_url": "a2"},
        ]}
        projected = project_service_group(group, by_url)
        self.assertEqual(projected, [
            {"trigger_service": "T1", "action_service": "A1"},
            {"trigger_service": "T2", "action_service": "A2"},
        ])
        self.assertNotIn({"trigger_service": "T1", "action_service": "A2"}, projected)

    def test_08_real_service_dev_projection_equals_valid_pair_projection(self):
        trigger = {row["url"]: row["channel"] for row in read_json(DATA_ROOT / "corpus/triggers.json")}
        action = {row["url"]: row["channel"] for row in read_json(DATA_ROOT / "corpus/actions.json")}
        source = {row["group_id"]: row for row in read_json(DATA_ROOT / "splits/dev.json")}
        for row in read_json(self.service_root / "dev.json"):
            expected = {
                (trigger[pair["trigger_url"]], action[pair["action_url"]])
                for pair in source[row["group_id"]]["valid_pairs"]
            }
            actual = {(pair["trigger_service"], pair["action_service"]) for pair in row["valid_pairs"]}
            self.assertEqual(actual, expected)

    def test_09_plain_and_schema_render_contract(self):
        for kind in ("triggers", "actions"):
            for row in read_json(DATA_ROOT / "corpus" / f"{kind}.json"):
                self.assertTrue(row["text_schema"].startswith(row["text_plain"] + "\n") or row["text_schema"] == row["text_plain"])
                self.assertIn("channel:", row["text_plain"])
                self.assertIn("function:", row["text_plain"])

    def test_10_all_function_positives_match_index_bytes(self):
        for side in ("trigger", "action"):
            corpus = {row["url"]: row for row in read_json(DATA_ROOT / "corpus" / f"{side}s.json")}
            for view in ("plain", "schema"):
                pairs = read_json(DATA_ROOT / "pairs" / f"{side}_encoder_train_{view}.json")
                self.assertTrue(all(row["positive"] == corpus[row["label_url"]][f"text_{view}"] for row in pairs))

    def test_11_hard_negative_rows_are_complete_and_safe(self):
        hard_root = RUN_ROOT / "derived_data" / "e3_hard_negatives"
        if not hard_root.is_dir():
            if os.environ.get("FARM_REQUIRE_DERIVED") == "1":
                self.fail("E3 hard-negative artifacts are required")
            self.skipTest("hard negatives are prepared on GPU 4")
        for side in ("trigger", "action"):
            source = read_json(DATA_ROOT / "pairs" / f"{side}_encoder_train_schema.json")
            mined = read_json(hard_root / f"{side}.json")
            self.assertEqual(len(mined), len(source))
            for row in mined:
                self.assertEqual(len(row["negatives"]), 4)
                self.assertEqual(len(row["negative_urls"]), 4)
                self.assertFalse(set(row["valid_label_urls"]) & set(row["negative_urls"]))
                self.assertNotIn(row["positive"], row["negatives"])
                self.assertTrue(all(score <= row["negative_similarity_ceiling"] + 1e-7 for score in row["negative_scores"]))
        manifest = read_json(hard_root / "manifest.json")
        self.assertEqual(manifest["model_revision"], "57c266a740f537b4dc058e1b0cda161fd15afa75")
        for side in ("trigger", "action"):
            side_manifest = manifest["sides"][side]
            self.assertEqual(
                side_manifest["hard_window_negatives"] + side_manifest["easy_fallback_negatives"],
                side_manifest["rows"] * 4,
            )
            self.assertEqual(sha256_file(hard_root / f"{side}.json"), side_manifest["output_sha256"])
            self.assertEqual(side_manifest["reranker_train_window_audit"]["rows"], 1145)


class SafetyAndStateTests(unittest.TestCase):
    def test_12_explicit_single_gpu_pin(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=False):
            self.assertEqual(assert_one_visible_gpu(3), "3")

    def test_13_multiple_or_missing_gpu_rejected(self):
        for value in ("", "0,1", "abc"):
            with self.subTest(value=value), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}, clear=False):
                with self.assertRaises(RuntimeError):
                    assert_one_visible_gpu()

    def test_14_sequential_state_transitions(self):
        path = ["not_started", "preflight", "trigger_training", "action_training", "dev_evaluation", "complete"]
        for previous, following in zip(path, path[1:]):
            assert_legal_transition(previous, following)
        with self.assertRaises(RuntimeError):
            assert_legal_transition("trigger_training", "dev_evaluation")
        with self.assertRaises(RuntimeError):
            assert_legal_transition("complete", "trigger_training")

    def test_15_new_output_enforcement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(RuntimeError):
                assert_new_or_resumable_output(path, False)
            assert_new_or_resumable_output(path, True)
            assert_new_or_resumable_output(path / "new", False)

    def test_16_resume_selects_latest_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step, complete in ((100, True), (200, False), (150, True)):
                checkpoint = root / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text("{}")
                if complete:
                    (checkpoint / "optimizer.pt").touch()
                    (checkpoint / "scheduler.pt").touch()
            self.assertEqual(latest_complete_checkpoint(root).name, "checkpoint-150")

    def test_17_finite_loss_guard(self):
        self.assertTrue(finite_number(0.0))
        self.assertFalse(finite_number(float("nan")))
        self.assertFalse(finite_number(float("inf")))

    def test_18_secret_redaction(self):
        secret = "sample-super-secret"
        value = redact_text(f"Authorization: Bearer {secret} password={secret}", [secret])
        self.assertNotIn(secret, value)
        self.assertIn("[REDACTED]", value)

    def test_19_no_configured_secret_is_persisted(self):
        env_path = PROJECT_ROOT / ".env"
        if not env_path.is_file():
            self.skipTest("no local env file")
        secrets = {}
        for line in env_path.read_text(errors="ignore").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if any(token in key.upper() for token in ("PASSWORD", "TOKEN", "API_KEY")) and len(value.strip()) >= 8:
                    secrets[key.strip()] = value.strip().strip('"').strip("'")
        for path in RUN_ROOT.rglob("*"):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or "checkpoints" in path.parts
                or path.stat().st_size > 5 * 1024 * 1024
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for key, secret in secrets.items():
                if secret and secret in text:
                    self.fail(f"configured secret for {key} was persisted in {path.relative_to(RUN_ROOT)}")

    def test_19a_partial_pipeline_requires_resume_but_dry_run_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = root / "manifests/e0/launch_command.txt"
            command.parent.mkdir(parents=True)
            command.write_text("record only")
            dry_manifest = root / "dry_run/manifests/e0/pipeline.json"
            dry_manifest.parent.mkdir(parents=True)
            dry_manifest.write_text("{}")
            assert_fresh_pipeline_output(root, "e0", "full")
            pipeline = root / "manifests/e0/pipeline.json"
            pipeline.write_text("{}")
            with self.assertRaises(RuntimeError):
                assert_fresh_pipeline_output(root, "e0", "full")

    def test_19b_pid_ownership_helper(self):
        self.assertTrue(pid_descends_from(os.getpid(), os.getpid()))
        self.assertFalse(pid_descends_from(0, os.getpid()))


class EvaluationAgentAndOrchestrationTests(unittest.TestCase):
    def test_20_ndcg_is_bounded(self):
        self.assertAlmostEqual(ndcg_at_k([0], 1, 10), 1.0)
        self.assertGreater(ndcg_at_k([2], 1, 10), 0.0)
        self.assertLess(ndcg_at_k([2], 1, 10), 1.0)

    def test_21_ollama_json_parser(self):
        expected = {
            "trigger_url": "https://ifttt.com/x/triggers/y",
            "action_url": "https://ifttt.com/a/actions/b",
        }
        self.assertEqual(parse_json_pair(json.dumps(expected)), expected)
        self.assertEqual(parse_json_pair(f"```json\n{json.dumps(expected)}\n```"), expected)
        with self.assertRaises(ValueError):
            parse_json_pair(json.dumps(expected | {"extra": 1}))

    def test_22_evaluator_cli_cannot_name_locked_test(self):
        source = (SCRIPT_ROOT / "evaluate.py").read_text()
        self.assertIn('choices=("dev",)', source)
        self.assertNotIn('choices=("dev", "test")', source)

    def test_23_training_is_trigger_then_action(self):
        source = (SCRIPT_ROOT / "launch_pipeline.py").read_text()
        trigger_pos = source.index('(("trigger", "trigger_training"), ("action", "action_training"))')
        eval_pos = source.index('"dev_evaluation"', trigger_pos)
        self.assertLess(trigger_pos, eval_pos)

    def test_24_launch_sessions_and_gpus_are_one_to_one(self):
        source = (SCRIPT_ROOT / "launch_all.sh").read_text()
        for name in ("farm-e0-service", "farm-e1-plain", "farm-e2-schema", "farm-e3-enhanced"):
            self.assertIn(name, source)
        self.assertIn("for gpu in 0 1 2 3", source)
        self.assertIn("CUDA_VISIBLE_DEVICES='$index'", source)
        self.assertIn("CUDA_DEVICE_ORDER=PCI_BUS_ID", source)
        self.assertIn("check_fresh_launch.py", source)

    def test_25_resume_is_idempotent_and_non_destructive(self):
        source = (SCRIPT_ROOT / "resume_all.sh").read_text()
        self.assertIn("--resume", source)
        self.assertIn("already alive", source)
        self.assertIn("verify_pipeline.py", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("kill -", source)

    def test_26_shell_scripts_parse(self):
        for name in ("launch_all.sh", "launch_smokes.sh", "resume_all.sh", "sync_artifacts.sh"):
            result = subprocess.run(["bash", "-n", str(SCRIPT_ROOT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_27_all_arms_share_model_and_core_budget(self):
        configs = [read_json(path) for path in sorted((RUN_ROOT / "configs").glob("*.yaml"))]
        common = ("base_model", "base_model_revision", "seed", "epochs", "batch_size", "learning_rate", "warmup_ratio", "weight_decay", "max_seq_length", "scale")
        for key in common:
            self.assertEqual(len({json.dumps(config[key], sort_keys=True) for config in configs}), 1, key)

    def test_28_e1_e2_differ_only_in_view_and_identity_fields(self):
        e1 = read_json(RUN_ROOT / "configs/e1_function_plain.yaml")
        e2 = read_json(RUN_ROOT / "configs/e2_function_schema.yaml")
        ignored = {"experiment_id", "gpu", "session", "view"}
        self.assertEqual({k: v for k, v in e1.items() if k not in ignored}, {k: v for k, v in e2.items() if k not in ignored})

    def test_29_ragas_is_not_misused_as_primary_retrieval_metric(self):
        research = (RUN_ROOT / "RESEARCH.md").read_text()
        self.assertIn("Faithfulness and answer relevancy", research)
        self.assertIn("must not be relabeled as exact retrieval quality", research)

    def test_30_authored_python_compiles(self):
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", str(SCRIPT_ROOT), str(RUN_ROOT / "tests")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_31_dry_run_is_isolated_from_full_artifacts(self):
        source = (SCRIPT_ROOT / "launch_pipeline.py").read_text()
        self.assertIn('scope_root = run_root / "dry_run"', source)
        self.assertIn("assert_fresh_pipeline_output", source)

    def test_32_health_check_binds_child_pid_and_gpu_uuid(self):
        source = (SCRIPT_ROOT / "status.py").read_text()
        self.assertIn("progress_pid_on_expected_gpu", source)
        self.assertIn("owned_gpu_pids", source)
        self.assertIn("gpu_uuid_matches_manifest", source)
        self.assertNotIn('gpu.get("memory_used_mib", 0) > 100', source)

    def test_33_agent_candidates_are_hash_bound_and_final_turn_submits(self):
        source = (SCRIPT_ROOT / "ollama_baseline.py").read_text()
        self.assertIn('candidate_manifest.get("output_sha256") != candidate_hash', source)
        self.assertIn("final_agent_turn_reserved_for_submit", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

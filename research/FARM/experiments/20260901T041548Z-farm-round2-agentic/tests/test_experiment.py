from __future__ import annotations

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

from agent_compare import (  # noqa: E402
    attach_scores, bootstrap_delta, exact_mcnemar, reflect_agent, run_scored,
)
from agent_status import work_health  # noqa: E402
from evaluate import ndcg_at_k  # noqa: E402
from experiment_lib import (  # noqa: E402
    CONFIG_NAMES,
    DATASET_ID,
    EXPERIMENT_IDS,
    assert_fresh_pipeline_output,
    assert_legal_transition,
    assert_one_visible_gpu,
    assert_split_allowed,
    config_from_path,
    finite_number,
    read_json,
    redact_text,
    sha256_file,
    validate_dataset,
    validate_negative_derived,
    validate_training_rows,
)
from train_one import latest_complete_checkpoint  # noqa: E402


class DatasetAndControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_manifest = read_json(DATA_ROOT / "manifest.json")
        cls.configs = {
            config_from_path(RUN_ROOT / "configs" / name)["experiment_id"]:
            config_from_path(RUN_ROOT / "configs" / name)
            for name in CONFIG_NAMES
        }

    def test_01_dataset_id_and_hashes(self):
        self.assertEqual(self.dataset_manifest["dataset_id"], DATASET_ID)
        self.assertEqual(validate_dataset(DATA_ROOT)["dataset_id"], DATASET_ID)

    def test_02_locked_test_is_prohibited(self):
        with self.assertRaises(ValueError):
            assert_split_allowed("test", "eval")
        with self.assertRaises(ValueError):
            assert_split_allowed("test", "train")
        candidate_source = (SCRIPT_ROOT / "prepare_agent_candidates.py").read_text()
        self.assertIn('choices=("reranker_train", "dev")', candidate_source)
        self.assertNotIn('"reranker_train", "dev", "test"', candidate_source)

    def test_03_four_unique_ids_gpus_and_sessions(self):
        self.assertEqual(set(self.configs), set(EXPERIMENT_IDS))
        self.assertEqual({value["gpu"] for value in self.configs.values()}, {0, 1, 2, 3})
        self.assertEqual(len({value["session"] for value in self.configs.values()}), 4)

    def test_04_core_training_budget_is_identical(self):
        keys = (
            "level", "view", "base_model", "base_model_revision", "epochs",
            "batch_size", "learning_rate", "warmup_ratio", "weight_decay",
            "max_seq_length", "scale", "cached_loss", "mini_batch_size",
        )
        for key in keys:
            self.assertEqual(len({json.dumps(config[key], sort_keys=True) for config in self.configs.values()}), 1, key)
        self.assertTrue(all(config["level"] == "function" and config["view"] == "schema" for config in self.configs.values()))

    def test_05_seed_replications_are_exact_except_seed_identity(self):
        a = self.configs["e3_seed1337"]
        b = self.configs["e3_seed2025"]
        ignored = {"experiment_id", "gpu", "session", "seed"}
        self.assertEqual({k: v for k, v in a.items() if k not in ignored}, {k: v for k, v in b.items() if k not in ignored})
        self.assertEqual((a["seed"], b["seed"]), (1337, 2025))

    def test_06_cache_control_has_no_explicit_negatives(self):
        config = self.configs["cache0"]
        self.assertEqual(config["hard_negatives"], 0)
        self.assertEqual(config["negative_source"], "none")
        self.assertTrue(config["cached_loss"])

    def test_07_random_control_matches_negative_count(self):
        config = self.configs["random4"]
        self.assertEqual(config["hard_negatives"], 4)
        self.assertEqual(config["negative_source"], "random4")
        self.assertEqual(config["seed"], 42)

    def test_08_mined_artifact_is_hash_bound(self):
        manifest = validate_negative_derived(self.configs["e3_seed1337"], RUN_ROOT, DATA_ROOT)
        self.assertEqual(manifest["negative_source"], "mined4")
        self.assertEqual(manifest["sides"]["trigger"]["output_sha256"], "632cba0251e9ef63bc120a07e5241ac43795f789700996ba7bd8297f8808c861")
        self.assertEqual(manifest["sides"]["action"]["output_sha256"], "e46914c4343dc7a3b885f7b6b5f7c3c2d8c4b948868f445a82563dd19d7c5778")

    def test_09_random_artifact_is_complete_and_hash_bound(self):
        config = self.configs["random4"]
        manifest = validate_negative_derived(config, RUN_ROOT, DATA_ROOT)
        self.assertEqual(manifest["negative_source"], "random4")
        for side, rows in (("trigger", 8120), ("action", 8273)):
            self.assertEqual(manifest["sides"][side]["rows"], rows)
            self.assertEqual(manifest["sides"][side]["negative_count"], rows * 4)
            self.assertEqual(sha256_file(RUN_ROOT / "derived_data/random4" / f"{side}.json"), manifest["sides"][side]["output_sha256"])

    def test_10_random_rows_are_identity_safe_and_match_corpus(self):
        config = self.configs["random4"]
        for side in ("trigger", "action"):
            rows = read_json(RUN_ROOT / "derived_data/random4" / f"{side}.json")
            summary = validate_training_rows(config, DATA_ROOT, RUN_ROOT, side, rows)
            self.assertEqual(summary["negative_count"], len(rows) * 4)
            for row in rows[::997]:
                self.assertEqual(len(row["negative_urls"]), len(set(row["negative_urls"])))
                self.assertFalse(set(row["valid_label_urls"]) & set(row["negative_urls"]))
                self.assertNotIn(row["positive"], row["negatives"])

    def test_11_random_builder_never_reads_evaluation_splits(self):
        source = (SCRIPT_ROOT / "prepare_random_negatives.py").read_text()
        self.assertNotIn('splits" / "dev.json', source)
        self.assertNotIn('splits" / "test.json', source)
        self.assertIn("sha256_uniform_without_replacement", source)


class SafetyOrchestrationAndMetadataTests(unittest.TestCase):
    def test_12_one_visible_gpu_is_required(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2"}, clear=False):
            self.assertEqual(assert_one_visible_gpu(2), "2")
        for value in ("", "0,1", "abc"):
            with self.subTest(value=value), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}, clear=False):
                with self.assertRaises(RuntimeError):
                    assert_one_visible_gpu()

    def test_13_state_machine_rejects_skipped_phases(self):
        path = ["not_started", "preflight", "trigger_training", "action_training", "dev_evaluation", "complete"]
        for old, new in zip(path, path[1:]):
            assert_legal_transition(old, new)
        with self.assertRaises(RuntimeError):
            assert_legal_transition("trigger_training", "dev_evaluation")

    def test_14_fresh_launch_refuses_real_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dry_run/manifests/x").mkdir(parents=True)
            (root / "dry_run/manifests/x/pipeline.json").write_text("{}")
            assert_fresh_pipeline_output(root, "x")
            (root / "manifests/x").mkdir(parents=True)
            (root / "manifests/x/pipeline.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                assert_fresh_pipeline_output(root, "x")

    def test_15_resume_selects_latest_complete_checkpoint(self):
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

    def test_16_finite_number_guard(self):
        self.assertTrue(finite_number(0.0))
        self.assertFalse(finite_number(float("nan")))
        self.assertFalse(finite_number(float("inf")))

    def test_17_secret_redaction(self):
        secret = "sample-super-secret"
        redacted = redact_text(f"Authorization: Bearer {secret} password={secret}", [secret])
        self.assertNotIn(secret, redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_18_no_configured_secret_is_persisted(self):
        env_path = PROJECT_ROOT / ".env"
        if not env_path.is_file():
            self.skipTest("no local environment file")
        secrets = []
        for line in env_path.read_text(errors="ignore").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if any(token in key.upper() for token in ("PASSWORD", "TOKEN", "API_KEY")):
                    value = value.strip().strip('"').strip("'")
                    if len(value) >= 8:
                        secrets.append(value)
        for path in RUN_ROOT.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.stat().st_size > 5 * 1024 * 1024:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for secret in secrets:
                self.assertNotIn(secret, text, str(path.relative_to(RUN_ROOT)))

    def test_19_launch_matrix_maps_one_session_to_each_gpu(self):
        source = (SCRIPT_ROOT / "launch_all.sh").read_text()
        for session in ("farm-r2-e3-s1337", "farm-r2-e3-s2025", "farm-r2-cache0", "farm-r2-random4"):
            self.assertIn(session, source)
        self.assertIn("for gpu in 0 1 2 3", source)
        self.assertIn("CUDA_VISIBLE_DEVICES='$index'", source)
        self.assertIn("check_fresh_launch.py", source)

    def test_20_resume_is_non_destructive(self):
        source = (SCRIPT_ROOT / "resume_all.sh").read_text()
        self.assertIn("--resume", source)
        self.assertNotIn("pkill", source)
        self.assertNotIn("kill -", source)

    def test_21_negative_source_is_recorded_honestly(self):
        source = (SCRIPT_ROOT / "train_one.py").read_text()
        self.assertIn('"negative_source": config["negative_source"]', source)
        self.assertIn('"explicit_negative_count": config["hard_negatives"]', source)
        self.assertNotIn("explicit_mined_negatives", source)

    def test_22_shell_scripts_parse(self):
        for name in (
            "launch_all.sh", "launch_smokes.sh", "resume_all.sh", "launch_agents.sh",
            "resume_agents.sh", "prepare_agent_inputs.sh", "run_agent_smokes.sh", "sync_artifacts.sh",
        ):
            result = subprocess.run(["bash", "-n", str(SCRIPT_ROOT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class EvaluationAndAgentTests(unittest.TestCase):
    @staticmethod
    def agent_row():
        def candidate(side, suffix):
            return {
                "url": f"https://ifttt.com/{side}/{suffix}", "channel": side,
                "function_name": suffix, "text_plain": suffix,
                "text_schema": f"{suffix} schema", "retrieval_rank": 1,
            }
        trigger = candidate("trigger", "t1")
        action = candidate("action", "a1")
        return {
            "query": "connect them", "trigger_candidates": [trigger],
            "action_candidates": [action],
            "valid_pairs": [{"trigger_url": trigger["url"], "action_url": action["url"]}],
        }

    def test_23_ndcg_is_bounded(self):
        self.assertEqual(ndcg_at_k([0], 1, 10), 1.0)
        self.assertTrue(0 < ndcg_at_k([2], 1, 10) < 1)

    def test_24_alternative_observed_pair_is_accepted(self):
        row = {
            "valid_pairs": [
                {"trigger_url": "t1", "action_url": "a1"},
                {"trigger_url": "t2", "action_url": "a2"},
            ],
            "trigger_candidates": [{"url": "t1"}, {"url": "t2"}],
            "action_candidates": [{"url": "a1"}, {"url": "a2"}],
        }
        outcome = attach_scores({"pair": {"trigger_url": "t2", "action_url": "a2"}}, row)
        self.assertTrue(outcome["exact_valid_pair"])
        crossed = attach_scores({"pair": {"trigger_url": "t1", "action_url": "a2"}}, row)
        self.assertFalse(crossed["exact_valid_pair"])

    def test_25_pair_statistics_are_paired_and_finite(self):
        stats = exact_mcnemar([1, 0, 1, 0], [1, 1, 0, 1])
        self.assertEqual((stats["recoveries"], stats["regressions"]), (2, 1))
        interval = bootstrap_delta([1, 0, 1, 0], [1, 1, 0, 1], iterations=100)
        self.assertTrue(all(math.isfinite(value) for value in interval["ci95"]))

    def test_26_agent_arms_are_real_and_budgeted(self):
        source = (SCRIPT_ROOT / "agent_compare.py").read_text()
        for arm in ("one_shot_plain", "one_shot_schema", "reflect_agent", "role_agent", "tool_agent", "routed_tool_agent"):
            self.assertIn(arm, source)
        self.assertIn("max_turns: int = 4", source)
        self.assertIn("final_submit_turn_reserved", source)
        self.assertIn("expected-digest", source)
        self.assertIn("MAX_GENERATION_TOKENS", source)

    def test_27_router_is_calibrated_without_dev_labels(self):
        source = (SCRIPT_ROOT / "calibrate_router.py").read_text()
        self.assertIn('manifest.get("split") != "reranker_train"', source)
        self.assertIn('"supervised_labels_used": False', source)
        self.assertNotIn("valid_pairs", source)

    def test_28_candidates_use_actual_checkpoints_and_exact_urls(self):
        source = (SCRIPT_ROOT / "prepare_agent_candidates.py").read_text()
        self.assertIn("model.safetensors", source)
        self.assertIn('"identity": "exact_function_url"', source)
        self.assertIn("expected-joint-r1", source)
        self.assertIn("expected-joint-r10", source)

    def test_29_agent_jobs_checkpoint_and_resume(self):
        source = (SCRIPT_ROOT / "agent_compare.py").read_text()
        self.assertIn("os.fsync", source)
        self.assertIn("resume binding changed", source)
        self.assertIn("candidate hash does not match immutable manifest", source)

    def test_30_cloud_keys_are_named_not_embedded(self):
        source = (SCRIPT_ROOT / "launch_agents.sh").read_text()
        self.assertIn("OLLAMA_API_KEY", source)
        self.assertNotIn("Bearer ", source)
        self.assertNotRegex(source, r"(?i)(api[_-]?key|password)=[A-Za-z0-9_-]{16,}")

    def test_31_authored_python_compiles(self):
        result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", str(SCRIPT_ROOT), str(RUN_ROOT / "tests")],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_32_reflection_scores_only_final_turn(self):
        row = self.agent_row()
        valid = row["valid_pairs"][0]
        invalid = {"trigger_url": "https://ifttt.com/missing", "action_url": valid["action_url"]}

        class FakeClient:
            def __init__(self):
                self.arguments = [valid, valid, valid, invalid]

            def chat(self, **_kwargs):
                arguments = self.arguments.pop(0)
                return {"message": {"role": "assistant", "tool_calls": [{
                    "function": {"name": "submit_pair", "arguments": arguments}
                }]}}

        result = reflect_agent(FakeClient(), "fake", row)
        self.assertIsNone(result["pair"])
        self.assertFalse(result["protocol_valid"])
        self.assertTrue(result["proposals"][0]["catalog_valid"])
        self.assertFalse(result["proposals"][-1]["catalog_valid"])

    def test_33_reflection_does_not_emit_orphan_tool_result(self):
        row = self.agent_row()
        valid = row["valid_pairs"][0]

        class FakeClient:
            def __init__(self):
                self.calls = 0
                self.message_snapshots = []

            def chat(self, **kwargs):
                self.message_snapshots.append(list(kwargs["messages"]))
                self.calls += 1
                if self.calls == 1:
                    return {"message": {"role": "assistant", "content": "unsure"}}
                return {"message": {"role": "assistant", "tool_calls": [{
                    "function": {"name": "submit_pair", "arguments": valid}
                }]}}

        client = FakeClient()
        result = reflect_agent(client, "fake", row)
        self.assertTrue(result["protocol_valid"])
        second_messages = client.message_snapshots[1]
        self.assertEqual(second_messages[-2]["role"], "user")
        self.assertFalse(any(message.get("role") == "tool" for message in second_messages))

    def test_34_infrastructure_failure_is_not_scored(self):
        def fail():
            raise RuntimeError("transport failed")

        with self.assertRaises(RuntimeError):
            run_scored(fail, self.agent_row())

    def test_35_agent_health_requires_real_calls_and_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work.jsonl"
            arms = {
                arm: {"calls": 1, "usage": [{"eval_count": 1}], "protocol_valid": True}
                for arm in ("one_shot_plain", "one_shot_schema", "reflect_agent", "role_agent", "tool_agent")
            }
            work.write_text(json.dumps({"arms": arms}) + "\n")
            health = work_health(work)
            self.assertTrue(health["all_rows_have_successful_llm_calls"])
            self.assertTrue(health["minimum_protocols_exercised"])
            arms["tool_agent"] = {"calls": 0, "usage": [], "protocol_valid": False}
            work.write_text(json.dumps({"arms": arms}) + "\n")
            health = work_health(work)
            self.assertFalse(health["all_rows_have_successful_llm_calls"])
            self.assertFalse(health["minimum_protocols_exercised"])

    def test_36_agent_launch_preflights_every_target_before_tmux(self):
        for name in ("launch_agents.sh", "run_agent_smokes.sh"):
            source = (SCRIPT_ROOT / name).read_text()
            first_launch = (
                source.index("tmux new-session") if name == "launch_agents.sh"
                else source.index('"$PYTHON" "$SCRIPT_DIR/agent_compare.py"')
            )
            self.assertLess(source.index("for slug in \"${slugs[@]}\""), first_launch)
            self.assertIn("validate_agent_smokes.py", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)

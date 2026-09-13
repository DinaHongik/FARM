from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "run_configuration_experiment",
    ROOT / "scripts" / "run_configuration_experiment.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class _ForbiddenCloudClient:
    """System-boundary sentinel: rejected inputs must never construct a client."""

    constructed = False

    def __init__(self, **_: object) -> None:
        type(self).constructed = True
        raise AssertionError("Cloud client constructed for rejected input")


class ConfigurationRunnerAggregateTests(unittest.TestCase):
    def test_cloud_rejects_any_confidential_row_before_client_construction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            endpoint_records = root / "records.jsonl"
            output = root / "output"
            cases = [
                {
                    "case_id": f"public-case-{index}",
                    "benchmark": "recipegen_gold",
                    "data_classification": "public",
                    "data_source": "recipegen",
                }
                for index in range(150)
            ]
            cases[-1]["data_classification"] = "confidential"
            module.write_jsonl_atomic(candidates, cases)
            module.write_jsonl_atomic(
                endpoint_records,
                [{"case_id": row["case_id"], "prediction": None} for row in cases],
            )
            argv = [
                str(ROOT / "scripts" / "run_configuration_experiment.py"),
                "--candidates",
                str(candidates),
                "--endpoint-records",
                str(endpoint_records),
                "--output-dir",
                str(output),
                "--benchmark",
                "recipegen_gold",
                "--arm",
                "single_shot_configurator",
                "--host",
                "https://ollama.example",
                "--model",
                "test-model",
                "--model-digest",
                "a" * 64,
                "--cloud",
                "--cloud-limiter-dir",
                str(root / "leases"),
            ]
            _ForbiddenCloudClient.constructed = False
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(module, "OllamaChatClient", _ForbiddenCloudClient),
                mock.patch.dict(os.environ, {"OLLAMA_API_KEY": "test-only"}),
                self.assertRaisesRegex(RuntimeError, "confidential"),
            ):
                module.main()

            self.assertFalse(_ForbiddenCloudClient.constructed)
            self.assertFalse(output.exists())

    def test_every_candidate_row_must_match_cli_lineage(self) -> None:
        valid = [
            {
                "case_id": f"public-case-{index}",
                "benchmark": "recipegen_gold",
                "data_classification": "public",
                "data_source": "recipegen",
            }
            for index in range(150)
        ]
        mutations = {
            "benchmark": ("farm_v2_test", "candidate benchmark"),
            "data_classification": ("confidential", "candidate classification"),
            "data_source": ("farm_v2", "candidate source"),
        }

        for field, (value, message) in mutations.items():
            with self.subTest(field=field):
                cases = [dict(row) for row in valid]
                cases[-1][field] = value
                with self.assertRaisesRegex(RuntimeError, message):
                    module._validate_candidate_lineage(
                        cases,
                        benchmark="recipegen_gold",
                        classification=module.DataClassification.PUBLIC,
                        data_source=module.DataSource.RECIPEGEN,
                        cloud=False,
                    )

    def test_endpoint_manifest_lineage_mismatch_is_rejected_before_client_construction(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.jsonl"
            endpoint_directory = root / "endpoint-run"
            endpoint_records = endpoint_directory / "records.jsonl"
            output = root / "output"
            cases = [
                {
                    "case_id": f"public-case-{index}",
                    "benchmark": "recipegen_gold",
                    "data_classification": "public",
                    "data_source": "recipegen",
                }
                for index in range(150)
            ]
            module.write_jsonl_atomic(candidates, cases)
            module.write_jsonl_atomic(
                endpoint_records,
                [{"case_id": row["case_id"], "prediction": None} for row in cases],
            )
            module.write_json_atomic(
                endpoint_directory / "manifest.json",
                {
                    "schema_version": "round9-endpoint-run-manifest-v1",
                    "benchmark": "recipegen_noisy",
                    "data_classification": "public",
                    "data_source": "recipegen",
                    "candidate_artifact_sha256": module.sha256_file(candidates),
                    "ordered_case_ids_sha256": module.ordered_ids_sha256(cases),
                    "records_file": "records.jsonl",
                },
            )
            argv = [
                str(ROOT / "scripts" / "run_configuration_experiment.py"),
                "--candidates",
                str(candidates),
                "--endpoint-records",
                str(endpoint_records),
                "--output-dir",
                str(output),
                "--benchmark",
                "recipegen_gold",
                "--arm",
                "single_shot_configurator",
                "--host",
                "https://ollama.example",
                "--model",
                "test-model",
                "--model-digest",
                "a" * 64,
                "--cloud",
                "--cloud-limiter-dir",
                str(root / "leases"),
            ]
            _ForbiddenCloudClient.constructed = False
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(module, "OllamaChatClient", _ForbiddenCloudClient),
                mock.patch.dict(os.environ, {"OLLAMA_API_KEY": "test-only"}),
                self.assertRaisesRegex(RuntimeError, "endpoint manifest benchmark"),
            ):
                module.main()

            self.assertFalse(_ForbiddenCloudClient.constructed)
            self.assertFalse(output.exists())

    def test_every_endpoint_manifest_binding_field_is_validated(self) -> None:
        records_path = Path("/internal/endpoint-run/records.jsonl")
        expected = {
            "schema_version": "round9-endpoint-run-manifest-v1",
            "benchmark": "recipegen_gold",
            "data_classification": "public",
            "data_source": "recipegen",
            "candidate_artifact_sha256": "a" * 64,
            "ordered_case_ids_sha256": "b" * 64,
            "records_file": "records.jsonl",
        }
        mutations = {
            "schema_version": "unrecognized-version",
            "benchmark": "recipegen_noisy",
            "data_classification": "confidential",
            "data_source": "farm_v2",
            "candidate_artifact_sha256": "c" * 64,
            "ordered_case_ids_sha256": "d" * 64,
            "records_file": "other-records.jsonl",
        }

        for field, value in mutations.items():
            with self.subTest(field=field):
                manifest = dict(expected)
                manifest[field] = value
                with self.assertRaisesRegex(
                    RuntimeError,
                    f"endpoint manifest {field.replace('_', ' ')}",
                ):
                    module._validate_endpoint_lineage(
                        manifest,
                        records_path=records_path,
                        benchmark="recipegen_gold",
                        classification=module.DataClassification.PUBLIC,
                        data_source=module.DataSource.RECIPEGEN,
                        candidate_sha256="a" * 64,
                        ordered_case_ids_sha256="b" * 64,
                    )

    def test_manifest_is_exclusive_and_resume_identity_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            module._ensure_manifest(path, {"version": 1})
            self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
            module._ensure_manifest(path, {"version": 1})
            with self.assertRaises(RuntimeError):
                module._ensure_manifest(path, {"version": 2})

    def test_confidential_aggregate_passes_strict_egress_contract(self) -> None:
        metrics = {
            "protocol_success": True,
            "compiler_valid": True,
            "structurally_complete": False,
            "executable_ready_under_supplied_evidence": False,
            "schema_requiredness_known": True,
            "question_count": 1,
            "needs_input_decision_count": 1,
            "explicit_omit_decision_count": 0,
            "fabrication_attempt_count": 0,
            "accepted_fabrication_count": 0,
            "security_violation_attempt_count": 0,
            "accepted_security_violation_count": 0,
        }
        record = {
            "metrics": metrics,
            "protocol_failure": "compiler_validation_failed",
            "draft": {"opaque": True},
            "calls": [
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "provider_latency_ms": 3.0,
                    "queue_wait_ms": 0.0,
                    "physical_attempts": 1,
                    "cache_hit": False,
                }
            ],
        }
        result = module._aggregate(
            [record],
            benchmark="farm_v2_test",
            arm="bounded_configuration_agent",
            model="local-model",
            model_digest="a" * 64,
            input_hash="a" * 64,
        )
        self.assertEqual(result["n"], 1)
        self.assertEqual(
            result["raw_numerators"]["selected_endpoint_field_list_metadata_available"],
            1,
        )
        self.assertNotIn("schema_requiredness_known", result["raw_numerators"])
        self.assertNotIn("protocol_success", result["raw_numerators"])
        self.assertNotIn("benchmark_label", result)
        self.assertNotIn("model_metadata", result)
        self.assertNotIn("protocol_metadata", result)
        self.assertNotIn("operational_metrics", result)
        self.assertEqual(result["raw_numerators"]["structured_action_valid"], 1)
        self.assertEqual(result["raw_numerators"]["commit_emitted"], 1)
        self.assertEqual(result["raw_denominators"]["commit_emitted"], 1)
        self.assertEqual(
            result["failure_counts"]["failure_compiler_validation_failed"], 1
        )

    def test_aggregate_separates_attempted_and_accepted_unsafe_outputs(self) -> None:
        base_metrics = {
            "protocol_success": False,
            "compiler_valid": False,
            "structurally_complete": False,
            "executable_ready_under_supplied_evidence": False,
            "schema_requiredness_known": True,
            "question_count": 0,
            "needs_input_decision_count": 0,
            "explicit_omit_decision_count": 0,
        }
        records = [
            {
                "metrics": {
                    **base_metrics,
                    "fabrication_attempt_count": 2,
                    "accepted_fabrication_count": 0,
                    "security_violation_attempt_count": 1,
                    "accepted_security_violation_count": 1,
                },
                "protocol_failure": "compiler_validation_failed",
                "draft": None,
                "calls": [],
            },
            {
                "metrics": {
                    **base_metrics,
                    "fabrication_attempt_count": 1,
                    "accepted_fabrication_count": 1,
                    "security_violation_attempt_count": 0,
                    "accepted_security_violation_count": 0,
                },
                "protocol_failure": "compiler_validation_failed",
                "draft": None,
                "calls": [],
            },
        ]

        result = module._aggregate(
            records,
            benchmark="farm_v2_test",
            arm="bounded_configuration_agent",
            model="local-model",
            model_digest="a" * 64,
            input_hash="b" * 64,
        )

        self.assertEqual(result["raw_numerators"]["cases_with_fabrication_attempt"], 2)
        self.assertEqual(
            result["raw_numerators"]["cases_with_unrepaired_fabrication_violation"], 1
        )
        self.assertEqual(
            result["raw_numerators"]["cases_with_security_violation_attempt"], 1
        )
        self.assertEqual(
            result["raw_numerators"]["cases_with_unrepaired_security_violation"], 1
        )
        self.assertEqual(
            result["percentages"]["cases_with_unrepaired_fabrication_violation"], 50.0
        )
        self.assertEqual(
            result["percentages"]["cases_with_unrepaired_security_violation"], 50.0
        )
        self.assertEqual(
            result["failure_counts"]["cases_with_unrepaired_fabrication_violation"], 1
        )
        self.assertEqual(
            result["failure_counts"]["cases_with_unrepaired_security_violation"], 1
        )


if __name__ == "__main__":
    unittest.main()

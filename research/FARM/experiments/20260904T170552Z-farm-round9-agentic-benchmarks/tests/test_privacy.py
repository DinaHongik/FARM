from __future__ import annotations

import copy
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROUND9_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND9_ROOT / "src"))

from farm_r9.privacy import (
    CloudRoutingDenied,
    DataClassification,
    DataSource,
    PrivacyViolation,
    export_public_aggregate,
    redact_public_trace,
    require_cloud_route,
)


def _valid_confidential_aggregate() -> dict[str, object]:
    return {
        "n": 150,
        "raw_numerators": {
            "function_joint": 98,
            "service_joint": 109,
        },
        "raw_denominators": {
            "function_joint": 150,
            "service_joint": 150,
        },
        "percentages": {
            "function_joint": 65.333333,
            "service_joint": 72.666667,
        },
        "confidence_intervals": {
            "function_joint": {
                "low": 57.42,
                "high": 72.49,
                "level": 95.0,
            },
            "service_joint": {
                "low": 64.990,
                "high": 79.171,
                "level": 95.0,
            },
        },
        "failure_counts": {
            "protocol_failure": 3,
            "timeout": 1,
        },
        "hashes": {
            "source_artifact_sha256": "a" * 64,
        },
    }


def _valid_public_aggregate() -> dict[str, object]:
    value = copy.deepcopy(_valid_confidential_aggregate())
    value.update(
        {
            "benchmark_label": "recipegen_gold",
            "model_metadata": {
                "id": "granite4_small_h",
                "name": "granite4:small-h",
                "provider": "local_ollama",
                "version": "2026-09",
                "digest": "b" * 64,
                "role": "primary",
            },
            "protocol_metadata": {
                "id": "bounded_tool_agent",
                "version": "round9-v1",
                "semantic_calls_max": 3,
                "questions_max": 2,
                "repairs_max": 1,
            },
            "operational_metrics": {
                "semantic_calls_total": 180,
                "semantic_calls_mean_per_case": 1.2,
                "semantic_calls_p50": 1.0,
                "semantic_calls_p95": 2.0,
                "physical_attempts_total": 180,
                "cache_hits_total": 0,
                "input_tokens_total": 1000,
                "completion_tokens_total": 200,
                "tokens_total": 1200,
                "provider_latency_ms_total": 12000.0,
                "provider_latency_ms_mean_per_call": 66.66666666666667,
                "provider_latency_ms_p50": 60.0,
                "provider_latency_ms_p95": 100.0,
                "queue_wait_ms_total": 0.0,
                "questions_total": 30,
                "questions_mean_per_case": 0.2,
                "answered_questions_total": 10,
                "repair_calls_total": 5,
                "cost_available": False,
                "cost_usd_total": None,
                "cost_usd_mean_per_case": None,
            },
        }
    )
    value["hashes"] = {
        "source_artifact_sha256": "a" * 64,
        "sample_manifest_sha256": "c" * 64,
    }
    return value


class CloudRoutingPrivacyTests(unittest.TestCase):
    def test_only_explicit_public_non_farm_data_can_use_cloud(self) -> None:
        permit = require_cloud_route(
            DataClassification.PUBLIC,
            benchmark_label="recipegen_gold",
            source=DataSource.RECIPEGEN,
        )
        self.assertEqual(permit.classification, DataClassification.PUBLIC)
        self.assertEqual(permit.source, DataSource.RECIPEGEN)

        denied_routes = (
            (DataClassification.CONFIDENTIAL, DataSource.FARM_V2, "farm_v2_test"),
            (DataClassification.CONFIDENTIAL, DataSource.FARM_V2, "recipegen_gold"),
            (DataClassification.PUBLIC, DataSource.FARM_V2, "dataset_v2"),
            (DataClassification.PUBLIC, DataSource.FARM_V2, "renamed-public-benchmark"),
            (DataClassification.CONFIDENTIAL, DataSource.RECIPEGEN, "recipegen_gold"),
            ("public", DataSource.RECIPEGEN, "recipegen_gold"),
            (None, DataSource.RECIPEGEN, "recipegen_gold"),
            (DataClassification.PUBLIC, "recipegen", "recipegen_gold"),
        )
        for classification, source, benchmark_label in denied_routes:
            with self.subTest(
                classification=classification,
                source=source,
                benchmark_label=benchmark_label,
            ):
                with self.assertRaises(CloudRoutingDenied):
                    require_cloud_route(
                        classification,  # type: ignore[arg-type]
                        benchmark_label=benchmark_label,
                        source=source,  # type: ignore[arg-type]
                    )


class ConfidentialAggregateExportTests(unittest.TestCase):
    def test_export_contains_only_the_explicit_aggregate_allowlist(self) -> None:
        source = _valid_confidential_aggregate()

        exported = export_public_aggregate(
            source,
            source_classification=DataClassification.CONFIDENTIAL,
        )

        self.assertEqual(exported, source)
        self.assertIsNot(exported, source)
        self.assertEqual(
            set(exported),
            {
                "n",
                "raw_numerators",
                "raw_denominators",
                "percentages",
                "confidence_intervals",
                "failure_counts",
                "hashes",
            },
        )

    def test_forbidden_confidential_fields_are_rejected_at_any_depth(self) -> None:
        forbidden_fields = (
            "query",
            "schemas",
            "candidate_set",
            "messages",
            "prompt_text",
            "responses",
            "transcript",
            "tool-results",
            "rawCaseIds",
            "case_ids",
            "gold_query",
            "endpoint_schema",
            "top10_candidates",
            "inference_response",
            "tool_call_result",
            "ｐｒｏｍｐｔ",
        )

        for forbidden_field in forbidden_fields:
            with self.subTest(forbidden_field=forbidden_field):
                source = _valid_confidential_aggregate()
                source["failure_counts"] = {forbidden_field: 1}
                with self.assertRaisesRegex(
                    PrivacyViolation,
                    "forbidden confidential field",
                ):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.CONFIDENTIAL,
                    )

    def test_non_allowlisted_aggregate_field_is_rejected_not_silently_dropped(
        self,
    ) -> None:
        source = _valid_confidential_aggregate()
        source["provider_latency_ms"] = 1234

        with self.assertRaisesRegex(PrivacyViolation, "not public-exportable"):
            export_public_aggregate(
                source,
                source_classification=DataClassification.CONFIDENTIAL,
            )

    def test_operational_aggregates_reject_inconsistent_or_content_fields(self) -> None:
        for mutate in ("bad_mean", "bad_tokens", "raw_trace"):
            with self.subTest(mutate=mutate):
                source = copy.deepcopy(_valid_public_aggregate())
                operations = source["operational_metrics"]  # type: ignore[index]
                if mutate == "bad_mean":
                    operations["semantic_calls_mean_per_case"] = 9.0
                elif mutate == "bad_tokens":
                    operations["tokens_total"] = 999
                else:
                    operations["raw_trace"] = "must not escape"
                with self.assertRaises(PrivacyViolation):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.PUBLIC,
                    )

    def test_export_requires_an_explicit_classification_enum(self) -> None:
        with self.assertRaisesRegex(PrivacyViolation, "DataClassification"):
            export_public_aggregate(
                _valid_confidential_aggregate(),
                source_classification="confidential",  # type: ignore[arg-type]
            )

    def test_invalid_or_reversible_hashes_are_rejected(self) -> None:
        for unsafe_hash in ("case-17", "abc123", "a" * 63, "g" * 64):
            with self.subTest(unsafe_hash=unsafe_hash):
                source = copy.deepcopy(_valid_confidential_aggregate())
                source["hashes"]["source_artifact_sha256"] = unsafe_hash  # type: ignore[index]
                with self.assertRaisesRegex(PrivacyViolation, "SHA-256"):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.CONFIDENTIAL,
                    )

    def test_public_labels_and_metadata_cannot_smuggle_free_text(self) -> None:
        mutations = (
            ("benchmark_label", "a/../../private"),
            ("model_name", "send my private FARM query to the model"),
            ("model_name", "sk-abcdefghijklmnopqrstuvwxyz123456"),
        )
        for target, unsafe_value in mutations:
            with self.subTest(target=target):
                source = copy.deepcopy(_valid_public_aggregate())
                if target == "benchmark_label":
                    source["benchmark_label"] = unsafe_value
                else:
                    source["model_metadata"]["name"] = unsafe_value  # type: ignore[index]
                with self.assertRaises(PrivacyViolation):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.PUBLIC,
                    )

    def test_confidential_export_rejects_reproducibility_metadata(self) -> None:
        for field, value in (
            ("benchmark_label", "farm_v2_test"),
            ("model_metadata", {"id": "model"}),
            ("protocol_metadata", {"id": "protocol"}),
            ("operational_metrics", {"semantic_calls_total": 150}),
        ):
            with self.subTest(field=field):
                source = copy.deepcopy(_valid_confidential_aggregate())
                source[field] = value
                with self.assertRaisesRegex(PrivacyViolation, "not public-exportable"):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.CONFIDENTIAL,
                    )

    def test_confidential_metric_columns_must_align_and_recompute(self) -> None:
        mutations = (
            ("missing_denominator", "raw_denominators", "service_joint", None),
            ("wrong_percentage", "percentages", "service_joint", 99.0),
            ("impossible_numerator", "raw_numerators", "service_joint", 151),
        )
        for label, section, metric, replacement in mutations:
            with self.subTest(label=label):
                source = copy.deepcopy(_valid_confidential_aggregate())
                values = source[section]  # type: ignore[index]
                if replacement is None:
                    del values[metric]
                else:
                    values[metric] = replacement
                with self.assertRaises(PrivacyViolation):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.CONFIDENTIAL,
                    )

    def test_only_whole_artifact_hash_categories_are_exportable(self) -> None:
        for unsafe_category in ("source_sha256", "ordered_ids_sha256", "case_hash"):
            with self.subTest(unsafe_category=unsafe_category):
                source = copy.deepcopy(_valid_confidential_aggregate())
                source["hashes"] = {unsafe_category: "a" * 64}
                with self.assertRaises(PrivacyViolation):
                    export_public_aggregate(
                        source,
                        source_classification=DataClassification.CONFIDENTIAL,
                    )


class PublicUpstreamTraceRedactionTests(unittest.TestCase):
    def test_mock_credentials_are_redacted_without_mutating_the_trace(self) -> None:
        trace = {
            "events": [
                {"mock_password": "mock-password-value"},
                {"api-key": "mock-api-key-value"},
                {"refreshToken": "mock-refresh-value"},
                {"authorization": "bearer mock-auth-value"},
                {"access_tokens": ["access-token-one", "access-token-two"]},
                {"auth": "mock-short-auth"},
                {"session_cookie": "mock-session-cookie"},
                {"private_key": "mock-private-key"},
                {"credentials": "mock-credentials"},
                {"secret_ref": "not-really-an-opaque-reference"},
                {"name": "api_key", "value": "name-value-secret"},
                {"field": "password", "value": "field-value-secret"},
                {
                    "name": "Credential key (required)",
                    "content": "decorated-name-secret",
                },
            ],
            "message": (
                "Authorization: bearer inline-secret-value; token=generic-token-value; "
                'config={"password":"json-password-value"}; '
                "sk-abcdefghijklmnopqrstuvwxyz123456; "
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturevalue"
            ),
            "usage": {
                "input_tokens": 101,
                "output_tokens": 17,
                "token_count": 118,
            },
        }
        original = copy.deepcopy(trace)

        redacted = redact_public_trace(
            trace,
            source_classification=DataClassification.PUBLIC,
            source=DataSource.RECIPEGEN,
        )

        self.assertEqual(trace, original)
        for event in redacted["events"][:10]:
            self.assertIn("[REDACTED]", event.values())
        self.assertEqual(redacted["events"][10]["name"], "api_key")
        self.assertEqual(redacted["events"][10]["value"], "[REDACTED]")
        self.assertEqual(redacted["events"][11]["field"], "password")
        self.assertEqual(redacted["events"][11]["value"], "[REDACTED]")
        self.assertEqual(
            redacted["events"][12]["content"],
            "[REDACTED]",
        )
        self.assertNotIn("inline-secret-value", redacted["message"])
        self.assertEqual(redacted["usage"], trace["usage"])

        serialized = json.dumps(redacted, sort_keys=True)
        for secret in (
            "mock-password-value",
            "mock-api-key-value",
            "mock-refresh-value",
            "mock-auth-value",
            "name-value-secret",
            "field-value-secret",
            "inline-secret-value",
            "access-token-one",
            "access-token-two",
            "mock-short-auth",
            "mock-session-cookie",
            "mock-private-key",
            "mock-credentials",
            "not-really-an-opaque-reference",
            "decorated-name-secret",
            "generic-token-value",
            "json-password-value",
            "sk-abcdefghijklmnopqrstuvwxyz123456",
            "eyJhbGciOiJIUzI1NiJ9",
        ):
            self.assertNotIn(secret, serialized)

    def test_redaction_cannot_be_used_to_downgrade_a_confidential_trace(self) -> None:
        with self.assertRaisesRegex(PrivacyViolation, "public upstream"):
            redact_public_trace(
                {"query": "confidential FARM request", "api_key": "mock"},
                source_classification=DataClassification.CONFIDENTIAL,
                source=DataSource.FARM_V2,
            )

        with self.assertRaisesRegex(PrivacyViolation, "registered public"):
            redact_public_trace(
                {"api_key": "mock"},
                source_classification=DataClassification.PUBLIC,
                source=DataSource.FARM_V2,
            )


class PublicExporterCliTests(unittest.TestCase):
    def test_confidential_aggregate_is_atomically_written_mode_0600(self) -> None:
        source = _valid_confidential_aggregate()
        expected = export_public_aggregate(
            source,
            source_classification=DataClassification.CONFIDENTIAL,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "confidential-result.json"
            output_path = directory / "public-aggregate.json"
            input_path.write_text(json.dumps(source), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROUND9_ROOT / "scripts" / "export_public_aggregates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--classification",
                    "confidential",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                json.loads(output_path.read_text(encoding="utf-8")),
                expected,
            )
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o600)

    def test_export_refuses_to_overwrite_an_existing_output(self) -> None:
        source = _valid_confidential_aggregate()
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "confidential-result.json"
            output_path = directory / "public-aggregate.json"
            input_path.write_text(json.dumps(source), encoding="utf-8")
            output_path.write_text("stale", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROUND9_ROOT / "scripts" / "export_public_aggregates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--classification",
                    "confidential",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "stale")

    def test_export_failure_writes_nothing_without_echoing_private_data(self) -> None:
        source = _valid_confidential_aggregate()
        secret_key = "query_confidential-key-must-not-appear"
        source[secret_key] = "confidential-value-must-not-appear"

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "confidential-result.json"
            output_path = directory / "public-aggregate.json"
            input_path.write_text(json.dumps(source), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROUND9_ROOT / "scripts" / "export_public_aggregates.py"),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--classification",
                    "confidential",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output_path.exists())
            self.assertNotIn(secret_key, completed.stderr)
            self.assertNotIn("confidential-value-must-not-appear", completed.stderr)


class GitReleaseBoundaryTests(unittest.TestCase):
    def test_private_working_artifacts_are_ignored_and_source_is_not(self) -> None:
        repository = ROUND9_ROOT.parents[1]
        worktree_check = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        if worktree_check.returncode != 0:
            self.skipTest("requires a Git worktree to verify release boundaries")

        round9_relative = ROUND9_ROOT.relative_to(repository)
        private_paths = (
            round9_relative / "prepared" / "farm_v2_test" / "cases.jsonl",
            round9_relative / "manifests" / "samples" / "farm_v2_test.json",
            round9_relative / "DATA_PROVENANCE.json",
            round9_relative / "results" / "private-run" / "records_private.jsonl",
            round9_relative / "results" / "private-run" / "manifest.json",
            round9_relative / "results" / "bfcl" / "model_result.json",
            round9_relative / "audits" / "farm" / "case_audit.jsonl",
        )

        for path in private_paths:
            with self.subTest(path=path):
                completed = subprocess.run(
                    ["git", "check-ignore", "--quiet", "--", str(path)],
                    cwd=repository,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0)

        source = round9_relative / "src" / "farm_r9" / "privacy.py"
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(source)],
            cwd=repository,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)


if __name__ == "__main__":
    unittest.main()

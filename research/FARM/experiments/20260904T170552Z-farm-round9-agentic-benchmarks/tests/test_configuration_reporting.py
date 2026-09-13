from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.configuration_reporting import (  # noqa: E402
    BOUNDED_ARM,
    ONE_SHOT_ARM,
    ConfigurationReportInputError,
    build_configuration_internal_report,
    build_configuration_paper_tables,
)
from farm_r9.privacy import DataClassification, export_public_aggregate  # noqa: E402


PRIVATE_SENTINEL = "PRIVATE QUERY AND DRAFT MUST NEVER APPEAR"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _ids() -> list[str]:
    return [f"private-case-{index:03d}" for index in range(150)]


def _ordered_ids_sha256() -> str:
    return hashlib.sha256(("\n".join(_ids()) + "\n").encode()).hexdigest()


def _record(index: int, arm: str) -> dict:
    bounded = arm == BOUNDED_ARM
    final_limit = 130 if bounded else 120
    structural_limit = 120 if bounded else 100
    compiler_limit = 100 if bounded else 80
    resolved_limit = 70 if bounded else 50
    final_commit = index < final_limit
    structurally_complete = index < structural_limit
    compiler_valid = index < compiler_limit
    resolved = index < resolved_limit

    questions = []
    if bounded and index % 3 == 0:
        questions = [
            {
                "component": "action:destination",
                "question": PRIVATE_SENTINEL,
                "answer_available": index % 6 == 0,
            }
        ]

    validation_errors: list[dict] = []
    draft = compilation = None
    if final_commit:
        if compiler_valid and not resolved:
            source = {"kind": "needs_input", "question": PRIVATE_SENTINEL}
        else:
            source = {"kind": "omit", "reason": PRIVATE_SENTINEL}
        draft = {
            "selection": {"trigger_alias": "T01", "action_alias": "A01"},
            "trigger_fields": [],
            "action_fields": [{"field_slug": "destination", "source": source}],
            "preview": PRIVATE_SENTINEL,
            "evidence_aliases": ["T01", "A01"],
        }
        issues = []
        if not compiler_valid:
            code = (
                "ungrounded_query_literal"
                if structurally_complete
                else "missing_field_decision"
            )
            issues = [
                {
                    "code": code,
                    "side": "action",
                    "field_slug": "destination",
                    "message": PRIVATE_SENTINEL,
                    "repairable": True,
                }
            ]
            validation_errors = [{"call": 1, **issues[0]}]
        compilation = {
            "valid": compiler_valid,
            "structurally_complete": structurally_complete,
            "executable_ready": resolved,
            "issues": issues,
            "trigger_values": [],
            "action_values": [],
        }
        protocol_failure = None if compiler_valid else "compiler_validation_failed"
    else:
        protocol_failure = "json_parse_failure"
        validation_errors = [{"call": 1, "code": "json_parse_failure"}]

    needs_input_count = int(compiler_valid and not resolved)
    omit_count = int(final_commit and not needs_input_count)
    source_attempt_count = int(
        bool(validation_errors)
        and validation_errors[0]["code"] == "ungrounded_query_literal"
    )
    source_unrepaired_count = int(
        compilation is not None
        and bool(compilation["issues"])
        and compilation["issues"][0]["code"] == "ungrounded_query_literal"
    )
    return {
        "case_id": _ids()[index],
        "arm": arm,
        "private_query": PRIVATE_SENTINEL,
        "draft": draft,
        "compilation": compilation,
        "calls": [
            {
                "ordinal": 1,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "provider_latency_ms": 3.0,
                "physical_attempts": 1,
                "cache_hit": False,
            }
        ],
        "questions": questions,
        "validation_errors": validation_errors,
        "protocol_failure": protocol_failure,
        "metrics": {
            "protocol_success": compiler_valid,
            "compiler_valid": compiler_valid,
            "structurally_complete": structurally_complete,
            "executable_ready_under_supplied_evidence": resolved,
            "question_count": len(questions),
            "answered_question_count": sum(
                item["answer_available"] for item in questions
            ),
            "needs_input_decision_count": needs_input_count,
            "explicit_omit_decision_count": omit_count,
            "fabrication_attempt_count": source_attempt_count,
            "accepted_fabrication_count": source_unrepaired_count,
            "security_violation_attempt_count": 0,
            "accepted_security_violation_count": 0,
            "semantic_value_accuracy": None,
            "semantic_accuracy_supported": False,
            "schema_requiredness_known": True,
        },
    }


def _endpoint_records() -> list[dict]:
    rows = []
    for index, case_id in enumerate(_ids()):
        rows.append(
            {
                "case_id": case_id,
                "terminal": True,
                "private_query": PRIVATE_SENTINEL,
                "scores": {
                    "service_joint": index < 120,
                    "function_joint": index < 90,
                },
            }
        )
    return rows


def _manifest(arm: str, endpoint_sha256: str) -> dict:
    return {
        "schema_version": "round9-configuration-run-manifest-v3",
        "benchmark": "farm_v2_test",
        "arm": arm,
        "data_classification": "confidential",
        "data_source": "farm_v2",
        "n": 150,
        "ordered_case_ids_sha256": _ordered_ids_sha256(),
        "candidate_artifact_sha256": "a" * 64,
        "endpoint_records_sha256": endpoint_sha256,
        "prompt_sha256": ("b" if arm == ONE_SHOT_ARM else "c") * 64,
        "model_metadata": {
            "id": "local-model",
            "digest": "d" * 64,
            "provider": "ollama_local",
        },
        "protocol_metadata": {
            "version": "round9-config-v3",
            "questions_max": 0 if arm == ONE_SHOT_ARM else 2,
            "repairs_max": 0 if arm == ONE_SHOT_ARM else 1,
            "temperature": 0,
            "seed": 42,
            "thinking_mode": "disabled",
            "failure_policy": "invalid_or_ungrounded_is_incorrect_no_fallback",
        },
    }


def _artifacts(root: Path) -> dict[str, Path]:
    paths = {
        "one_records": root / "one-records.jsonl",
        "bounded_records": root / "bounded-records.jsonl",
        "endpoint_records": root / "endpoint-records.jsonl",
        "one_manifest": root / "one-manifest.json",
        "bounded_manifest": root / "bounded-manifest.json",
    }
    _write_jsonl(
        paths["one_records"], [_record(index, ONE_SHOT_ARM) for index in range(150)]
    )
    _write_jsonl(
        paths["bounded_records"], [_record(index, BOUNDED_ARM) for index in range(150)]
    )
    _write_jsonl(paths["endpoint_records"], _endpoint_records())
    endpoint_sha = _sha256(paths["endpoint_records"])
    _write_json(paths["one_manifest"], _manifest(ONE_SHOT_ARM, endpoint_sha))
    _write_json(paths["bounded_manifest"], _manifest(BOUNDED_ARM, endpoint_sha))
    return paths


def _build(paths: dict[str, Path]) -> dict:
    return build_configuration_internal_report(
        one_shot_records=paths["one_records"],
        one_shot_manifest=paths["one_manifest"],
        bounded_records=paths["bounded_records"],
        bounded_manifest=paths["bounded_manifest"],
        endpoint_records=paths["endpoint_records"],
    )


class ConfigurationReportingTests(unittest.TestCase):
    def test_paper_tables_are_strict_aggregate_only_and_revalidatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifacts(Path(directory))
            tables = build_configuration_paper_tables(
                one_shot_records=paths["one_records"],
                one_shot_manifest=paths["one_manifest"],
                bounded_records=paths["bounded_records"],
                bounded_manifest=paths["bounded_manifest"],
                endpoint_records=paths["endpoint_records"],
            )
            self.assertEqual(set(tables), {"one_shot", "bounded", "paired"})
            expected_fields = {
                "n",
                "raw_numerators",
                "raw_denominators",
                "percentages",
                "confidence_intervals",
                "hashes",
            }
            for name, table in tables.items():
                with self.subTest(name=name):
                    self.assertEqual(table["n"], 150)
                    self.assertTrue(expected_fields <= set(table))
                    self.assertNotIn("benchmark_label", table)
                    self.assertNotIn("model_metadata", table)
                    self.assertNotIn("protocol_metadata", table)
                    self.assertNotIn("operational_metrics", table)
                    self.assertEqual(
                        export_public_aggregate(
                            table,
                            source_classification=DataClassification.CONFIDENTIAL,
                        ),
                        table,
                    )
            serialized = json.dumps(tables, sort_keys=True)
            self.assertNotIn(PRIVATE_SENTINEL, serialized)
            self.assertNotIn("private-case-", serialized)

    def test_report_recomputes_narrow_aggregate_metrics_and_never_emits_private_content(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifacts(Path(directory))
            report = _build(paths)

        self.assertEqual(report["n"], 150)
        endpoint = report["endpoint_context"]
        self.assertEqual(endpoint["exact_service_pair_selected"]["raw_numerator"], 120)
        self.assertEqual(endpoint["exact_function_pair_selected"]["raw_numerator"], 90)

        compiler = report["outcome_metrics"]["compiler_accepted_draft"]
        self.assertEqual(compiler["one_shot"]["raw_numerator"], 80)
        self.assertEqual(compiler["bounded_agent"]["raw_numerator"], 100)
        self.assertEqual(
            compiler["paired_raw_counts"],
            {
                "both": 80,
                "bounded_only": 20,
                "one_shot_only": 0,
                "neither": 50,
                "denominator": 150,
            },
        )
        self.assertEqual(
            compiler["bounded_minus_one_shot_percentage_points"], 13.333333
        )

        exact_function = report["outcome_metrics"][
            "exact_function_pair_and_compiler_accepted_draft"
        ]
        self.assertEqual(exact_function["one_shot"]["raw_numerator"], 80)
        self.assertEqual(exact_function["bounded_agent"]["raw_numerator"], 90)
        resolved = report["outcome_metrics"][
            "exact_function_pair_and_compiler_accepted_no_unresolved_inputs"
        ]
        self.assertEqual(resolved["one_shot"]["raw_numerator"], 50)
        self.assertEqual(resolved["bounded_agent"]["raw_numerator"], 70)

        behavior = report["behavior_and_safety_observations"]
        self.assertEqual(
            behavior["case_asked_clarification"]["bounded_agent"]["raw_numerator"],
            50,
        )
        self.assertEqual(
            behavior["case_received_supplied_clarification"]["bounded_agent"][
                "raw_numerator"
            ],
            25,
        )
        self.assertEqual(
            behavior["case_with_compiler_detected_source_violation_attempt"][
                "one_shot"
            ]["raw_numerator"],
            20,
        )
        self.assertEqual(
            behavior["case_with_unrepaired_compiler_detected_source_violation"][
                "bounded_agent"
            ]["raw_numerator"],
            20,
        )
        self.assertNotIn("protocol_success", report["outcome_metrics"])
        self.assertNotIn(
            "executable_ready_under_supplied_evidence",
            report["outcome_metrics"],
        )

        audit = report["legacy_metric_audit"]
        self.assertEqual(
            audit["protocol_success_is_redundant_with"], "compiler_accepted_draft"
        )
        self.assertEqual(
            audit["legacy_executable_ready_is_renamed_to"],
            "compiler_accepted_no_unresolved_inputs",
        )
        self.assertIn(
            "external_platform_dry_run_success",
            report["evaluation_scope"]["not_measured"],
        )
        self.assertEqual(
            report["statistical_protocol"]["multiplicity_adjustment"],
            "none_treat_all_configuration_tests_as_exploratory",
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn(PRIVATE_SENTINEL, serialized)
        self.assertNotIn("private-case-", serialized)

    def test_report_fails_closed_on_stale_metric_or_noncomparable_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _artifacts(root)
            records = [
                json.loads(line)
                for line in paths["bounded_records"].read_text().splitlines()
            ]
            records[4]["metrics"]["protocol_success"] = False
            _write_jsonl(paths["bounded_records"], records)
            with self.assertRaisesRegex(
                ConfigurationReportInputError, "stale or inconsistent"
            ):
                _build(paths)

            paths = _artifacts(root)
            bounded_manifest = json.loads(paths["bounded_manifest"].read_text())
            bounded_manifest["model_metadata"]["digest"] = "e" * 64
            _write_json(paths["bounded_manifest"], bounded_manifest)
            with self.assertRaisesRegex(
                ConfigurationReportInputError, "same data and model binding"
            ):
                _build(paths)

    def test_report_rejects_legacy_executable_flag_that_is_not_static_resolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _artifacts(Path(directory))
            records = [
                json.loads(line)
                for line in paths["one_records"].read_text().splitlines()
            ]
            # This case has a needs_input decision; changing both stored flags
            # still cannot evade recomputation from the retained draft.
            records[60]["compilation"]["executable_ready"] = True
            records[60]["metrics"]["executable_ready_under_supplied_evidence"] = True
            _write_jsonl(paths["one_records"], records)
            with self.assertRaisesRegex(
                ConfigurationReportInputError, "static invariant"
            ):
                _build(paths)

    def test_cli_writes_fresh_private_permissioned_aggregate_only_artifact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _artifacts(root)
            output = root / "internal" / "configuration-report.json"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "export_configuration_public_report.py"),
                "--one-shot-records",
                str(paths["one_records"]),
                "--one-shot-manifest",
                str(paths["one_manifest"]),
                "--bounded-records",
                str(paths["bounded_records"]),
                "--bounded-manifest",
                str(paths["bounded_manifest"]),
                "--endpoint-records",
                str(paths["endpoint_records"]),
                "--output",
                str(output),
            ]
            first = subprocess.run(command, check=True, capture_output=True, text=True)
            original = output.read_bytes()
            second = subprocess.run(
                command, check=False, capture_output=True, text=True
            )

            self.assertEqual(json.loads(first.stdout), json.loads(original))
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(output.parent).st_mode & 0o777, 0o700)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            self.assertNotIn("Traceback", second.stderr)
            self.assertNotIn(PRIVATE_SENTINEL, original.decode())


if __name__ == "__main__":
    unittest.main()

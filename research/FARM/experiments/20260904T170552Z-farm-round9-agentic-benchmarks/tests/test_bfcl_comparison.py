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

from farm_r9.bfcl_comparison import (  # noqa: E402
    BFCLComparisonInputError,
    compare_bfcl_arms,
)


N = 150
IDS = json.loads(
    (ROOT / "manifests" / "samples" / "bfcl_v4_multi_turn_miss_param.json").read_text(
        encoding="utf-8"
    )
)["ordered_case_ids"]
SAMPLE_SHA256 = "e2848d00c5f4b9781e162a60cc66b1f0c795763de3491b26fce3abbb22ab264e"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result_rows(protocol_failure_ids: set[str]) -> list[dict]:
    return [
        {
            "id": case_id,
            "result": [[{"private_tool": {"password": "never-release"}}]],
            "inference_log": [{"content": "never release provider trace"}],
            **({"protocol_failure": True} if case_id in protocol_failure_ids else {}),
        }
        for case_id in IDS
    ]


def _score_rows(correct_ids: set[str]) -> list[dict]:
    failures = [case_id for case_id in IDS if case_id not in correct_ids]
    return [
        {
            "accuracy": len(correct_ids) / N,
            "correct_count": len(correct_ids),
            "total_count": N,
        },
        *[
            {
                "id": case_id,
                "valid": False,
                "error": {"error_message": ["never release evaluator text"]},
                "model_result_raw": [{"private_tool_arg": "never-release"}],
            }
            for case_id in failures
        ],
    ]


def _write_binding(
    path: Path,
    *,
    arm: str,
    model_sha256: str = "d" * 64,
    sample_sha256: str = SAMPLE_SHA256,
    max_physical_calls_per_turn: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_physical_calls_per_turn is None:
        max_physical_calls_per_turn = 1 if arm == "single_step_per_turn_fc" else 4
    path.write_text(
        json.dumps(
            {
                "schema_version": "farm-round9-bfcl-run-binding-v2",
                "benchmark": "bfcl_v4_multi_turn_miss_param",
                "evaluation_scope": "partial:150_of_200",
                "registry_name": f"test-{arm.replace('_', '-')}",
                "arm": arm,
                "model_sha256": model_sha256,
                "sample_sha256": sample_sha256,
                "official_commit": "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000",
                "temperature": 0.0,
                "seed": 9052026,
                "thinking_mode": "low",
                "max_physical_calls_per_turn": max_physical_calls_per_turn,
                "worker_policy": "thread-pool-atomic-case-resume-v1",
                "workers": 2,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class BFCLComparisonTests(unittest.TestCase):
    def test_official_failure_rows_drive_order_independent_paired_comparison(
        self,
    ) -> None:
        both = set(IDS[:100])
        native_only = set(IDS[100:120])
        single_only = set(IDS[120:130])
        single_correct = both | single_only
        native_correct = both | native_only
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_tree = root / "raw" / "single-registry"
            native_tree = root / "raw" / "native-registry"
            single_result = single_tree / "multi_turn" / "result.jsonl"
            native_result = native_tree / "multi_turn" / "result.jsonl"
            single_score = root / "scores-baseline" / "single-score.jsonl"
            native_score = root / "scores-agent-retry-v2" / "native-score.jsonl"
            single_binding = single_tree / "RUN_BINDING.json"
            native_binding = native_tree / "RUN_BINDING.json"
            _write_jsonl(single_result, _result_rows(set(IDS[100:105])))
            _write_jsonl(single_score, _score_rows(single_correct))
            _write_jsonl(
                native_result,
                list(reversed(_result_rows(set(IDS[120:123])))),
            )
            native_score_rows = _score_rows(native_correct)
            _write_jsonl(
                native_score, [native_score_rows[0], *reversed(native_score_rows[1:])]
            )
            _write_binding(single_binding, arm="single_step_per_turn_fc")
            _write_binding(native_binding, arm="native_tool_agent")
            expected_hashes = {
                "single_step_result_sha256": hashlib.sha256(
                    single_result.read_bytes()
                ).hexdigest(),
                "single_step_score_sha256": hashlib.sha256(
                    single_score.read_bytes()
                ).hexdigest(),
                "native_agent_result_sha256": hashlib.sha256(
                    native_result.read_bytes()
                ).hexdigest(),
                "native_agent_score_sha256": hashlib.sha256(
                    native_score.read_bytes()
                ).hexdigest(),
                "single_step_binding_sha256": hashlib.sha256(
                    single_binding.read_bytes()
                ).hexdigest(),
                "native_agent_binding_sha256": hashlib.sha256(
                    native_binding.read_bytes()
                ).hexdigest(),
                "frozen_case_id_set_sha256": "557d2da1af0c7a54b9cd0d506037ae26452150e87e4a2d210d6e8c302f4f0269",
            }

            result = compare_bfcl_arms(
                single_step_result=single_result,
                single_step_score=single_score,
                native_agent_result=native_result,
                native_agent_score=native_score,
                single_step_binding=single_binding,
                native_agent_binding=native_binding,
            )

        self.assertEqual(result["n"], 150)
        self.assertEqual(result["arms"]["single_step_per_turn_fc"]["correct"], 110)
        self.assertEqual(
            result["arms"]["single_step_per_turn_fc"]["accuracy_percent"], 73.333333
        )
        self.assertEqual(
            result["arms"]["single_step_per_turn_fc"]["protocol_failures"], 5
        )
        self.assertEqual(result["arms"]["native_tool_agent"]["correct"], 120)
        self.assertEqual(result["arms"]["native_tool_agent"]["accuracy_percent"], 80.0)
        self.assertEqual(result["arms"]["native_tool_agent"]["protocol_failures"], 3)
        self.assertEqual(
            result["paired_outcomes"]["raw_counts"],
            {
                "both_correct": 100,
                "native_tool_agent_only_rescue": 20,
                "single_step_only_regression": 10,
                "neither": 20,
            },
        )
        self.assertEqual(result["delta_percentage_points"], 6.666667)
        self.assertEqual(result["format_version"], "round9-bfcl-paired-comparison-v3")
        self.assertEqual(result["arms"]["single_step_per_turn_fc"]["denominator"], 150)
        self.assertEqual(
            result["arms"]["native_tool_agent"]["protocol_failure_denominator"], 150
        )
        self.assertEqual(
            result["delta_confidence_interval_95"],
            {
                "method": "paired_case_bootstrap_percentile",
                "confidence_level": 0.95,
                "resamples": 10000,
                "seed": 9052026,
                "rng": "python_random_mt19937_v2",
                "quantile_method": "hyndman_fan_type_7",
                "experimental_unit": "paired_case",
                "stratified": False,
                "stratum_counts": None,
                "observed_delta_percentage_points": 6.666667,
                "low_percentage_points": -0.666667,
                "high_percentage_points": 14.0,
            },
        )
        self.assertAlmostEqual(
            result["mcnemar_exact_two_sided_p"],
            0.09873714670538902,
        )
        self.assertEqual(
            result["multiplicity_policy"]["confirmatory_metric"],
            "official_accuracy",
        )
        self.assertEqual(
            result["arms"]["single_step_per_turn_fc"]["wilson_95_percent"]["low"],
            65.73855,
        )
        self.assertEqual(
            result["arms"]["native_tool_agent"]["wilson_95_percent"]["high"],
            85.615918,
        )
        self.assertEqual(result["hashes"], expected_hashes)
        self.assertEqual(
            result["paired_protocol_binding"]["same_model_sha256"], "d" * 64
        )
        self.assertEqual(
            result["paired_protocol_binding"]["effective_calls_per_turn"],
            {"single_step_per_turn_fc": 1, "native_tool_agent": 4},
        )
        self.assertEqual(
            result["paired_protocol_binding"]["configured_max_physical_calls_per_turn"],
            {"single_step_per_turn_fc": 1, "native_tool_agent": 4},
        )
        encoded = json.dumps(result, sort_keys=True)
        self.assertNotIn("multi_turn_miss_param_000", encoded)
        self.assertNotIn("never release", encoded)
        self.assertNotIn("private_tool", encoded)

    def test_comparison_rejects_incomplete_or_inconsistent_official_artifacts(
        self,
    ) -> None:
        single_correct = set(IDS[:110])
        native_correct = set(IDS[:120])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_result = root / "single-result.jsonl"
            single_score = root / "single-score.jsonl"
            native_result = root / "native-result.jsonl"
            native_score = root / "native-score.jsonl"
            single_binding = root / "single-binding.json"
            native_binding = root / "native-binding.json"
            _write_jsonl(single_result, _result_rows(set(IDS[110:115])))
            _write_jsonl(single_score, _score_rows(single_correct))
            _write_jsonl(native_result, _result_rows(set(IDS[120:123])))
            _write_jsonl(native_score, _score_rows(native_correct))
            _write_binding(single_binding, arm="single_step_per_turn_fc")
            _write_binding(native_binding, arm="native_tool_agent")

            def compare(
                *,
                result: Path = single_result,
                score: Path = single_score,
                other_result: Path = native_result,
                other_score: Path = native_score,
            ) -> dict:
                return compare_bfcl_arms(
                    single_step_result=result,
                    single_step_score=score,
                    native_agent_result=other_result,
                    native_agent_score=other_score,
                    single_step_binding=single_binding,
                    native_agent_binding=native_binding,
                )

            with self.subTest("missing artifact"):
                with self.assertRaisesRegex(BFCLComparisonInputError, "missing"):
                    compare(result=root / "absent.jsonl")

            with self.subTest("result is not 150"):
                partial_result = root / "partial-result.jsonl"
                _write_jsonl(partial_result, _result_rows(set())[:-1])
                with self.assertRaisesRegex(BFCLComparisonInputError, "exactly 150"):
                    compare(result=partial_result)

            with self.subTest("duplicate result ID"):
                duplicate_rows = _result_rows(set())
                duplicate_rows[-1]["id"] = duplicate_rows[0]["id"]
                duplicate_result = root / "duplicate-result.jsonl"
                _write_jsonl(duplicate_result, duplicate_rows)
                with self.assertRaisesRegex(BFCLComparisonInputError, "duplicate ID"):
                    compare(result=duplicate_result)

            with self.subTest("paired ID sets differ"):
                mismatched_rows = _result_rows(set(IDS[120:123]))
                mismatched_rows[-1]["id"] = "multi_turn_miss_param_not_shared"
                mismatched_result = root / "mismatched-result.jsonl"
                mismatched_score_rows = _score_rows(native_correct)
                for row in mismatched_score_rows[1:]:
                    if row["id"] == IDS[-1]:
                        row["id"] = "multi_turn_miss_param_not_shared"
                mismatched_score = root / "mismatched-score.jsonl"
                _write_jsonl(mismatched_result, mismatched_rows)
                _write_jsonl(mismatched_score, mismatched_score_rows)
                with self.assertRaisesRegex(
                    BFCLComparisonInputError, "frozen 150-case sample"
                ) as raised:
                    compare(
                        other_result=mismatched_result, other_score=mismatched_score
                    )
                self.assertNotIn("not_shared", str(raised.exception))

            with self.subTest("header n is not official sample size"):
                bad_header = _score_rows(single_correct)
                bad_header[0]["total_count"] = 149
                bad_score = root / "bad-n-score.jsonl"
                _write_jsonl(bad_score, bad_header)
                with self.assertRaisesRegex(BFCLComparisonInputError, "n=150"):
                    compare(score=bad_score)

            with self.subTest("header accuracy disagrees"):
                bad_accuracy = _score_rows(single_correct)
                bad_accuracy[0]["accuracy"] = 0.99
                bad_score = root / "bad-accuracy-score.jsonl"
                _write_jsonl(bad_score, bad_accuracy)
                with self.assertRaisesRegex(BFCLComparisonInputError, "accuracy"):
                    compare(score=bad_score)

            with self.subTest("official failure row count disagrees"):
                missing_failure = _score_rows(single_correct)[:-1]
                bad_score = root / "missing-failure-score.jsonl"
                _write_jsonl(bad_score, missing_failure)
                with self.assertRaisesRegex(BFCLComparisonInputError, "failure count"):
                    compare(score=bad_score)

            with self.subTest("duplicate official failure ID"):
                duplicate_failure = _score_rows(single_correct)
                duplicate_failure[-1]["id"] = duplicate_failure[-2]["id"]
                bad_score = root / "duplicate-failure-score.jsonl"
                _write_jsonl(bad_score, duplicate_failure)
                with self.assertRaisesRegex(
                    BFCLComparisonInputError, "duplicate failure ID"
                ):
                    compare(score=bad_score)

            with self.subTest("failure row must be invalid"):
                valid_failure = _score_rows(single_correct)
                valid_failure[-1]["valid"] = True
                bad_score = root / "valid-failure-score.jsonl"
                _write_jsonl(bad_score, valid_failure)
                with self.assertRaisesRegex(
                    BFCLComparisonInputError, "not marked invalid"
                ):
                    compare(score=bad_score)

            with self.subTest("protocol failure cannot score correct"):
                inconsistent_result = root / "protocol-scored-correct.jsonl"
                _write_jsonl(inconsistent_result, _result_rows({IDS[0]}))
                with self.assertRaisesRegex(BFCLComparisonInputError, "scored correct"):
                    compare(result=inconsistent_result)

            with self.subTest("protocol failure flag is strict boolean"):
                malformed_rows = _result_rows(set())
                malformed_rows[0]["protocol_failure"] = 1
                malformed_result = root / "non-boolean-protocol.jsonl"
                _write_jsonl(malformed_result, malformed_rows)
                with self.assertRaisesRegex(BFCLComparisonInputError, "not boolean"):
                    compare(result=malformed_result)

            with self.subTest("six input artifacts must be distinct"):
                with self.assertRaisesRegex(BFCLComparisonInputError, "six distinct"):
                    compare(other_result=single_result)

            with self.subTest("paired bindings require the same model"):
                _write_binding(
                    native_binding, arm="native_tool_agent", model_sha256="e" * 64
                )
                with self.assertRaisesRegex(
                    BFCLComparisonInputError, "differ in model"
                ):
                    compare()
                _write_binding(native_binding, arm="native_tool_agent")

            with self.subTest("binding requires the frozen sample"):
                _write_binding(
                    native_binding, arm="native_tool_agent", sample_sha256="e" * 64
                )
                with self.assertRaisesRegex(
                    BFCLComparisonInputError, "frozen protocol"
                ):
                    compare()
                _write_binding(native_binding, arm="native_tool_agent")

    def test_cli_writes_aggregate_exclusively_with_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_tree = root / "raw" / "single-registry"
            native_tree = root / "raw" / "native-registry"
            single_result = single_tree / "multi_turn" / "result.jsonl"
            native_result = native_tree / "multi_turn" / "result.jsonl"
            single_score = root / "scores-baseline" / "single-score.jsonl"
            native_score = root / "scores-agent-retry-v2" / "native-score.jsonl"
            single_binding = single_tree / "RUN_BINDING.json"
            native_binding = native_tree / "RUN_BINDING.json"
            output = root / "paired-public.json"
            _write_jsonl(single_result, _result_rows(set(IDS[110:115])))
            _write_jsonl(single_score, _score_rows(set(IDS[:110])))
            _write_jsonl(native_result, _result_rows(set(IDS[120:123])))
            _write_jsonl(native_score, _score_rows(set(IDS[:120])))
            _write_binding(single_binding, arm="single_step_per_turn_fc")
            _write_binding(native_binding, arm="native_tool_agent")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "compare_bfcl_arms.py"),
                "--single-step-result",
                str(single_result),
                "--single-step-score",
                str(single_score),
                "--native-agent-result",
                str(native_result),
                "--native-agent-score",
                str(native_score),
                "--output",
                str(output),
            ]

            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            artifact = json.loads(output.read_text(encoding="utf-8"))
            original = output.read_bytes()
            repeated = subprocess.run(
                command, check=False, capture_output=True, text=True
            )

            self.assertEqual(json.loads(completed.stdout), artifact)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertNotIn("Traceback", repeated.stderr)
            self.assertEqual(output.read_bytes(), original)
            encoded = json.dumps(artifact, sort_keys=True)
            self.assertNotIn("multi_turn_miss_param_000", encoded)
            self.assertNotIn("never-release", encoded)

    def test_official_horizon_comparison_requires_explicit_21_call_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            single_result = root / "single" / "multi_turn" / "result.jsonl"
            native_result = root / "native" / "multi_turn" / "result.jsonl"
            single_score = root / "single-score.jsonl"
            native_score = root / "native-score.jsonl"
            single_binding = single_result.parent.parent / "RUN_BINDING.json"
            native_binding = native_result.parent.parent / "RUN_BINDING.json"
            _write_jsonl(single_result, _result_rows(set()))
            _write_jsonl(native_result, _result_rows(set()))
            _write_jsonl(single_score, _score_rows(set(IDS[:5])))
            _write_jsonl(native_score, _score_rows(set(IDS[:38])))
            _write_binding(single_binding, arm="single_step_per_turn_fc")
            _write_binding(
                native_binding,
                arm="native_tool_agent",
                max_physical_calls_per_turn=21,
            )
            arguments = {
                "single_step_result": single_result,
                "single_step_score": single_score,
                "native_agent_result": native_result,
                "native_agent_score": native_score,
                "single_step_binding": single_binding,
                "native_agent_binding": native_binding,
            }

            with self.assertRaisesRegex(
                BFCLComparisonInputError,
                "declared call budget",
            ):
                compare_bfcl_arms(**arguments)

            result = compare_bfcl_arms(
                **arguments,
                native_agent_call_budget=21,
            )

        self.assertEqual(
            result["paired_protocol_binding"]["configured_max_physical_calls_per_turn"][
                "native_tool_agent"
            ],
            21,
        )
        self.assertEqual(
            result["paired_protocol_binding"]["effective_calls_per_turn"][
                "native_tool_agent"
            ],
            21,
        )


if __name__ == "__main__":
    unittest.main()

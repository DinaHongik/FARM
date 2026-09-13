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

from farm_r9.configuration_comparison import (
    ComparisonInputError,
    compare_configuration_ledgers,
)


ONE_SHOT = "single_shot_configurator"
BOUNDED = "bounded_configuration_agent"


def _outcomes(index: int, metric: str) -> tuple[bool, bool]:
    if metric == "compiler_valid":
        boundaries = (100, 120, 130)
    elif metric == "structurally_complete":
        boundaries = (70, 85, 90)
    else:
        boundaries = (0, 9, 9)
    both_end, bounded_only_end, one_shot_only_end = boundaries
    if index < both_end:
        return True, True
    if index < bounded_only_end:
        return False, True
    if index < one_shot_only_end:
        return True, False
    return False, False


def _records(arm: str) -> list[dict]:
    use_bounded = arm == BOUNDED
    rows = []
    for index in range(150):
        metrics = {}
        for metric in (
            "compiler_valid",
            "structurally_complete",
            "executable_ready_under_supplied_evidence",
        ):
            one_shot_value, bounded_value = _outcomes(index, metric)
            metrics[metric] = bounded_value if use_bounded else one_shot_value
        rows.append({
            "case_id": f"private-case-{index:03d}",
            "arm": arm,
            "metrics": metrics,
            "private_query": f"never release this query {index}",
            "draft": {"private": f"never release this draft {index}"},
        })
    return rows


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class ConfigurationComparisonTests(unittest.TestCase):
    def test_order_independent_pairing_emits_only_aggregate_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one_shot_path = root / "one-shot.jsonl"
            bounded_path = root / "bounded.jsonl"
            _write_jsonl(one_shot_path, _records(ONE_SHOT))
            _write_jsonl(bounded_path, list(reversed(_records(BOUNDED))))

            result = compare_configuration_ledgers(one_shot_path, bounded_path)
            expected_one_shot_hash = hashlib.sha256(one_shot_path.read_bytes()).hexdigest()
            expected_bounded_hash = hashlib.sha256(bounded_path.read_bytes()).hexdigest()

        self.assertEqual(result["n"], 150)
        compiler = result["metrics"]["compiler_valid"]
        self.assertEqual(compiler["raw_counts"], {
            "one_shot_correct": 110,
            "bounded_correct": 120,
            "both_correct": 100,
            "bounded_only_rescue": 20,
            "one_shot_only_regression": 10,
            "neither": 20,
        })
        self.assertEqual(compiler["percentages"]["one_shot"], 73.333333)
        self.assertEqual(compiler["percentages"]["bounded"], 80.0)
        self.assertEqual(compiler["delta_percentage_points"], 6.666667)
        self.assertAlmostEqual(
            compiler["mcnemar_exact_two_sided_p"],
            0.09873714670538902,
        )
        self.assertEqual(
            compiler["confidence_intervals"]["one_shot"]["low"],
            65.73855,
        )
        self.assertEqual(
            compiler["confidence_intervals"]["bounded"]["high"],
            85.615918,
        )
        self.assertEqual(
            result["metrics"]["executable_ready_under_supplied_evidence"]
            ["raw_counts"]["bounded_only_rescue"],
            9,
        )
        self.assertEqual(
            result["hashes"]["one_shot_records_sha256"],
            expected_one_shot_hash,
        )
        self.assertEqual(
            result["hashes"]["bounded_records_sha256"],
            expected_bounded_hash,
        )
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn("private-case", serialized)
        self.assertNotIn("never release", serialized)

    def test_comparison_fails_closed_for_incomplete_or_unpairable_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_one_shot = root / "valid-one-shot.jsonl"
            valid_bounded = root / "valid-bounded.jsonl"
            _write_jsonl(valid_one_shot, _records(ONE_SHOT))
            _write_jsonl(valid_bounded, _records(BOUNDED))

            with self.subTest("missing file"):
                with self.assertRaises(ComparisonInputError):
                    compare_configuration_ledgers(root / "missing.jsonl", valid_bounded)

            with self.subTest("not 150"):
                partial = root / "partial.jsonl"
                _write_jsonl(partial, _records(ONE_SHOT)[:-1])
                with self.assertRaisesRegex(ComparisonInputError, "exactly 150"):
                    compare_configuration_ledgers(partial, valid_bounded)

            with self.subTest("duplicate ID"):
                duplicate_rows = _records(ONE_SHOT)
                duplicate_rows[-1]["case_id"] = duplicate_rows[0]["case_id"]
                duplicate = root / "duplicate.jsonl"
                _write_jsonl(duplicate, duplicate_rows)
                with self.assertRaisesRegex(ComparisonInputError, "duplicate"):
                    compare_configuration_ledgers(duplicate, valid_bounded)

            with self.subTest("mismatched ID sets"):
                mismatched_rows = _records(BOUNDED)
                mismatched_rows[-1]["case_id"] = "private-case-not-shared"
                mismatched = root / "mismatched.jsonl"
                _write_jsonl(mismatched, mismatched_rows)
                with self.assertRaisesRegex(ComparisonInputError, "different case-ID sets") as raised:
                    compare_configuration_ledgers(valid_one_shot, mismatched)
                self.assertNotIn("private-case-not-shared", str(raised.exception))

            with self.subTest("wrong arm"):
                wrong_arm_rows = _records(BOUNDED)
                wrong_arm_rows[20]["arm"] = ONE_SHOT
                wrong_arm = root / "wrong-arm.jsonl"
                _write_jsonl(wrong_arm, wrong_arm_rows)
                with self.assertRaisesRegex(ComparisonInputError, "wrong or mixed arm"):
                    compare_configuration_ledgers(valid_one_shot, wrong_arm)

            with self.subTest("missing outcome"):
                missing_metric_rows = _records(BOUNDED)
                del missing_metric_rows[40]["metrics"]["compiler_valid"]
                missing_metric = root / "missing-metric.jsonl"
                _write_jsonl(missing_metric, missing_metric_rows)
                with self.assertRaisesRegex(ComparisonInputError, "non-boolean outcome"):
                    compare_configuration_ledgers(valid_one_shot, missing_metric)

            with self.subTest("integer is not boolean"):
                non_boolean_rows = _records(BOUNDED)
                non_boolean_rows[60]["metrics"]["structurally_complete"] = 1
                non_boolean = root / "non-boolean.jsonl"
                _write_jsonl(non_boolean, non_boolean_rows)
                with self.assertRaisesRegex(ComparisonInputError, "non-boolean outcome"):
                    compare_configuration_ledgers(valid_one_shot, non_boolean)

    def test_standalone_cli_writes_exclusively_with_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one_shot_path = root / "one-shot.jsonl"
            bounded_path = root / "bounded.jsonl"
            output_path = root / "paired-public.json"
            _write_jsonl(one_shot_path, _records(ONE_SHOT))
            _write_jsonl(bounded_path, _records(BOUNDED))
            command = [
                sys.executable,
                str(ROOT / "scripts" / "compare_configuration_arms.py"),
                "--one-shot-records",
                str(one_shot_path),
                "--bounded-records",
                str(bounded_path),
                "--output",
                str(output_path),
            ]

            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            artifact = json.loads(output_path.read_text(encoding="utf-8"))
            original_bytes = output_path.read_bytes()
            repeated = subprocess.run(command, check=False, capture_output=True, text=True)

            self.assertEqual(json.loads(completed.stdout), artifact)
            self.assertEqual(os.stat(output_path).st_mode & 0o777, 0o600)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertNotIn("Traceback", repeated.stderr)
            self.assertEqual(output_path.read_bytes(), original_bytes)
            self.assertNotIn("private-case", json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    unittest.main()

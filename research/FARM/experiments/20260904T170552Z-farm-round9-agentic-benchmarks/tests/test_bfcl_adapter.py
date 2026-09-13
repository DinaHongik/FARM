from __future__ import annotations

import hashlib
import json
import copy
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.adapters.bfcl import prepare_bfcl_missing_parameter, redact_for_log


class BfclAdapterTests(unittest.TestCase):
    def test_checksum_is_verified_before_json_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.json"
            answer_path = root / "answers.json"
            test_path.write_text("not json\n", encoding="utf-8")
            answer_path.write_text("not json either\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                prepare_bfcl_missing_parameter(test_path, answer_path)

    def test_official_rows_are_aligned_and_sampled_by_missing_turn_profile(self) -> None:
        profile_counts = {
            (0,): 81,
            (1,): 51,
            (2,): 33,
            (3,): 24,
            (4,): 6,
            (5,): 3,
            (1, 5): 1,
            (1, 4, 5): 1,
        }
        tests = []
        answers = []
        index = 0
        for profile, count in profile_counts.items():
            for _ in range(count):
                turn_count = max(profile) + 2
                case_id = f"multi_turn_miss_param_{index}"
                tests.append({
                    "id": case_id,
                    "question": [[{"role": "user", "content": f"turn {turn}"}]
                                 for turn in range(turn_count)],
                    "initial_config": {"MockAPI": {"password": f"mock-{index}"}},
                    "path": ["MockAPI.noop"],
                    "involved_classes": ["MockAPI" if index % 2 else "OtherAPI"],
                })
                answers.append({
                    "id": case_id,
                    "ground_truth": [
                        [] if turn in profile else [f"noop(value='{index}-{turn}')"]
                        for turn in range(turn_count)
                    ],
                })
                index += 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test.json"
            answer_path = root / "answers.json"
            test_payload = "".join(json.dumps(row) + "\n" for row in tests)
            answer_payload = "".join(json.dumps(row) + "\n" for row in answers)
            test_path.write_text(test_payload, encoding="utf-8")
            answer_path.write_text(answer_payload, encoding="utf-8")

            selected_tests, selected_answers, manifest = prepare_bfcl_missing_parameter(
                test_path,
                answer_path,
                expected_test_sha256=hashlib.sha256(test_payload.encode()).hexdigest(),
                expected_possible_answer_sha256=hashlib.sha256(answer_payload.encode()).hexdigest(),
            )

        self.assertEqual(len(selected_tests), 150)
        self.assertEqual(
            [row["id"] for row in selected_tests],
            [row["id"] for row in selected_answers],
        )
        self.assertEqual(
            manifest["sample_strata"],
            {
                "missing_turn_indices=0": 61,
                "missing_turn_indices=1": 38,
                "missing_turn_indices=1+4+5": 1,
                "missing_turn_indices=1+5": 1,
                "missing_turn_indices=2": 25,
                "missing_turn_indices=3": 18,
                "missing_turn_indices=4": 4,
                "missing_turn_indices=5": 2,
            },
        )
        self.assertEqual(manifest["evaluation_scope"], "partial:150_of_200")
        self.assertEqual(sum(manifest["population_turn_counts"].values()), 200)
        self.assertEqual(sum(manifest["sample_involved_class_profiles"].values()), 150)
        self.assertTrue(all(set(row) >= {"id", "question", "initial_config", "path", "involved_classes"}
                            for row in selected_tests))
        self.assertEqual(
            Counter(row["id"] for row in selected_tests),
            Counter(manifest["ordered_case_ids"]),
        )
        # The evaluator input remains upstream-compatible; only log views are redacted.
        self.assertTrue(any(row["initial_config"]["MockAPI"]["password"].startswith("mock-")
                            for row in selected_tests))

    def test_log_view_redacts_mock_credentials_without_mutating_evaluator_source(self) -> None:
        source = {
            "id": "multi_turn_miss_param_4",
            "initial_config": {
                "TwitterAPI": {
                    "username": "tech_guru",
                    "password": "securePass123",
                    "access_token": "mock-access-token",
                }
            },
        }
        before = copy.deepcopy(source)

        safe = redact_for_log(source)

        self.assertEqual(source, before)
        self.assertEqual(safe["initial_config"]["TwitterAPI"]["username"], "tech_guru")
        self.assertEqual(safe["initial_config"]["TwitterAPI"]["password"], "[REDACTED]")
        self.assertEqual(safe["initial_config"]["TwitterAPI"]["access_token"], "[REDACTED]")
        self.assertNotIn("securePass123", json.dumps(safe))


if __name__ == "__main__":
    unittest.main()

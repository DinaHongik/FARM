from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


ROUND9_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_yao_successor",
    ROUND9_ROOT / "scripts" / "launch_yao_successor.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReferenceCompletionTests(unittest.TestCase):
    def test_waits_until_both_reference_artifacts_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = root / "records.jsonl"
            aggregate = root / "aggregate.json"
            ready, reason = MODULE._load_complete_reference(
                records, aggregate, expected_n=150
            )
            self.assertFalse(ready)
            self.assertEqual(reason, "reference_artifacts_pending")

            records.write_text(
                "".join(
                    json.dumps({"case_id": f"case-{index}"}) + "\n"
                    for index in range(149)
                ),
                encoding="utf-8",
            )
            aggregate.write_text(
                json.dumps({"n": 150, "arm": "same_model_one_shot"}),
                encoding="utf-8",
            )
            ready, reason = MODULE._load_complete_reference(
                records, aggregate, expected_n=150
            )
            self.assertFalse(ready)
            self.assertEqual(reason, "reference_records_incomplete")

            with records.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"case_id": "case-149"}) + "\n")
            ready, reason = MODULE._load_complete_reference(
                records, aggregate, expected_n=150
            )
            self.assertTrue(ready)
            self.assertEqual(reason, "reference_complete")

    def test_rejects_wrong_arm_and_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            records = root / "records.jsonl"
            aggregate = root / "aggregate.json"
            records.write_text(
                "".join(
                    json.dumps({"case_id": "duplicate"}) + "\n" for _ in range(150)
                ),
                encoding="utf-8",
            )
            aggregate.write_text(
                json.dumps({"n": 150, "arm": "bounded_clarification_agent"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "one-shot"):
                MODULE._load_complete_reference(records, aggregate, expected_n=150)

            aggregate.write_text(
                json.dumps({"n": 150, "arm": "same_model_one_shot"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                MODULE._load_complete_reference(records, aggregate, expected_n=150)

    def test_state_files_are_private_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            state = Path(temporary_directory) / "state.json"
            MODULE._secure_exclusive_json(state, {"status": "waiting"})
            self.assertEqual(os.stat(state).st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                MODULE._secure_exclusive_json(state, {"status": "duplicate"})


if __name__ == "__main__":
    unittest.main()

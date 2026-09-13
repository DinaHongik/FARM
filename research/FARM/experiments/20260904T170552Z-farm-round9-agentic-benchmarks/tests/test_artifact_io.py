from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.artifact_io import assert_no_secrets, ordered_ids_sha256, read_jsonl, write_jsonl_atomic


class ArtifactIoTests(unittest.TestCase):
    def test_jsonl_round_trip_and_id_hash(self) -> None:
        rows = [{"case_id": "a", "value": 1}, {"case_id": "b", "value": 2}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl_atomic(path, rows)
            self.assertEqual(read_jsonl(path), rows)
        self.assertEqual(len(ordered_ids_sha256(rows)), 64)

    def test_secret_scanner_allows_opaque_ref_only(self) -> None:
        assert_no_secrets({"secret_ref": "calendar_account"})
        with self.assertRaisesRegex(ValueError, "credential"):
            assert_no_secrets({"api_key": "this-must-never-be-persisted"})
        with self.assertRaisesRegex(ValueError, "credential-shaped"):
            assert_no_secrets({"text": "Bearer abcdefghijklmnopqrstuvwxyz"})


if __name__ == "__main__":
    unittest.main()

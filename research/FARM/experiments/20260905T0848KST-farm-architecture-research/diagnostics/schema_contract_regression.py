"""Synthetic regression at the actual schema-ingestion seam.

Legacy run is expected to fail and is retained as evidence, not a benchmark.
FARM_SCHEMA_IMPL=evidence exercises the corrected independent implementation.
"""
import os
from pathlib import Path
import sys
import unittest

ROUND = Path(__file__).resolve().parents[1]
FARM = ROUND.parents[1]
sys.path.insert(0, str(FARM))
sys.path.insert(0, str(ROUND / "src"))
if os.environ.get("FARM_SCHEMA_IMPL") == "evidence":
    from farm_arch.schema import canonical_schema
else:
    from farm.dataset_v2 import canonical_schema


class SchemaContractRegression(unittest.TestCase):
    def test_input_type_and_help_are_not_dropped(self):
        component = {"api_info": {"Action fields": {"Threshold": {
            "Slug": "threshold", "Type": "Number", "Required": True,
            "Help text": "Use the threshold stated in the request.",
        }}}}
        fields, _, _ = canonical_schema([component], "action", "synthetic:action")
        self.assertEqual(fields[0].get("type"), "Number")
        self.assertEqual(fields[0].get("help_text"), "Use the threshold stated in the request.")

    def test_explicit_optional_label_conflict_is_not_silently_required(self):
        component = {"api_info": {"Trigger fields": {"Note (optional)": {
            "Slug": "note", "Required": True,
        }}}}
        fields, _, conflicts = canonical_schema([component], "trigger", "synthetic:trigger")
        self.assertIsNone(fields[0]["required"])
        self.assertTrue(any("required" in row.get("attributes", {}) for row in conflicts))


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND.parents[1]))
sys.path.insert(0, str(ROUND / "src"))
from farm_arch.schema import canonical_schema


class EvidenceSchemaTests(unittest.TestCase):
    def normalize(self, *entries):
        return canonical_schema([{"api_info": {"Action fields": {"Target": e}}} for e in entries], "action", "synthetic:a")

    def test_missing_type_is_unknown_not_any(self):
        fields, _, _ = self.normalize({"Slug": "target", "Required": True})
        self.assertEqual(fields[0]["type_state"], "unknown")
        self.assertEqual(fields[0]["type"], "")

    def test_duplicate_source_does_not_outvote_conflicting_requiredness(self):
        a, b = {"Slug": "target", "Required": True}, {"Slug": "target", "Required": False}
        first = self.normalize(a, a, a, b)
        second = self.normalize(b, a, a, a)
        self.assertEqual(first, second)
        self.assertIsNone(first[0][0]["required"])
        self.assertEqual(len(first[0][0]["source_evidence"]), 2)

    def test_helper_variants_and_raw_unknown_attributes_survive(self):
        fields, _, _ = self.normalize({"Slug": "target", "Helper text": "Use a folder.", "Widget": "chooser"})
        self.assertEqual(fields[0]["help_text"], "Use a folder.")
        self.assertEqual(fields[0]["source_evidence"][0]["metadata"]["Widget"], "chooser")

    def test_allow_default_does_not_create_default_value(self):
        fields, _, _ = self.normalize({"Slug": "target", "Can have default value": True})
        self.assertEqual(fields[0]["default_state"], "unknown")
        self.assertEqual(fields[0]["default_values"], [])

    def test_false_default_is_preserved(self):
        fields, _, _ = self.normalize({"Slug": "target", "Default": False})
        self.assertEqual(fields[0]["default_values"], [False])

    def test_conflicting_types_never_become_silent_modal_type(self):
        fields, _, issues = self.normalize({"Slug": "target", "Type": "String"}, {"Slug": "target", "Type": "Number"})
        self.assertEqual(fields[0]["type_state"], "conflicting")
        self.assertEqual(fields[0]["type"], "")
        self.assertTrue(any("type" in issue["attributes"] for issue in issues))

    def test_optional_hint_alone_does_not_supply_requiredness(self):
        fields, _, _ = self.normalize({"Slug": "target", "Label": "Target (optional)"})
        self.assertIsNone(fields[0]["required"])
        self.assertEqual(fields[0]["requiredness_state"], "unknown")


if __name__ == "__main__":
    unittest.main()

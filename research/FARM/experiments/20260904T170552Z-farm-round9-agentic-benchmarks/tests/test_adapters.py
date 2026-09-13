from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.adapters.common import normalize_label, split_fields
from farm_r9.adapters.recipegen import parse_target
from farm_r9.adapters.targe import parse_side


class AdapterTests(unittest.TestCase):
    def test_recipegen_field_names_are_not_values(self) -> None:
        parsed = parse_target(
            "Gmail <sep> Gmail.New_email <sep> Search for <sep> "
            "Dropbox <sep> Dropbox.Add_file <sep> File URL ### File name ### Folder"
        )
        self.assertEqual(parsed["trigger_fields"], ["Search for"])
        self.assertEqual(parsed["action_fields"], ["File URL", "File name", "Folder"])

    def test_targe_companion_parser(self) -> None:
        self.assertEqual(parse_side("TRIGGER SERVICE: iOS Photos, TRIGGER EVENT: New screenshot", "trigger"), ("iOS Photos", "New screenshot"))
        self.assertEqual(parse_side("ACTION SERVICE: iOS Photos, ACTION EVENT: Add photo to album", "action"), ("iOS Photos", "Add photo to album"))

    def test_normalization(self) -> None:
        self.assertEqual(normalize_label("Google_Calendar.Quick_add_event"), "google calendar quick add event")
        self.assertEqual(split_fields("Title ### Body ###  Tags "), ["Title", "Body", "Tags"])


if __name__ == "__main__":
    unittest.main()

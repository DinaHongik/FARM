from __future__ import annotations

import unittest

from build_dataset_v2 import make_groups
from farm.dataset_v2 import CORPUS_FILENAMES, canonical_schema, query_key, render_plain, render_schema


class ArtifactContractTests(unittest.TestCase):
    def test_corpus_filenames_are_explicit_and_grammatical(self):
        self.assertEqual(CORPUS_FILENAMES, {
            "trigger": "triggers.json",
            "action": "actions.json",
            "query": "queries.json",
        })


class UnicodeNormalizerTests(unittest.TestCase):
    def test_preserves_non_latin_scripts(self):
        values = [
            "카메라 켜기",
            "カメラをオン",
            "Ενεργοποίηση κάμερας",
            "Включить камеру",
            "تشغيل الكاميرا",
        ]
        keys = [query_key(value) for value in values]
        self.assertTrue(all(keys))
        self.assertEqual(len(keys), len(set(keys)))

    def test_nfkc_casefold_and_idempotence(self):
        self.assertEqual(query_key("ＴＶ\u00a0TIME"), "tv time")
        self.assertEqual(query_key("Straße"), "strasse")
        value = query_key("CO₂ alert")
        self.assertEqual(query_key(value), value)

    def test_keeps_order_and_rejects_symbol_only(self):
        self.assertNotEqual(query_key("WordPress to Buffer"), query_key("Buffer to WordPress"))
        self.assertEqual(query_key("🔥 → ✅"), "")


class CanonicalSchemaTests(unittest.TestCase):
    URL = "https://ifttt.com/demo/triggers/new_item"

    def test_duplicate_ingredient_slug_is_canonicalised_once(self):
        component = {
            "api_info": {
                "Ingredients": {
                    "ContactName": {"Slug": "ContactName", "Type": "String"},
                    "Contact name\nText": {
                        "Slug": "ContactName",
                        "Type": "String",
                        "Filter code": "Demo.newItem.ContactName",
                        "Example": "Ada",
                    },
                }
            }
        }
        _, ingredients, _ = canonical_schema([component], "trigger", self.URL)
        self.assertEqual(len(ingredients), 1)
        self.assertEqual(ingredients[0]["slug"], "ContactName")
        self.assertEqual(ingredients[0]["filter_code"], "Demo.newItem.ContactName")

    def test_critical_conflict_is_modal_and_explicit(self):
        def component(kind):
            return {"api_info": {"Ingredients": {"OccurredAt": {
                "Slug": "OccurredAt", "Filter code": "Demo.newItem.OccurredAt", "Type": kind,
            }}}}

        occurrences = [component("String"), component("DateTime"), component("String")]
        _, ingredients, conflicts = canonical_schema(occurrences, "trigger", self.URL)
        self.assertEqual(ingredients[0]["type"], "String")
        self.assertEqual(conflicts[0]["attributes"]["type"], ["DateTime", "String"])
        _, reversed_ingredients, _ = canonical_schema(list(reversed(occurrences)), "trigger", self.URL)
        self.assertEqual(reversed_ingredients, ingredients)

    def test_schema_view_is_a_controlled_plain_prefix(self):
        record = {
            "kind": "action",
            "channel": "google_sheets",
            "function_name": "Add row",
            "description": "Append a row to a spreadsheet.",
            "categories": ["Popular services"],
            "input_fields": [{
                "slug": "row", "label": "Row", "required": True,
                "can_have_default": True, "filter_code_method": "Sheets.addRow.setRow", "bindable": True,
            }],
            "ingredients": [],
        }
        plain = render_plain(record)
        schema = render_schema(record)
        self.assertTrue(schema.startswith(plain))
        self.assertNotIn("Popular services", plain)
        self.assertNotIn("Sheets.addRow", schema)


class MultiGoldTests(unittest.TestCase):
    def test_pair_truth_does_not_become_a_cartesian_product(self):
        rows = [
            {
                "applet_url": "https://ifttt.com/applets/a-one",
                "query": "Turn off camera when I am home",
                "query_norm": query_key("Turn off camera when I am home"),
                "additional_description": "",
                "trigger_url": "https://ifttt.com/location/triggers/exit",
                "action_url": "https://ifttt.com/camera_a/actions/off",
                "trigger_channel": "location",
                "action_channel": "camera_a",
                "user_counts": [],
                "quality_flags": [],
            },
            {
                "applet_url": "https://ifttt.com/applets/b-two",
                "query": "Turn off camera when I am home",
                "query_norm": query_key("Turn off camera when I am home"),
                "additional_description": "",
                "trigger_url": "https://ifttt.com/location/triggers/enter",
                "action_url": "https://ifttt.com/camera_b/actions/off",
                "trigger_channel": "location",
                "action_channel": "camera_b",
                "user_counts": [],
                "quality_flags": [],
            },
        ]
        groups, _, _ = make_groups(rows, aliases=[])
        self.assertEqual(len(groups), 1)
        valid = {(row["trigger_url"], row["action_url"]) for row in groups[0]["valid_pairs"]}
        self.assertEqual(len(valid), 2)
        self.assertNotIn(
            ("https://ifttt.com/location/triggers/exit", "https://ifttt.com/camera_b/actions/off"),
            valid,
        )


if __name__ == "__main__":
    unittest.main()

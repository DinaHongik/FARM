from pathlib import Path
import sys
import unittest

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "scripts"))
from audit_configuration_inputs import index_raw
from build_schema_sidecar import build_record


class SourceIndexTests(unittest.TestCase):
    def test_malformed_scrape_containers_are_counted_not_crashes(self):
        versions, _, anomalies = index_raw([{"applets": "timeout"}, {"applets": ["error", {}]}])
        self.assertEqual(dict(versions), {})
        self.assertEqual(anomalies["service_or_applet_collection_invalid"], 1)
        self.assertEqual(anomalies["applet_not_object"], 1)
        self.assertEqual(anomalies["component_collection_not_list"], 1)

    def test_missing_api_info_is_an_empty_evidence_record(self):
        url = "https://ifttt.com/example/actions/add"
        versions, _, _ = index_raw([{"applets": [{"components": [{"url": url}]}]}])
        record = build_record({"url": url, "input_fields": []}, "action", versions)
        self.assertEqual(record["input_fields"], [])
        self.assertEqual(record["validation_capabilities"]["execution"], "unavailable")

    def test_source_only_fields_do_not_change_frozen_field_identity(self):
        url = "https://ifttt.com/example/actions/add"
        raw = [{"applets": [{"components": [{"url": url, "api_info": {"Action fields": {
            "A": {"Slug": "a", "Required": True}, "B": {"Slug": "b", "Required": False}}}}]}]}]
        versions, _, _ = index_raw(raw)
        record = build_record({"url": url, "input_fields": [{"slug": "a"}]}, "action", versions)
        self.assertEqual([f["slug"] for f in record["input_fields"]], ["a"])
        self.assertEqual([f["slug"] for f in record["source_only_input_fields"]], ["b"])


if __name__ == "__main__":
    unittest.main()

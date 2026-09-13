from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROUND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROUND.parents[1]), str(ROUND / "src"), str(ROUND / "scripts")]
from farm_arch.configuration import ConfigurationSession, Decision, Endpoint
from farm_arch.sidecar import endpoint_from_sidecar
from build_schema_sidecar import build_record
from audit_configuration_inputs import index_raw


def record(*, with_type=False):
    metadata = {"Slug": "target", "Label": "Target", "Required": True, "Helper text": "Choose a folder."}
    if with_type:
        metadata["Type"] = "String"
    # Same source-normalizer -> sidecar -> compiler seam as real FARM inputs.
    return build_record({"url": "synthetic:a", "input_fields": [{"slug": "target"}]}, "action",
        {("action", "synthetic:a"): {"synthetic-revision": {"api_info": {"Action fields": {"Target": metadata}}}}})


class SidecarIntegrationTests(unittest.TestCase):
    def test_raw_metadata_reaches_configuration_interface_without_type_invention(self):
        endpoint = endpoint_from_sidecar(record())
        self.assertEqual(endpoint.fields[0].help_text, "Choose a folder.")
        self.assertIsNone(endpoint.fields[0].value_type)
        session = ConfigurationSession(query="green", trigger=Endpoint("t", "trigger", "r"), action=endpoint)
        result = session.compile((Decision("action", "target", {"kind": "query_span", "start": 0,
            "end": 5, "text": "green", "transform": "identity"}),))
        self.assertEqual(result.status, "needs_evidence")
        self.assertFalse(result.execution_verified)

    def test_declared_synthetic_type_permits_only_local_checks(self):
        endpoint = endpoint_from_sidecar(record(with_type=True))
        session = ConfigurationSession(query="green", trigger=Endpoint("t", "trigger", "r"), action=endpoint)
        result = session.compile((Decision("action", "target", {"kind": "query_span", "start": 0,
            "end": 5, "text": "green", "transform": "identity"}),))
        self.assertEqual(result.status, "locally_checked")

    def test_source_evidence_change_invalidates_schema_revision(self):
        first = record()
        second = deepcopy(first)
        second["input_fields"][0]["help_text"] = "Updated source guidance."
        self.assertNotEqual(endpoint_from_sidecar(first).schema_revision, endpoint_from_sidecar(second).schema_revision)

    def test_unreconciled_extra_field_fails_closed(self):
        row = record()
        row["source_only_input_fields"] = [{"slug": "extra"}]
        with self.assertRaisesRegex(ValueError, "unreconciled"):
            endpoint_from_sidecar(row)


if __name__ == "__main__":
    unittest.main()

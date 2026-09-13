from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.endpoint_agent import validate_prediction
from farm_r9.contracts import EndpointPrediction
from farm_r9.metrics_endpoint import score_endpoint


class EndpointAgentTests(unittest.TestCase):
    def test_validation_requires_exact_selected_schema_fields_and_citations(self):
        case = {"public_evidence": {
            "trigger_candidates": [{"alias": "T01", "field_names": ["Topic"]}],
            "action_candidates": [{"alias": "A01", "field_names": ["Body", "Folder"]}],
        }}
        valid, errors = validate_prediction({
            "trigger_alias": "T01", "action_alias": "A01",
            "trigger_field_names": ["Topic"], "action_field_names": ["Body", "Folder"],
            "preview": "Save matching items.", "evidence_aliases": ["T01", "A01"],
        }, case)
        self.assertIsNotNone(valid)
        self.assertEqual(errors, [])
        invalid, errors = validate_prediction({
            "trigger_alias": "T01", "action_alias": "A01",
            "trigger_field_names": [], "action_field_names": ["invented"],
            "preview": "Save matching items.", "evidence_aliases": ["T01"],
        }, case)
        self.assertIsNone(invalid)
        self.assertEqual({error["code"] for error in errors}, {
            "trigger_field_names_not_exact_schema", "action_field_names_not_exact_schema", "missing_action_citation",
        })

    def test_farm_service_scoring_uses_canonical_id_not_display_label(self):
        case = {
            "benchmark": "farm_v2_test",
            "public_evidence": {
                "trigger_candidates": [{"alias": "T01", "service": "IFTTT Notifications", "service_id": "if_notifications"}],
                "action_candidates": [{"alias": "A01", "service": "Dropbox", "service_id": "dropbox"}],
            },
            "private": {
                "trigger_alias_map": {"T01": "trigger-id"},
                "action_alias_map": {"A01": "action-id"},
                "gold": {
                    "trigger_ids": ["trigger-id"], "action_ids": ["action-id"],
                    "channel_pairs": [{"trigger_channel": "if_notifications", "action_channel": "dropbox"}],
                },
            },
        }
        prediction = EndpointPrediction(
            trigger_alias="T01", action_alias="A01", trigger_field_names=[], action_field_names=[],
            preview="Connect them.", evidence_aliases=["T01", "A01"],
        )
        score = score_endpoint(case, prediction)
        self.assertTrue(score["service_joint"])
        self.assertTrue(score["function_joint"])

    def test_external_field_exact_never_credits_a_wrong_endpoint(self):
        case = {
            "benchmark": "recipegen_gold",
            "public_evidence": {
                "trigger_candidates": [{"alias": "T01", "service": "Wrong", "function": "Wrong", "field_names": []}],
                "action_candidates": [{"alias": "A01", "service": "Wrong", "function": "Wrong", "field_names": []}],
            },
            "private": {"gold": {
                "trigger_channel_norm": "right", "action_channel_norm": "right",
                "trigger_function_norm": "right", "action_function_norm": "right",
                "trigger_fields_norm": [], "action_fields_norm": [],
            }},
        }
        prediction = EndpointPrediction(
            trigger_alias="T01", action_alias="A01", trigger_field_names=[], action_field_names=[],
            preview="Wrong selection.", evidence_aliases=["T01", "A01"],
        )
        score = score_endpoint(case, prediction)
        self.assertFalse(score["field_trigger_exact"])
        self.assertFalse(score["field_action_exact"])
        self.assertFalse(score["field_joint_exact"])


if __name__ == "__main__":
    unittest.main()

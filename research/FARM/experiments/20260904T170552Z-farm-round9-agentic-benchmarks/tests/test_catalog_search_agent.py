from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_endpoint_runner import _case
from farm_r9.catalog_search_agent import execute_search_case


class Catalog:
    def __init__(self, case):
        self.rows = {}
        self.documents = {}
        for side in ("trigger", "action"):
            rows = copy.deepcopy(case["public_evidence"][side + "_candidates"])
            for row in rows:
                row.update(id=row["alias"], side=side, description="public schema")
                if side == "trigger":
                    row["function"] = "incorrect event"
                self.documents[row["id"]] = row
            self.rows[side] = rows
        self.documents["TNEW"] = {"id": "TNEW", "side": "trigger", "service": "svc",
            "function": "good", "description": "public matching event", "field_names": []}
    def initial(self, case):
        return self.rows
    def search(self, arguments):
        return [self.documents["TNEW"]]


def response(name, arguments):
    return SimpleNamespace(content="", tool_calls=[{"function": {"name": name, "arguments": arguments}}],
        raw_response={"done": True, "done_reason": "stop"}, prompt_tokens=1,
        completion_tokens=1, physical_attempts=1, cache_hit=False,
        provider_latency_ms=1, queue_wait_ms=0)


class Client:
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []
    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return next(self.outputs)


class CatalogAgentTests(unittest.TestCase):
    def fixture(self):
        c = _case()
        c["benchmark"] = "recipegen_noisy"
        return c, Catalog(c)

    def test_fixed_candidate_arm_cannot_select_unobserved_gold(self):
        c, catalog = self.fixture()
        record = execute_search_case(case=c, catalog=catalog, arm="native_fixed_top10",
            client=Client([response("commit_applet", {"trigger_id": "TNEW", "action_id": "A04"})]))
        self.assertFalse(record["initial_joint_coverage"])
        self.assertFalse(record["scores"]["function_joint"])
        self.assertEqual(record["failure_code"], "unobserved_or_wrong_side_selection")

    def test_search_can_recover_a_missing_trigger_without_gold_in_messages(self):
        c, catalog = self.fixture()
        client = Client([response("search_functions", {"side": "trigger", "query": "event", "service": "svc"}),
                         response("commit_applet", {"trigger_id": "TNEW", "action_id": "A04"})])
        record = execute_search_case(case=c, catalog=catalog, arm="native_search_agent", client=client)
        self.assertFalse(record["initial_joint_coverage"])
        self.assertTrue(record["observed_joint_coverage"])
        self.assertTrue(record["scores"]["function_joint"])
        self.assertEqual(record["search_calls"], 1)
        self.assertNotIn("TNEW", str(client.calls[0]["messages"]))
        self.assertIn("TNEW", str(client.calls[1]["messages"]))
        self.assertNotIn("private", str(client.calls[0]["messages"][0]))
        self.assertEqual(client.calls[1]["messages"][-1]["role"], "tool")

    def test_private_farm_input_is_rejected_before_any_model_call(self):
        c, catalog = self.fixture()
        c["data_classification"] = "confidential"
        client = Client([])
        with self.assertRaises(ValueError):
            execute_search_case(case=c, catalog=catalog, arm="native_search_agent", client=client)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()

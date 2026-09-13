from __future__ import annotations
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from test_yao_agent import CATALOG, case
from farm_r9.yao_native import execute_native_case


def output(name, arguments, reason="stop"):
    return SimpleNamespace(content="", tool_calls=({"function": {
        "name": name, "arguments": arguments}},), raw_response={"done": True, "done_reason": reason},
        prompt_tokens=10, completion_tokens=5, physical_attempts=1,
        cache_hit=False, provider_latency_ms=1, queue_wait_ms=0)


class Client:
    model = "fixture"
    def __init__(self, outputs):
        self.outputs, self.calls = iter(outputs), []
    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return next(self.outputs)


class NativeYaoTests(unittest.TestCase):
    def labels(self):
        return {key: values[0] for key, values in CATALOG.items()}

    def test_one_shot_withholds_all_answers_and_uses_native_schema(self):
        client = Client([output("commit_applet", self.labels())])
        record = execute_native_case(case=case(), catalog=CATALOG, arm="native_one_shot", client=client)
        self.assertTrue(record["scores"]["joint"])
        self.assertEqual(record["question_count"], 0)
        self.assertEqual(client.calls[0]["think"], False)
        self.assertEqual(client.calls[0]["max_output_tokens"], 8192)
        self.assertEqual(len(client.calls[0]["tools"]), 1)
        self.assertNotIn("answer for", str(client.calls[0]["messages"]))
        self.assertNotIn("pseudo_ask", str(client.calls[0]))

    def test_adaptive_preserves_native_tool_history_and_only_requested_answer(self):
        client = Client([output("ask_component", {"component": "trigger_channel"}),
                         output("commit_applet", self.labels())])
        record = execute_native_case(case=case(), catalog=CATALOG, arm="native_adaptive", client=client)
        self.assertEqual(record["question_count"], 1)
        history = client.calls[1]["messages"]
        self.assertEqual(history[-2]["role"], "assistant")
        self.assertEqual(history[-2]["tool_calls"][0]["function"]["name"], "ask_component")
        self.assertEqual(history[-1]["role"], "tool")
        self.assertEqual(history[-1]["tool_name"], "ask_component")
        self.assertIn("answer for trigger_channel", history[-1]["content"])
        self.assertNotIn("answer for action_function", str(history))

    def test_ask_all_counts_four_questions_and_one_batched_turn(self):
        record = execute_native_case(case=case(), catalog=CATALOG, arm="native_ask_all",
                                     client=Client([output("commit_applet", self.labels())]))
        self.assertEqual(record["question_count"], 4)
        self.assertEqual(record["interaction_turns"], 1)
        self.assertEqual(record["usage"]["semantic_calls"], 1)

    def test_length_termination_is_explicit_failure_even_with_arguments(self):
        record = execute_native_case(case=case(), catalog=CATALOG, arm="native_one_shot",
                                     client=Client([output("commit_applet", self.labels(), "length")]))
        self.assertEqual(record["failure_code"], "incomplete_generation")
        self.assertFalse(record["scores"]["joint"])

    def test_repeated_questions_are_rejected(self):
        question = output("ask_component", {"component": "trigger_channel"})
        record = execute_native_case(case=case(), catalog=CATALOG, arm="native_adaptive",
                                     client=Client([question, question]))
        self.assertEqual(record["failure_code"], "invalid_or_repeated_question")
        self.assertEqual(record["question_count"], 1)

    def test_noncanonical_labels_are_not_scored_or_filled(self):
        labels = self.labels()
        labels["action_channel"] = "not a real label"
        record = execute_native_case(case=case(), catalog=CATALOG, arm="native_one_shot",
                                     client=Client([output("commit_applet", labels)]))
        self.assertIsNone(record["prediction"])
        self.assertEqual(record["failure_code"], "noncanonical_label")


if __name__ == "__main__":
    unittest.main()

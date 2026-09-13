from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK))

from audit_targe_release import RELEASE_FILES, audit_release  # noqa: E402
from evaluator import Endpoint, Recipe, evaluate_payload, evaluate_rankings  # noqa: E402


def recipe(trigger_channel: str, trigger_function: str, action_channel: str, action_function: str) -> Recipe:
    return Recipe(
        trigger=Endpoint(channel=trigger_channel, function=trigger_function),
        action=Endpoint(channel=action_channel, function=action_function),
    )


def recipe_value(value: Recipe) -> dict:
    return {
        "trigger": {"channel": value.trigger.channel, "function": value.trigger.function},
        "action": {"channel": value.action.channel, "function": value.action.function},
    }


class EvaluatorTests(unittest.TestCase):
    def test_empty_ranking_receives_zero_for_every_metric(self):
        reference = recipe("Trigger Service", "New item", "Action Service", "Send item")

        report = evaluate_rankings([reference], [[]], cutoffs=(1, 3, 5))

        self.assertEqual(report["empty_rankings"], 1)
        self.assertEqual(report["top1"]["trigger_em"]["score"], 0.0)
        self.assertEqual(report["top1"]["action_em"]["score"], 0.0)
        self.assertEqual(report["top1"]["recipe_em"]["score"], 0.0)
        for cutoff in ("1", "3", "5"):
            for component in ("trigger", "action", "recipe"):
                self.assertEqual(report["at_k"][cutoff][f"{component}_recall"]["score"], 0.0)
                self.assertEqual(report["at_k"][cutoff][f"{component}_mrr"]["score"], 0.0)
                self.assertEqual(report["full_ranking"][f"{component}_mrr"]["score"], 0.0)

    def test_exact_top1_match_receives_one_for_both_sides_and_recipe(self):
        reference = recipe("Trigger Service", "New item", "Action Service", "Send item")

        report = evaluate_rankings([reference], [[reference]], cutoffs=(1, 5))

        self.assertEqual(report["top1"]["trigger_em"]["score"], 1.0)
        self.assertEqual(report["top1"]["action_em"]["score"], 1.0)
        self.assertEqual(report["top1"]["recipe_em"]["score"], 1.0)
        self.assertEqual(report["at_k"]["1"]["recipe_recall"]["score"], 1.0)
        self.assertEqual(report["at_k"]["1"]["recipe_mrr"]["score"], 1.0)

    def test_side_and_recipe_ranks_are_factorized_and_exact(self):
        reference = recipe("T", "tf", "A", "af")
        ranking = [
            recipe("T", "tf", "wrong-a", "wrong-af"),
            recipe("wrong-t", "wrong-tf", "A", "af"),
            reference,
        ]

        report = evaluate_rankings([reference], [ranking], cutoffs=(1, 2, 3))

        self.assertEqual(report["top1"]["trigger_em"]["score"], 1.0)
        self.assertEqual(report["top1"]["action_em"]["score"], 0.0)
        self.assertEqual(report["top1"]["recipe_em"]["score"], 0.0)
        self.assertEqual(report["at_k"]["2"]["action_recall"]["score"], 1.0)
        self.assertEqual(report["at_k"]["2"]["action_mrr"]["score"], 0.5)
        self.assertEqual(report["at_k"]["2"]["recipe_recall"]["score"], 0.0)
        self.assertEqual(report["at_k"]["3"]["recipe_mrr"]["score"], 1.0 / 3.0)

    def test_payload_parser_rejects_length_mismatch_and_malformed_types(self):
        reference = recipe_value(recipe("T", "tf", "A", "af"))
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            evaluate_payload({"references": [reference], "rankings": []})
        with self.assertRaisesRegex(ValueError, "must be an array"):
            evaluate_payload({"references": [reference], "rankings": ["not-a-ranking"]})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluate_payload({"references": [reference], "rankings": [[]]}, cutoffs=(True,))

    def test_payload_parser_preserves_case_sensitive_exactness(self):
        reference = recipe("Service", "Function", "Action", "Do")
        prediction = recipe("service", "Function", "Action", "Do")
        report = evaluate_payload(
            {"references": [recipe_value(reference)], "rankings": [[recipe_value(prediction)]]},
            cutoffs=(1,),
        )
        self.assertEqual(report["top1"]["trigger_em"]["score"], 0.0)
        self.assertEqual(report["top1"]["action_em"]["score"], 1.0)
        self.assertEqual(report["top1"]["recipe_em"]["score"], 0.0)


class ReleaseAuditTests(unittest.TestCase):
    @staticmethod
    def _write_split(root: Path, name: str, records: list[dict]) -> bytes:
        path = root / RELEASE_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")
        path.write_bytes(raw)
        return raw

    def test_audit_reports_hashes_duplicates_and_exact_overlap_without_writes(self):
        a = {"input": "query-a", "output": "recipe-a"}
        b = {"input": "query-b", "output": "recipe-b"}
        c = {"input": "query-c", "output": "recipe-c"}
        d = {"input": "query-d", "output": "recipe-d"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_raw = self._write_split(root, "train", [a, a, b])
            self._write_split(root, "gold", [a, c, c])
            self._write_split(root, "noisy", [d])
            self._write_split(root, "oneshot", [c])
            before = sorted(path.relative_to(root) for path in root.rglob("*"))

            report = audit_release(root)

            after = sorted(path.relative_to(root) for path in root.rglob("*"))
            self.assertEqual(before, after)
            self.assertTrue(report["read_only"])
            self.assertEqual(report["splits"]["train"]["rows"], 3)
            self.assertEqual(report["splits"]["train"]["unique_input_output_tuples"], 2)
            self.assertEqual(report["splits"]["train"]["duplicate_rows_beyond_first"], 1)
            self.assertEqual(
                report["splits"]["train"]["sha256"], hashlib.sha256(train_raw).hexdigest()
            )
            self.assertEqual(report["overlap_with_train"]["gold"]["unique_tuple_overlap"], 1)
            self.assertEqual(report["overlap_with_train"]["noisy"]["unique_tuple_overlap"], 0)
            self.assertEqual(report["pairwise_unique_tuple_overlap"]["gold__oneshot"]["unique_tuple_overlap"], 1)

    def test_audit_rejects_malformed_release_record(self):
        valid = [{"input": "q", "output": "r"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in RELEASE_FILES:
                self._write_split(root, name, valid)
            self._write_split(root, "gold", [{"input": "missing output"}])
            with self.assertRaisesRegex(ValueError, r"gold\[0\]\.output"):
                audit_release(root)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.candidate_sets import opaque_candidate_view, reciprocal_rank_fusion


class CandidateTests(unittest.TestCase):
    def test_rrf_is_input_order_invariant_and_no_gold_injection(self):
        a = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "q"]])
        b = reciprocal_rank_fusion([["y", "x", "q"], ["x", "y", "z"]])
        self.assertEqual(a, b)
        self.assertNotIn("gold", {identifier for identifier, _ in a})

    def test_opaque_view_hides_identity_rank_and_score(self):
        candidates = [
            {"canonical_id": f"id-{index}", "rank": index, "score": 1 / index, "side": "trigger", "service": "s", "function": str(index)}
            for index in range(1, 11)
        ]
        public, alias_map = opaque_candidate_view(candidates, case_id="case", side="trigger", seed=42)
        self.assertEqual(len(public), 10)
        self.assertEqual(set(alias_map), {f"T{index:02d}" for index in range(1, 11)})
        self.assertTrue(all("canonical_id" not in row and "rank" not in row and "score" not in row for row in public))
        self.assertNotEqual([row["function"] for row in public], [str(index) for index in range(1, 11)])


if __name__ == "__main__":
    unittest.main()

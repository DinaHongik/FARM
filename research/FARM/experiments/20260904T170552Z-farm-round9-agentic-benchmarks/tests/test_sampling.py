from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.sampling import proportional_quotas, stratified_sample


class SamplingTests(unittest.TestCase):
    def test_exact_reproducible_stratified_sample(self) -> None:
        rows = [
            {"case_id": f"a-{index}", "stratum": "a"} for index in range(200)
        ] + [
            {"case_id": f"b-{index}", "stratum": "b"} for index in range(100)
        ]
        first, manifest = stratified_sample(
            rows,
            size=150,
            seed=42,
            benchmark="fixture",
            id_of=lambda row: row["case_id"],
            stratum_of=lambda row: row["stratum"],
        )
        second, _ = stratified_sample(
            list(reversed(rows)),
            size=150,
            seed=42,
            benchmark="fixture",
            id_of=lambda row: row["case_id"],
            stratum_of=lambda row: row["stratum"],
        )
        self.assertEqual([row["case_id"] for row in first], [row["case_id"] for row in second])
        self.assertEqual(manifest["sample_strata"], {"a": 100, "b": 50})
        self.assertEqual(len(first), 150)

    def test_small_stratum_is_retained_when_possible(self) -> None:
        self.assertEqual(proportional_quotas({"rare": 1, "common": 999}, 10)["rare"], 1)

    def test_reject_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            stratified_sample(
                [{"case_id": "x", "stratum": "a"}, {"case_id": "x", "stratum": "b"}],
                size=1,
                seed=1,
                benchmark="fixture",
                id_of=lambda row: row["case_id"],
                stratum_of=lambda row: row["stratum"],
            )


if __name__ == "__main__":
    unittest.main()

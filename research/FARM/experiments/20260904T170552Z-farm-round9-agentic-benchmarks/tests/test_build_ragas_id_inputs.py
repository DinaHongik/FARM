from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("build_ragas_id_inputs", ROOT / "scripts" / "build_ragas_id_inputs.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class RagasInputBuilderTests(unittest.TestCase):
    def test_single_gold_ids_are_side_prefixed_only_for_joint(self):
        row = {
            "benchmark": "recipegen_gold", "case_id": "c1",
            "private": {
                "trigger": {"alias_map": {f"T{i:02d}": f"ts::tf{i}" for i in range(1, 11)}},
                "action": {"alias_map": {f"A{i:02d}": f"as::af{i}" for i in range(1, 11)}},
                "gold": {
                    "trigger_channel_norm": "ts", "trigger_function_norm": "tf1",
                    "action_channel_norm": "as", "action_function_norm": "af1",
                },
            },
        }
        rows = [dict(row, case_id=f"c{i}") for i in range(150)]
        trigger = module.build(rows, "trigger")[0]
        joint = module.build(rows, "joint")[0]
        self.assertEqual(trigger["reference_context_ids"], ["ts::tf1"])
        self.assertEqual(len(joint["retrieved_context_ids"]), 20)
        self.assertEqual(joint["reference_context_ids"], ["trigger:ts::tf1", "action:as::af1"])


if __name__ == "__main__":
    unittest.main()

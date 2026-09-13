from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_recipegen_noisy_after_bfcl",
    ROOT / "scripts" / "launch_recipegen_noisy_after_bfcl.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BFCLCompletionTests(unittest.TestCase):
    def test_requires_complete_official_native_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            ready, reason = MODULE._summary_complete(path, "registry")
            self.assertFalse(ready)
            self.assertEqual(reason, "summary_pending")
            value = {
                "registry_name": "registry",
                "arm": "native_tool_agent",
                "generation": {"n": 150, "resumable_completed_cases": 149},
                "evaluation": {"n": 150, "official_evaluator": True},
                "official_state_based_evaluator": True,
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                MODULE._summary_complete(path, "registry"),
                (False, "summary_incomplete"),
            )
            value["generation"]["resumable_completed_cases"] = 150
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(
                MODULE._summary_complete(path, "registry"),
                (True, "summary_complete"),
            )

    def test_rejects_wrong_registry_or_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                json.dumps({"registry_name": "wrong", "arm": "native_tool_agent"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "registry"):
                MODULE._summary_complete(path, "expected")


if __name__ == "__main__":
    unittest.main()

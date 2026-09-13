from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.agent_state import fresh_state
from farm_r9.contracts import CompilationResult, ValidationIssue


class AgentStateTests(unittest.TestCase):
    def test_fresh_state_and_bounded_nonrepeating_questions(self):
        first = fresh_state("case-a", max_questions=2)
        second = fresh_state("case-b", max_questions=2)
        first.ask("trigger_function")
        first.observe(component="trigger_function", answer="new photo", source="official_simulator")
        self.assertEqual(second.question_count, 0)
        with self.assertRaisesRegex(ValueError, "repeated"):
            first.ask("trigger_function")
        first.ask("action_function")
        with self.assertRaisesRegex(ValueError, "budget"):
            first.ask("trigger_service")

    def test_repair_requires_real_repairable_validation_error(self):
        state = fresh_state("case-a")
        with self.assertRaisesRegex(ValueError, "validation"):
            state.begin_repair()
        invalid = CompilationResult(
            valid=False, structurally_complete=False, executable_ready=False,
            issues=(ValidationIssue(code="missing_field_decision", side="action", field_slug="body", message="missing", repairable=True),),
        )
        state.last_validation = invalid
        issues = state.begin_repair()
        self.assertEqual(issues[0]["code"], "missing_field_decision")
        with self.assertRaisesRegex(ValueError, "budget"):
            state.begin_repair()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


if importlib.util.find_spec("pydantic") is None:
    raise unittest.SkipTest("pydantic v2 is tested in the pinned DGX environment")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.compiler import AppletCompiler
from farm_r9.contracts import AppletDraft, EndpointCandidate, ResourceObservation


def candidates():
    trigger = EndpointCandidate.model_validate({
        "alias": "T01", "side": "trigger", "service": "News", "function": "New article",
        "fields": [{"slug": "topic", "label": "Topic", "required": True, "bindable": False, "value_type": "string"}],
        "ingredients": [{"slug": "ArticleUrl", "label": "Article URL", "value_type": "string"}],
    })
    action = EndpointCandidate.model_validate({
        "alias": "A01", "side": "action", "service": "Notes", "function": "Create note",
        "fields": [
            {"slug": "body", "label": "Body", "required": True, "bindable": True, "value_type": "string"},
            {"slug": "notebook", "label": "Notebook", "required": True, "bindable": False, "value_type": "string", "resource_like": True},
            {"slug": "tags", "label": "Tags", "required": False, "bindable": True, "value_type": "string"},
        ],
    })
    return trigger, action


class CompilerTests(unittest.TestCase):
    def test_grounded_literal_binding_resource_and_optional_omit(self):
        trigger, action = candidates()
        query = "Send recipe articles to my Research notebook"
        start = query.index("recipe")
        draft = AppletDraft.model_validate({
            "selection": {"trigger_alias": "T01", "action_alias": "A01"},
            "trigger_fields": [{"field_slug": "topic", "source": {"kind": "query_literal", "start": start, "end": start + 6, "text": "recipe"}}],
            "action_fields": [
                {"field_slug": "body", "source": {"kind": "trigger_output", "ingredient_slug": "ArticleUrl"}},
                {"field_slug": "notebook", "source": {"kind": "resource_ref", "resource_id": "notebook-7", "observation_id": "obs-1"}},
                {"field_slug": "tags", "source": {"kind": "omit", "reason": "not requested"}},
            ],
            "preview": "Watch recipe articles and save their URL to Research.", "evidence_aliases": ["T01", "A01"],
        })
        result = AppletCompiler(
            query=query, trigger_candidates=[trigger], action_candidates=[action],
            resource_observations=[ResourceObservation(observation_id="obs-1", resource_ids=("notebook-7",))],
        ).compile(draft)
        self.assertTrue(result.valid)
        self.assertTrue(result.executable_ready)

    def test_rejects_fabricated_literal_and_non_bindable_binding(self):
        trigger, action = candidates()
        draft = AppletDraft.model_validate({
            "selection": {"trigger_alias": "T01", "action_alias": "A01"},
            "trigger_fields": [{"field_slug": "topic", "source": {"kind": "query_literal", "start": 0, "end": 6, "text": "sports"}}],
            "action_fields": [
                {"field_slug": "body", "source": {"kind": "trigger_output", "ingredient_slug": "ArticleUrl"}},
                {"field_slug": "notebook", "source": {"kind": "trigger_output", "ingredient_slug": "ArticleUrl"}},
                {"field_slug": "tags", "source": {"kind": "omit", "reason": "not requested"}},
            ], "preview": "Draft", "evidence_aliases": ["T01", "A01"],
        })
        result = AppletCompiler(query="recipe", trigger_candidates=[trigger], action_candidates=[action]).compile(draft)
        self.assertFalse(result.valid)
        self.assertIn("ungrounded_query_literal", {issue.code for issue in result.issues})
        self.assertIn("non_bindable_field", {issue.code for issue in result.issues})

    def test_required_needs_input_is_honest_but_not_executable(self):
        trigger, action = candidates()
        draft = AppletDraft.model_validate({
            "selection": {"trigger_alias": "T01", "action_alias": "A01"},
            "trigger_fields": [{"field_slug": "topic", "source": {"kind": "needs_input", "question": "Which topic should I monitor?"}}],
            "action_fields": [
                {"field_slug": "body", "source": {"kind": "trigger_output", "ingredient_slug": "ArticleUrl"}},
                {"field_slug": "notebook", "source": {"kind": "needs_input", "question": "Which notebook should receive it?"}},
                {"field_slug": "tags", "source": {"kind": "omit", "reason": "not requested"}},
            ], "preview": "Draft awaiting two choices.", "evidence_aliases": ["T01", "A01"],
        })
        result = AppletCompiler(query="Save articles", trigger_candidates=[trigger], action_candidates=[action]).compile(draft)
        self.assertTrue(result.valid)
        self.assertTrue(result.structurally_complete)
        self.assertFalse(result.executable_ready)


if __name__ == "__main__":
    unittest.main()

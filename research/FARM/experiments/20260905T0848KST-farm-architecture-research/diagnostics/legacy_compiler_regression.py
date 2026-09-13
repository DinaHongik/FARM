"""Synthetic counterexamples to Round9 compiler readiness; expected legacy red.

Run with the existing Round9 src on PYTHONPATH. No private inputs or model calls.
These are engineering regressions, not benchmark success measurements.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

from farm_r9.compiler import AppletCompiler
from farm_r9.contracts import (AppletDraft, EndpointCandidate, EndpointSelection,
    FieldDecision, FieldSpec, QueryLiteral, ResourceObservation, ResourceReference)


class LegacyCompilerRegression(unittest.TestCase):
    def fixture(self, value_type="string", source=None, observations=()):
        trigger = EndpointCandidate(alias="T01", side="trigger", service="Demo", function="Event")
        action = EndpointCandidate(alias="A01", side="action", service="Demo", function="Act",
            fields=(FieldSpec(slug="target", label="Target", required=True, value_type=value_type),))
        compiler = AppletCompiler(query="green", trigger_candidates=(trigger,), action_candidates=(action,),
            resource_observations=observations)
        draft = AppletDraft(selection=EndpointSelection(trigger_alias="T01", action_alias="A01"),
            action_fields=(FieldDecision(field_slug="target", source=source or
                QueryLiteral(kind="query_literal", start=0, end=5, text="green")),), preview="Synthetic applet")
        if os.environ.get("FARM_COMPILER_IMPL") == "evidence":
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
            from farm_arch.configuration import ConfigurationSession, Decision, Endpoint, Field, Observation
            converted = draft.action_fields[0].source.model_dump()
            if converted["kind"] == "query_literal":
                converted.update(kind="query_span", transform="identity")
            # Old observations contain no scope. Do not fabricate one during
            # migration: an unscoped observation cannot certify a chosen field.
            evidence = tuple(Observation(o.observation_id, "unscoped", "unscoped", "unscoped", o.resource_ids)
                for o in observations)
            session = ConfigurationSession(query="green", trigger=Endpoint("t", "trigger", "r"),
                action=Endpoint("a", "action", "r", (Field("target", True, value_type),)), observations=evidence)
            result = session.compile((Decision("action", "target", converted),))
            return SimpleNamespace(valid=result.locally_valid, executable_ready=result.execution_verified)
        return compiler.compile(draft)

    def test_query_span_does_not_make_wrong_primitive_type_valid(self):
        self.assertFalse(self.fixture("integer").valid)

    def test_compiler_acceptance_does_not_prove_execution_readiness(self):
        self.assertFalse(self.fixture().executable_ready)

    def test_unknown_type_is_not_verified_compatibility(self):
        self.assertFalse(self.fixture("any").executable_ready)

    def test_resource_observation_needs_endpoint_and_field_scope(self):
        observed = ResourceObservation(observation_id="calendar_options", resource_ids=("calendar_17",))
        source = ResourceReference(kind="resource_ref", observation_id="calendar_options", resource_id="calendar_17")
        self.assertFalse(self.fixture(source=source, observations=(observed,)).valid)


if __name__ == "__main__":
    unittest.main()

from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "src"))
from farm_arch.configuration import ConfigurationSession, Decision, Endpoint, Field, Ingredient, Observation, Receipt


def span(text="green", transform="identity"):
    return {"kind": "query_span", "start": 0, "end": len(text), "text": text, "transform": transform}


class ConfigurationTests(unittest.TestCase):
    def session(self, value_type="string", *, required=True, observations=(), field=None, query="green"):
        return ConfigurationSession(query=query,
            trigger=Endpoint("trigger", "trigger", "v1", ingredients=(Ingredient("amount", "integer"),)),
            action=Endpoint("action", "action", "v1", fields=(field or Field("target", required, value_type),)),
            observations=observations)

    def compile(self, session=None, source=None, **kwargs):
        return (session or self.session()).compile((Decision("action", "target", source or span()),), **kwargs)

    def test_wrong_primitive_query_span_is_rejected(self):
        result = self.compile(self.session("integer"))
        self.assertFalse(result.locally_valid)
        self.assertEqual(result.issues[0].code, "type_mismatch")

    def test_local_acceptance_is_not_execution(self):
        result = self.compile()
        self.assertTrue(result.locally_valid)
        self.assertEqual(result.status, "locally_checked")
        self.assertFalse(result.execution_verified)
        self.assertFalse(result.dry_run_verified)

    def test_unknown_type_and_requiredness_need_evidence(self):
        for session in (self.session(None), self.session("any"), self.session(required=None)):
            self.assertEqual(self.compile(session).status, "needs_evidence")

    def test_resource_must_be_scoped_to_endpoint_field_and_revision(self):
        good = Observation("obs", "action", "target", "v1", ("r17",))
        source = {"kind": "resource_ref", "observation_id": "obs", "resource_id": "r17"}
        self.assertTrue(self.compile(self.session(observations=(good,)), source).locally_valid)
        for bad in (replace(good, endpoint_id="elsewhere"), replace(good, field_slug="other"), replace(good, schema_revision="v0")):
            result = self.compile(self.session(observations=(bad,)), source)
            self.assertEqual(result.issues[0].code, "resource_scope_mismatch")

    def test_no_observation_cannot_invent_resource(self):
        result = self.compile(source={"kind": "resource_ref", "observation_id": "missing", "resource_id": "r17"})
        self.assertEqual(result.issues[0].code, "unobserved_resource")

    def test_explicit_integer_transformation_records_typed_value(self):
        result = self.compile(self.session("integer", query="15"), span("15", "parse_integer"))
        self.assertTrue(result.locally_valid)
        self.assertEqual(result.resolved[0]["value"], 15)

    def test_percentage_conversion_is_not_invented(self):
        result = self.compile(self.session("number", query="5%"), span("5%", "parse_number"))
        self.assertFalse(result.locally_valid)

    def test_fabricated_span_is_rejected(self):
        result = self.compile(source=span("blue"))
        self.assertEqual(result.issues[0].code, "ungrounded_query_span")

    def test_trigger_output_requires_declared_bindable_action_field(self):
        source = {"kind": "trigger_output", "ingredient_slug": "amount"}
        result = self.compile(self.session(field=Field("target", True, "number", bindable=True)), source)
        self.assertTrue(result.locally_valid)
        self.assertEqual(result.resolved[0]["value"], {"trigger_ingredient": "amount"})
        self.assertFalse(self.compile(self.session("integer"), source).locally_valid)

    def test_trigger_cannot_bind_to_its_future_output(self):
        session = ConfigurationSession(query="q", trigger=Endpoint("t", "trigger", "v1",
            fields=(Field("f", True, "integer", bindable=True),), ingredients=(Ingredient("x", "integer"),)),
            action=Endpoint("a", "action", "v1"))
        result = session.compile((Decision("trigger", "f", {"kind": "trigger_output", "ingredient_slug": "x"}),))
        self.assertEqual(result.issues[0].code, "future_trigger_output_unavailable")

    def test_optional_omission_and_required_omission_are_distinct(self):
        self.assertTrue(self.compile(self.session(required=False), {"kind": "omit"}).locally_valid)
        self.assertFalse(self.compile(source={"kind": "omit"}).locally_valid)

    def test_receipt_is_bound_to_exact_context_and_draft(self):
        current = self.compile()
        receipt = Receipt("execution", current.state_sha256, True)
        verified = self.compile(receipts=(receipt,))
        self.assertTrue(verified.execution_verified)
        # A previous receipt must not certify a changed request or schema.
        changed = self.compile(self.session(query="blue"), span("blue"), receipts=(receipt,))
        self.assertFalse(changed.execution_verified)
        self.assertEqual(changed.issues[-1].code, "stale_receipt")

    def test_failed_or_conflicting_receipt_does_not_prove_execution(self):
        current = self.compile()
        receipts = (Receipt("execution", current.state_sha256, True), Receipt("execution", current.state_sha256, False))
        self.assertFalse(self.compile(receipts=receipts).execution_verified)

    def test_new_session_has_no_previous_resource_state(self):
        source = {"kind": "resource_ref", "observation_id": "obs", "resource_id": "r17"}
        first = self.session(observations=(Observation("obs", "action", "target", "v1", ("r17",)),))
        self.assertTrue(self.compile(first, source).locally_valid)
        self.assertFalse(self.compile(self.session(), source).locally_valid)

    def test_credentials_cannot_be_taken_from_query_spans(self):
        result = self.compile(self.session(field=Field("target", True, "string", auth_like=True)))
        self.assertEqual(result.issues[0].code, "credential_must_use_secret_ref")

    def test_malformed_model_source_fails_closed_without_crashing(self):
        for source in ([], {"kind": []}, {"kind": "resource_ref", "observation_id": [], "resource_id": "x"}):
            with self.subTest(source=source):
                result = self.session().compile((Decision("action", "target", source),))
                self.assertEqual(result.status, "invalid")
                self.assertEqual(result.issues[0].code, "invalid_source_shape")

    def test_failed_execution_cannot_leave_an_overall_verified_flag(self):
        current = self.compile()
        result = self.compile(receipts=(Receipt("backend_validation", current.state_sha256, True),
            Receipt("dry_run", current.state_sha256, True), Receipt("execution", current.state_sha256, False)))
        self.assertFalse(result.backend_validation_verified)
        self.assertFalse(result.dry_run_verified)
        self.assertFalse(result.execution_verified)
        # The local checks and environment outcome are different measurements.
        self.assertTrue(result.locally_valid)
        self.assertEqual(result.status, "invalid")

    def test_invalid_receipt_shape_cannot_crash(self):
        result = self.compile(receipts=(Receipt([], "bad", True),))
        self.assertEqual(result.issues[-1].code, "invalid_receipt")


if __name__ == "__main__":
    unittest.main()

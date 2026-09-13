"""Fail-closed structural compiler for a selected FARM applet draft.

This compiler establishes grounding and schema validity only. It deliberately
does not claim semantic field-value correctness in the absence of gold values.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from farm_r9.artifact_io import assert_no_secrets
from farm_r9.contracts import (
    AppletDraft,
    CompilationResult,
    EndpointCandidate,
    FieldDecision,
    FieldSpec,
    NeedsInput,
    OmitField,
    QueryLiteral,
    ResolvedField,
    ResourceObservation,
    ResourceReference,
    SecretReference,
    TriggerOutput,
    ValidationIssue,
)


def _type_compatible(source: str, target: str) -> bool:
    source_norm, target_norm = source.casefold().strip(), target.casefold().strip()
    if "any" in {source_norm, target_norm} or not source_norm or not target_norm:
        return True
    aliases = {
        "str": "string", "text": "string", "url": "string",
        "int": "integer", "number": "float", "double": "float",
        "bool": "boolean",
    }
    return aliases.get(source_norm, source_norm) == aliases.get(target_norm, target_norm)


class AppletCompiler:
    def __init__(
        self,
        *,
        query: str,
        trigger_candidates: Iterable[EndpointCandidate],
        action_candidates: Iterable[EndpointCandidate],
        resource_observations: Iterable[ResourceObservation] = (),
        available_secret_refs: Iterable[str] = (),
    ) -> None:
        self.query = query
        self.triggers = {candidate.alias: candidate for candidate in trigger_candidates}
        self.actions = {candidate.alias: candidate for candidate in action_candidates}
        if any(candidate.side != "trigger" for candidate in self.triggers.values()):
            raise ValueError("trigger candidate set contains a non-trigger")
        if any(candidate.side != "action" for candidate in self.actions.values()):
            raise ValueError("action candidate set contains a non-action")
        self.observations = {item.observation_id: set(item.resource_ids) for item in resource_observations}
        self.secret_refs = set(available_secret_refs)

    def compile(self, draft: AppletDraft) -> CompilationResult:
        # Scanner allows aliases but rejects credential-shaped strings anywhere.
        try:
            assert_no_secrets(draft.model_dump(mode="json"))
        except ValueError as error:
            return self._failure("secret_material", "security", str(error), repairable=False)

        trigger = self.triggers.get(draft.selection.trigger_alias)
        action = self.actions.get(draft.selection.action_alias)
        issues: list[ValidationIssue] = []
        if trigger is None:
            issues.append(self._issue("unknown_trigger_alias", "selection", None, "selected trigger alias is unavailable"))
        if action is None:
            issues.append(self._issue("unknown_action_alias", "selection", None, "selected action alias is unavailable"))
        if trigger is None or action is None:
            return CompilationResult(valid=False, structurally_complete=False, executable_ready=False, issues=tuple(issues))

        trigger_values, trigger_issues = self._compile_fields(
            side="trigger", specs=trigger.fields, decisions=draft.trigger_fields,
            trigger=trigger,
        )
        action_values, action_issues = self._compile_fields(
            side="action", specs=action.fields, decisions=draft.action_fields,
            trigger=trigger,
        )
        issues.extend(trigger_issues)
        issues.extend(action_issues)
        structurally_complete = not any(issue.code in {
            "unknown_field", "missing_field_decision", "duplicate_field_decision", "required_omitted"
        } for issue in issues)
        valid = not issues
        needs_input = any(value.source_kind == "needs_input" for value in (*trigger_values, *action_values))
        return CompilationResult(
            valid=valid,
            structurally_complete=structurally_complete,
            executable_ready=valid and not needs_input,
            issues=tuple(issues), trigger_values=tuple(trigger_values), action_values=tuple(action_values),
        )

    def _compile_fields(
        self,
        *,
        side: str,
        specs: tuple[FieldSpec, ...],
        decisions: tuple[FieldDecision, ...],
        trigger: EndpointCandidate,
    ) -> tuple[list[ResolvedField], list[ValidationIssue]]:
        spec_map = {spec.slug: spec for spec in specs}
        counts = Counter(decision.field_slug for decision in decisions)
        decision_map = {decision.field_slug: decision for decision in decisions}
        values: list[ResolvedField] = []
        issues: list[ValidationIssue] = []
        for slug, count in counts.items():
            if count > 1:
                issues.append(self._issue("duplicate_field_decision", side, slug, "field is configured more than once"))
            if slug not in spec_map:
                issues.append(self._issue("unknown_field", side, slug, "field is not present in the selected schema"))
        for slug, spec in spec_map.items():
            decision = decision_map.get(slug)
            if decision is None:
                issues.append(self._issue("missing_field_decision", side, slug, "every required and optional field needs an explicit decision"))
                continue
            source = decision.source
            if isinstance(source, OmitField):
                if spec.required:
                    issues.append(self._issue("required_omitted", side, slug, "required field cannot be omitted"))
                else:
                    values.append(ResolvedField(field_slug=slug, source_kind="omit", value=None, grounded=True))
                continue
            if isinstance(source, NeedsInput):
                values.append(ResolvedField(field_slug=slug, source_kind="needs_input", value=source.question, grounded=True))
                continue
            if isinstance(source, QueryLiteral):
                if source.end > len(self.query) or self.query[source.start:source.end] != source.text:
                    issues.append(self._issue("ungrounded_query_literal", side, slug, "literal must exactly equal the cited query span"))
                else:
                    values.append(ResolvedField(field_slug=slug, source_kind=source.kind, value=source.text, grounded=True))
                continue
            if isinstance(source, TriggerOutput):
                if side != "action":
                    issues.append(self._issue("trigger_self_binding", side, slug, "trigger configuration cannot use its future output"))
                    continue
                ingredient = next((item for item in trigger.ingredients if item.slug == source.ingredient_slug), None)
                if ingredient is None:
                    issues.append(self._issue("unknown_ingredient", side, slug, "binding does not name an ingredient from the selected trigger"))
                elif not spec.bindable:
                    issues.append(self._issue("non_bindable_field", side, slug, "selected action field does not accept trigger ingredients"))
                elif not _type_compatible(ingredient.value_type, spec.value_type):
                    issues.append(self._issue("type_mismatch", side, slug, "ingredient type is incompatible with action field type"))
                else:
                    values.append(ResolvedField(field_slug=slug, source_kind=source.kind, value=source.ingredient_slug, grounded=True))
                continue
            if isinstance(source, ResourceReference):
                observed = self.observations.get(source.observation_id)
                if observed is None or source.resource_id not in observed:
                    issues.append(self._issue("unobserved_resource", side, slug, "resource must come from a recorded tool observation"))
                else:
                    values.append(ResolvedField(field_slug=slug, source_kind=source.kind, value=source.resource_id, grounded=True))
                continue
            if isinstance(source, SecretReference):
                if source.reference not in self.secret_refs:
                    issues.append(self._issue("unavailable_secret_ref", side, slug, "secret alias is not available in the secure registry"))
                elif not spec.auth_like:
                    issues.append(self._issue("secret_in_non_auth_field", side, slug, "secret reference cannot populate a normal field"))
                else:
                    values.append(ResolvedField(field_slug=slug, source_kind=source.kind, value=source.reference, grounded=True))
                continue
            issues.append(self._issue("unsupported_source", side, slug, "unsupported field source", repairable=False))
        return values, issues

    @staticmethod
    def _issue(code: str, side: str, slug: str | None, message: str, repairable: bool = True) -> ValidationIssue:
        return ValidationIssue(code=code, side=side, field_slug=slug, message=message, repairable=repairable)

    @classmethod
    def _failure(cls, code: str, side: str, message: str, *, repairable: bool) -> CompilationResult:
        return CompilationResult(
            valid=False, structurally_complete=False, executable_ready=False,
            issues=(cls._issue(code, side, None, message, repairable),),
        )

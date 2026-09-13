"""Strict Round 9 contracts for endpoint choice and applet configuration."""
from __future__ import annotations

import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class FieldSpec(FrozenModel):
    slug: str = Field(min_length=1)
    label: str = Field(min_length=1)
    required: bool
    bindable: bool = False
    value_type: str = "any"
    resource_like: bool = False
    auth_like: bool = False


class IngredientSpec(FrozenModel):
    slug: str = Field(min_length=1)
    label: str = Field(min_length=1)
    value_type: str = "any"


class EndpointCandidate(FrozenModel):
    alias: str = Field(pattern=r"^[TA][0-9]{2}$")
    side: Literal["trigger", "action"]
    service: str = Field(min_length=1)
    service_id: str | None = None
    function: str = Field(min_length=1)
    description: str = ""
    fields: tuple[FieldSpec, ...] = ()
    ingredients: tuple[IngredientSpec, ...] = ()

    @field_validator("fields", "ingredients", mode="before")
    @classmethod
    def json_arrays_to_tuples(cls, value: Any) -> Any:
        # JSON has arrays but no tuple primitive. Convert only the container;
        # strict validation still applies to every nested field and scalar.
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_schema_ids(self) -> "EndpointCandidate":
        if len({field.slug for field in self.fields}) != len(self.fields):
            raise ValueError("endpoint field slugs must be unique")
        if len({ingredient.slug for ingredient in self.ingredients}) != len(self.ingredients):
            raise ValueError("endpoint ingredient slugs must be unique")
        return self


class QueryLiteral(FrozenModel):
    kind: Literal["query_literal"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_span(self) -> "QueryLiteral":
        if self.end <= self.start:
            raise ValueError("literal end must be greater than start")
        return self


class TriggerOutput(FrozenModel):
    kind: Literal["trigger_output"]
    ingredient_slug: str = Field(min_length=1)


class NeedsInput(FrozenModel):
    kind: Literal["needs_input"]
    question: str = Field(min_length=3, max_length=240)


class OmitField(FrozenModel):
    kind: Literal["omit"]
    reason: str = Field(min_length=2, max_length=160)


class ResourceReference(FrozenModel):
    kind: Literal["resource_ref"]
    resource_id: str = Field(min_length=1, max_length=160)
    observation_id: str = Field(min_length=1, max_length=160)


class SecretReference(FrozenModel):
    kind: Literal["secret_ref"]
    reference: str = Field(min_length=1, max_length=128)

    @field_validator("reference")
    @classmethod
    def opaque_alias_only(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", value):
            raise ValueError("secret reference must be an opaque alias")
        if re.search(r"(?:bearer|api.?key|password|token=)", value, re.IGNORECASE):
            raise ValueError("secret reference appears to contain credential material")
        return value


FieldSource = Annotated[
    QueryLiteral | TriggerOutput | NeedsInput | OmitField | ResourceReference | SecretReference,
    Field(discriminator="kind"),
]


class FieldDecision(FrozenModel):
    field_slug: str = Field(min_length=1)
    source: FieldSource


class EndpointSelection(FrozenModel):
    trigger_alias: str = Field(pattern=r"^T[0-9]{2}$")
    action_alias: str = Field(pattern=r"^A[0-9]{2}$")


class EndpointPrediction(FrozenModel):
    trigger_alias: str = Field(pattern=r"^T[0-9]{2}$")
    action_alias: str = Field(pattern=r"^A[0-9]{2}$")
    trigger_field_names: tuple[str, ...] = ()
    action_field_names: tuple[str, ...] = ()
    preview: str = Field(min_length=1, max_length=800)
    evidence_aliases: tuple[str, ...] = ()

    @field_validator("trigger_field_names", "action_field_names", "evidence_aliases", mode="before")
    @classmethod
    def json_arrays_to_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_fields_and_evidence(self) -> "EndpointPrediction":
        if len(self.trigger_field_names) != len(set(self.trigger_field_names)):
            raise ValueError("trigger field names must be unique")
        if len(self.action_field_names) != len(set(self.action_field_names)):
            raise ValueError("action field names must be unique")
        if len(self.evidence_aliases) != len(set(self.evidence_aliases)):
            raise ValueError("evidence aliases must be unique")
        return self


class AppletDraft(FrozenModel):
    selection: EndpointSelection
    trigger_fields: tuple[FieldDecision, ...] = ()
    action_fields: tuple[FieldDecision, ...] = ()
    preview: str = Field(min_length=1, max_length=1200)
    evidence_aliases: tuple[str, ...] = ()

    @field_validator("trigger_fields", "action_fields", "evidence_aliases", mode="before")
    @classmethod
    def json_arrays_to_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def unique_decisions(self) -> "AppletDraft":
        for side, decisions in (("trigger", self.trigger_fields), ("action", self.action_fields)):
            slugs = [decision.field_slug for decision in decisions]
            if len(slugs) != len(set(slugs)):
                raise ValueError(f"{side} field decisions must be unique")
        return self


class ValidationIssue(FrozenModel):
    code: str = Field(min_length=1)
    side: Literal["selection", "trigger", "action", "security"]
    field_slug: str | None = None
    message: str = Field(min_length=1)
    repairable: bool = True


class ResolvedField(FrozenModel):
    field_slug: str
    source_kind: str
    value: Any = None
    grounded: bool


class CompilationResult(FrozenModel):
    valid: bool
    structurally_complete: bool
    executable_ready: bool
    issues: tuple[ValidationIssue, ...]
    trigger_values: tuple[ResolvedField, ...] = ()
    action_values: tuple[ResolvedField, ...] = ()

    @field_validator("issues", "trigger_values", "action_values", mode="before")
    @classmethod
    def json_arrays_to_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class ResourceObservation(FrozenModel):
    observation_id: str = Field(min_length=1)
    resource_ids: tuple[str, ...]

    @field_validator("resource_ids", mode="before")
    @classmethod
    def json_arrays_to_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class AgentAsk(FrozenModel):
    action: Literal["ask"]
    component: str = Field(min_length=1)
    question: str = Field(min_length=3, max_length=240)


class AgentCommit(FrozenModel):
    action: Literal["commit"]
    draft: AppletDraft


AgentAction = Annotated[AgentAsk | AgentCommit, Field(discriminator="action")]

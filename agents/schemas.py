"""
JSON Schemas for FARM Agentic AI System
=======================================

Defines the structured output formats for agent responses and final applets.
Uses Pydantic for validation and JSON schema generation.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class Ingredient(BaseModel):
    """
    Represents a trigger ingredient (data provided by trigger).

    Ingredients are the "OFFER" from the Trigger Agent - data fields
    that become available when the trigger fires.
    """
    name: str = Field(description="Ingredient name (e.g., 'EntryTitle')")
    slug: str = Field(description="Ingredient slug identifier")
    type: str = Field(description="Data type (e.g., 'String', 'Number')")
    example: Optional[str] = Field(default=None, description="Example value")


class ActionField(BaseModel):
    """
    Represents an action field (data required by action).

    Action fields are the "REQUIREMENTS" from the Action Agent - data
    that must be provided for the action to execute.
    """
    name: str = Field(description="Field name (e.g., 'Message')")
    label: str = Field(description="Human-readable label")
    slug: str = Field(description="Field slug identifier")
    required: bool = Field(description="Whether this field is mandatory")


class Binding(BaseModel):
    """
    Represents a binding between a trigger ingredient and an action field.

    Bindings map the OFFER (ingredients) to REQUIREMENTS (fields) to
    create an executable applet configuration.
    """
    action_field: str = Field(description="Target action field name")
    source_type: Literal["ingredient", "static", "user_input"] = Field(
        description="Source of the value"
    )
    ingredient_name: Optional[str] = Field(
        default=None,
        description="Source ingredient name (if source_type='ingredient')"
    )
    static_value: Optional[str] = Field(
        default=None,
        description="Static value (if source_type='static')"
    )
    reasoning: str = Field(description="Why this binding makes sense")


class TriggerAgentResponse(BaseModel):
    """
    Structured response from the Trigger Agent.

    Contains reasoning about why the trigger fits and the OFFER
    (list of available ingredients).
    """
    reasoning: str = Field(
        description="Explanation of why this trigger matches the user query"
    )
    service_name: str = Field(description="Selected trigger service name")
    category: str = Field(description="Trigger category")
    ingredients: List[Ingredient] = Field(
        description="Available ingredients (the OFFER)"
    )


class ActionAgentResponse(BaseModel):
    """
    Structured response from the Action Agent.

    Contains reasoning, REQUIREMENTS (fields), compatibility decision,
    and binding map if accepted.
    """
    reasoning: str = Field(
        description="Explanation of why this action matches the user query"
    )
    service_name: str = Field(description="Selected action service name")
    category: str = Field(description="Action category")
    required_fields: List[ActionField] = Field(
        description="Required fields (the REQUIREMENTS)"
    )
    decision: Literal["ACCEPT", "REJECT"] = Field(
        description="Whether the OFFER satisfies REQUIREMENTS"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Reason for rejection (if decision='REJECT')"
    )
    bindings: Optional[List[Binding]] = Field(
        default=None,
        description="Field bindings (if decision='ACCEPT')"
    )


class VerifierResponse(BaseModel):
    """
    Structured response from the Contract Verifier.

    Scores the complete applet and provides critique.
    """
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Applet quality score (0.0 to 1.0)"
    )
    critique: str = Field(
        description="Detailed critique of the applet"
    )
    is_executable: bool = Field(
        description="Whether the applet can be executed as-is"
    )
    missing_fields: List[str] = Field(
        default_factory=list,
        description="List of fields without valid bindings"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Improvement suggestions"
    )


class ExecutableApplet(BaseModel):
    """
    Final executable applet structure.

    This is the complete output of the resolution process - a fully
    configured trigger-action program ready for execution.
    """
    query: str = Field(description="Original user query")

    trigger: Dict = Field(description="Trigger configuration")
    action: Dict = Field(description="Action configuration")
    bindings: List[Binding] = Field(description="Field bindings")

    verifier_score: Optional[float] = Field(
        default=None,
        description="Quality score from Contract Verifier"
    )
    verifier_critique: Optional[str] = Field(
        default=None,
        description="Critique from Contract Verifier"
    )

    resolution_rounds: int = Field(
        default=1,
        description="Number of resolution rounds until success"
    )

    def to_ifttt_format(self) -> Dict:
        """
        Convert to IFTTT-compatible JSON format.

        Returns:
            Dictionary in IFTTT applet format.
        """
        return {
            "trigger": {
                "service_name": self.trigger.get("service_name"),
                "category": self.trigger.get("category"),
                "ingredients_used": [
                    b.ingredient_name for b in self.bindings
                    if b.source_type == "ingredient"
                ]
            },
            "action": {
                "service_name": self.action.get("service_name"),
                "category": self.action.get("category"),
                "field_values": {
                    b.action_field: (
                        f"{{{{{'{'}{b.ingredient_name}{'}'}}}}}"
                        if b.source_type == "ingredient"
                        else b.static_value
                    )
                    for b in self.bindings
                }
            },
            "metadata": {
                "query": self.query,
                "score": self.verifier_score,
                "rounds": self.resolution_rounds
            }
        }

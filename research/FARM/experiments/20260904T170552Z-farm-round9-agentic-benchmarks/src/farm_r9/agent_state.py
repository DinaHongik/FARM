"""Bounded, case-local clarification and repair state."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from farm_r9.contracts import AppletDraft, CompilationResult


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    component: str
    answer: str
    source: str


class AgentState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, arbitrary_types_allowed=False)
    case_id: str = Field(min_length=1)
    max_questions: int = Field(default=2, ge=0, le=4)
    max_repairs: int = Field(default=1, ge=0, le=1)
    asked_components: set[str] = Field(default_factory=set)
    observations: list[Observation] = Field(default_factory=list)
    question_count: int = 0
    repair_count: int = 0
    semantic_call_count: int = 0
    last_valid_draft: AppletDraft | None = None
    last_validation: CompilationResult | None = None

    def begin_call(self) -> None:
        self.semantic_call_count += 1

    def ask(self, component: str) -> None:
        if self.question_count >= self.max_questions:
            raise ValueError("question budget exhausted")
        if component in self.asked_components:
            raise ValueError("repeated clarification is forbidden")
        self.asked_components.add(component)
        self.question_count += 1

    def observe(self, *, component: str, answer: str, source: str) -> None:
        if component not in self.asked_components:
            raise ValueError("cannot observe an answer for a component that was not asked")
        if any(item.component == component for item in self.observations):
            raise ValueError("component answer already observed")
        self.observations.append(Observation(component=component, answer=answer, source=source))

    def record_validation(self, *, draft: AppletDraft, result: CompilationResult) -> None:
        self.last_validation = result
        if result.valid:
            self.last_valid_draft = draft

    def begin_repair(self) -> tuple[dict[str, Any], ...]:
        if self.repair_count >= self.max_repairs:
            raise ValueError("repair budget exhausted")
        if self.last_validation is None or self.last_validation.valid:
            raise ValueError("repair requires actual validation errors")
        if not any(issue.repairable for issue in self.last_validation.issues):
            raise ValueError("validation has no repairable issue")
        self.repair_count += 1
        return tuple(issue.model_dump(mode="json") for issue in self.last_validation.issues if issue.repairable)


def fresh_state(case_id: str, *, max_questions: int = 2, max_repairs: int = 1) -> AgentState:
    return AgentState(case_id=case_id, max_questions=max_questions, max_repairs=max_repairs)

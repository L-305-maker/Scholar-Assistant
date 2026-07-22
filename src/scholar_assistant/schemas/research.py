from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class ProjectStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ResearchProject(BaseModel):
    project_id: str = Field(default_factory=lambda: f"proj_{uuid4().hex[:12]}")
    name: str
    user_goal: str
    research_questions: list[str] = Field(default_factory=list)
    scope: str | None = None
    status: ProjectStatus = ProjectStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class QueryPlan(BaseModel):
    user_question: str
    core_query: str
    english_terms: list[str] = Field(default_factory=list)
    synonym_queries: list[str] = Field(default_factory=list)
    method_queries: list[str] = Field(default_factory=list)
    problem_queries: list[str] = Field(default_factory=list)
    negative_result_queries: list[str] = Field(default_factory=list)
    verification_queries: list[str] = Field(default_factory=list)

    def all_queries(self) -> list[str]:
        seen: set[str] = set()
        queries: list[str] = []
        for query in [
            self.core_query,
            *self.synonym_queries,
            *self.method_queries,
            *self.problem_queries,
            *self.negative_result_queries,
            *self.verification_queries,
        ]:
            normalized = " ".join(query.split()).strip()
            if normalized and normalized.lower() not in seen:
                seen.add(normalized.lower())
                queries.append(normalized)
        return queries

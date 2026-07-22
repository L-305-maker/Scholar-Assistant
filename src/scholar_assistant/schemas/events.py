from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RunEventType(StrEnum):
    RUN_STARTED = "run.started"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    SEARCH_QUERY = "search.query"
    PAPER_DISCOVERED = "paper.discovered"
    PAPER_SELECTED = "paper.selected"
    PAPER_READ = "paper.read"
    EVIDENCE_CREATED = "evidence.created"
    CLAIM_CREATED = "claim.created"
    HYPOTHESIS_CREATED = "hypothesis.created"
    WARNING = "warning"
    ERROR = "error"
    RUN_COMPLETED = "run.completed"


class RunEvent(BaseModel):
    run_id: str
    event_type: RunEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    tool: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def new(
        cls,
        event_type: RunEventType,
        run_id: str | None = None,
        **kwargs: Any,
    ) -> RunEvent:
        return cls(run_id=run_id or str(uuid4()), event_type=event_type, **kwargs)

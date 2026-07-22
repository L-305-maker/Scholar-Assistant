from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    SEARCH = "search"
    READ = "read"
    ANALYZE = "analyze"
    VERIFY = "verify"
    REPORT = "report"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:12]}")
    task_type: TaskType
    assigned_role: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING

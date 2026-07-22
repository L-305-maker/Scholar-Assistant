from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from scholar_assistant.core.budget import BudgetManager


class ToolPermissionLevel(StrEnum):
    T0_LOCAL_READ = "T0"
    T1_EXTERNAL_READ = "T1"
    T2_PROJECT_WRITE = "T2"
    T3_ISOLATED_EXECUTION = "T3"
    T4_HIGH_RISK = "T4"


class ToolPermissionError(ValueError):
    pass


class ToolExecutionError(RuntimeError):
    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(message)
        self.tool_name = tool_name


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 0
    backoff_seconds: float = 0.25


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    output_schema: dict[str, Any] = field(default_factory=dict)
    version: str = "1"
    category: str = "general"
    permission_level: ToolPermissionLevel = ToolPermissionLevel.T0_LOCAL_READ
    timeout_seconds: float = 30.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    cost_unit: str = "call"
    idempotent: bool = True
    network_domains: list[str] = field(default_factory=list)
    side_effect_level: str = "none"
    write_allowed: bool = False


@dataclass(slots=True)
class ToolExecutionContext:
    run_id: str | None = None
    budget: BudgetManager | None = None
    allowed_permissions: set[ToolPermissionLevel] = field(
        default_factory=lambda: {
            ToolPermissionLevel.T0_LOCAL_READ,
            ToolPermissionLevel.T1_EXTERNAL_READ,
            ToolPermissionLevel.T2_PROJECT_WRITE,
        }
    )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            msg = f"Tool already registered: {spec.name}"
            raise ValueError(msg)
        self._tools[spec.name] = spec

    def list(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            msg = f"Unknown tool: {name}"
            raise ToolPermissionError(msg)
        return self._tools[name]

    def ensure_allowed(self, name: str, *, write: bool = False) -> ToolSpec:
        spec = self.get(name)
        if write and not spec.write_allowed:
            msg = f"Tool {name} is read-only"
            raise ToolPermissionError(msg)
        return spec


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, context: ToolExecutionContext | None = None) -> None:
        self.registry = registry
        self.context = context or ToolExecutionContext()
        self.executions: list[dict[str, Any]] = []

    async def execute(self, name: str, **kwargs: Any) -> Any:
        spec = self.registry.get(name)
        if spec.permission_level not in self.context.allowed_permissions:
            msg = f"Tool {name} requires permission {spec.permission_level}"
            raise ToolPermissionError(msg)
        if self.context.budget is not None:
            self.context.budget.check_tool(spec, kwargs)

        started = datetime.now(UTC)
        start_time = time.perf_counter()
        attempts = 0
        execution_id = f"tex_{uuid4().hex[:12]}"
        last_error: BaseException | None = None
        for attempt in range(spec.retry_policy.max_retries + 1):
            attempts = attempt + 1
            try:
                result = await asyncio.wait_for(
                    _maybe_await(spec.handler(**kwargs)),
                    timeout=spec.timeout_seconds,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                if self.context.budget is not None:
                    self.context.budget.record_tool(spec, kwargs, result)
                self.executions.append(
                    _execution_record(
                        execution_id=execution_id,
                        run_id=self.context.run_id,
                        spec=spec,
                        status="completed",
                        started=started,
                        latency_ms=latency_ms,
                        attempts=attempts,
                    )
                )
                return result
            except Exception as exc:
                last_error = exc
                if self.context.budget is not None:
                    self.context.budget.record_retry(spec.name)
                if attempt >= spec.retry_policy.max_retries:
                    break
                await asyncio.sleep(spec.retry_policy.backoff_seconds * (2**attempt))

        latency_ms = (time.perf_counter() - start_time) * 1000
        error_type = type(last_error).__name__ if last_error else "UnknownError"
        self.executions.append(
            _execution_record(
                execution_id=execution_id,
                run_id=self.context.run_id,
                spec=spec,
                status="failed",
                started=started,
                latency_ms=latency_ms,
                attempts=attempts,
                error_type=error_type,
            )
        )
        raise ToolExecutionError(spec.name, f"{spec.name} failed: {error_type}") from last_error


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _execution_record(
    *,
    execution_id: str,
    run_id: str | None,
    spec: ToolSpec,
    status: str,
    started: datetime,
    latency_ms: float,
    attempts: int,
    error_type: str | None = None,
) -> dict[str, Any]:
    return {
        "execution_id": execution_id,
        "run_id": run_id,
        "tool_name": spec.name,
        "tool_version": spec.version,
        "status": status,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "latency_ms": latency_ms,
        "attempts": attempts,
        "error_type": error_type,
        "permission_level": spec.permission_level.value,
        "category": spec.category,
        "cost_unit": spec.cost_unit,
    }

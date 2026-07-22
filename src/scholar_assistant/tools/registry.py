from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolPermissionError(ValueError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    write_allowed: bool = False


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
        return self._tools[name]

    def ensure_allowed(self, name: str, *, write: bool = False) -> ToolSpec:
        spec = self.get(name)
        if write and not spec.write_allowed:
            msg = f"Tool {name} is read-only"
            raise ToolPermissionError(msg)
        return spec

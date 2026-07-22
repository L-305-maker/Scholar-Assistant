from __future__ import annotations

from pathlib import Path


class PermissionError(ValueError):
    pass


def ensure_project_relative(project_path: Path, target: Path) -> Path:
    project = project_path.resolve()
    resolved = target.expanduser().resolve()
    try:
        resolved.relative_to(project)
    except ValueError as exc:
        msg = f"Path must stay inside project: {resolved}"
        raise PermissionError(msg) from exc
    return resolved

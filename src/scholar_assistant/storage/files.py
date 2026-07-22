from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scholar_assistant.core.config import default_config_text

PROJECT_DIRS = [
    ".scholar",
    ".scholar/runs",
    ".scholar/cache",
    ".scholar/index",
    ".scholar/logs",
    "papers",
    "notes",
    "reports",
]


def ensure_project_layout(project_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for rel in PROJECT_DIRS:
        path = project_path / rel
        path.mkdir(parents=True, exist_ok=True)
        paths[rel] = path
    config_path = project_path / ".scholar" / "config.toml"
    if not config_path.exists():
        config_path.write_text(default_config_text(), encoding="utf-8")
    return paths


def run_dir(project_path: Path, run_id: str) -> Path:
    path = project_path / ".scholar" / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")


def sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

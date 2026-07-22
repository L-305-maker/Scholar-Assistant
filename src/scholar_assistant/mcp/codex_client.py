from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class CodexClient(Protocol):
    def exec_json(self, prompt: str, *, cwd: Path) -> list[dict]: ...


@dataclass(slots=True)
class CodexCliAdapter:
    enabled: bool = False
    executable: str = "codex"

    def exec_json(self, prompt: str, *, cwd: Path) -> list[dict]:
        if not self.enabled:
            msg = "Codex integration is disabled; enable it explicitly in config before use"
            raise RuntimeError(msg)
        completed = subprocess.run(
            [self.executable, "exec", "--json", prompt],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            msg = (
                f"codex exec failed with exit code {completed.returncode}: {completed.stderr[:500]}"
            )
            raise RuntimeError(msg)
        return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]

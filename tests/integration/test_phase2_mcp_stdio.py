from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scholar_assistant.mcp.server import _safe_output_path, _safe_project_reference
from scholar_assistant.mcp.stdio_smoke import run_smoke


@pytest.mark.asyncio
async def test_mcp_stdio_initialize_list_and_consecutive_calls(tmp_path: Path) -> None:
    result = await run_smoke(tmp_path)
    assert result["tools_count"] == 8
    assert "scholar_get_claims" in result["tools"]
    assert result["status_call_serializable"] is True
    assert result["claims_call_serializable"] is True
    assert result["research_call_serializable"] is True
    assert (tmp_path / ".scholar" / "runs").exists()


def test_mcp_path_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.pdf"
    with pytest.raises(PermissionError):
        _safe_project_reference(tmp_path, str(outside))
    with pytest.raises(PermissionError):
        _safe_output_path(tmp_path, str(tmp_path.parent / "report.md"))


def test_mcp_stdio_smoke_script_stdout_is_json_only(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/mcp_stdio_smoke.py",
            "--project-path",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    payload = json.loads(result.stdout)
    assert payload["tools_count"] == 8
    assert "scholar_get_claims" in payload["tools"]
    assert "Processing request" not in result.stdout
    assert "Initialized Scholar project" not in result.stdout

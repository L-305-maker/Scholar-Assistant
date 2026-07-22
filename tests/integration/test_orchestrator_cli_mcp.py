from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from scholar_assistant.cli.app import app
from scholar_assistant.core.orchestrator import ResearchOrchestrator
from scholar_assistant.mcp.server import list_mcp_tools
from scholar_assistant.tools.arxiv import ArxivClient

runner = CliRunner()


def test_cli_help_init_providers_and_exec(tmp_path: Path) -> None:
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "Scholar-Assistant" in help_result.stdout

    init_result = runner.invoke(app, ["init", "--project-path", str(tmp_path)])
    assert init_result.exit_code == 0
    assert (tmp_path / ".scholar" / "state.db").exists()

    providers_result = runner.invoke(app, ["providers", "list", "--project-path", str(tmp_path)])
    assert providers_result.exit_code == 0
    assert "DEEPSEEK_API_KEY" in providers_result.stdout

    exec_result = runner.invoke(
        app,
        ["exec", "-", "--json", "--ephemeral", "--project-path", str(tmp_path)],
        input="搜索相关论文",
    )
    assert exec_result.exit_code == 0
    lines = [json.loads(line) for line in exec_result.stdout.splitlines()]
    assert lines[0]["event_type"] == "run.started"
    assert lines[-1]["event_type"] == "run.completed"
    assert not any((tmp_path / ".scholar" / "runs").glob("*"))


@pytest.mark.asyncio
async def test_orchestrator_offline_research_resume_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_search(self: ArxivClient, query: str, *, max_results: int = 25, start: int = 0):
        raise httpx.ConnectError("offline")

    monkeypatch.setenv("SCHOLAR_DEMO_MODE", "1")
    monkeypatch.setattr(ArxivClient, "search", fail_search)
    result = await ResearchOrchestrator(tmp_path).run_research(
        "调研 LLM Agent 长期记忆中的检索噪声问题",
        no_embeddings=True,
    )
    assert result.status.value == "COMPLETED"
    assert result.papers
    assert result.evidence_units
    assert (result.run_path / "report.md").exists()
    assert "[待验证假设]" in (result.run_path / "report.md").read_text(encoding="utf-8")
    event_types = [
        json.loads(line)["event_type"]
        for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "task.started" in event_types
    assert "task.completed" in event_types
    assert "warning" in event_types
    resumed = ResearchOrchestrator(tmp_path).get_run(result.run_id)
    assert resumed
    assert resumed["status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_orchestrator_budget_exhausted_saves_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHOLAR_DEMO_MODE", "1")
    result = await ResearchOrchestrator(tmp_path).run_research(
        "memory retrieval",
        no_embeddings=True,
        max_candidates=0,
    )
    assert result.status.value == "BUDGET_EXHAUSTED"
    assert (result.run_path / "run-manifest.json").exists()
    resumed = ResearchOrchestrator(tmp_path).get_run(result.run_id)
    assert resumed
    assert resumed["status"] == "BUDGET_EXHAUSTED"


def test_mcp_tool_registry_is_serializable() -> None:
    tools = list_mcp_tools()
    names = {tool["name"] for tool in tools}
    assert "scholar_search_papers" in names
    assert "scholar_run_research" in names
    json.dumps(tools)

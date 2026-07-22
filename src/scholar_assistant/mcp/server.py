from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from scholar_assistant.agents.reader import Reader
from scholar_assistant.agents.searcher import Searcher
from scholar_assistant.core.events import EventSink
from scholar_assistant.core.orchestrator import ResearchOrchestrator
from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout
from scholar_assistant.storage.repositories import ScholarRepository

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "scholar_search_papers",
        "description": "Search arXiv and local index for papers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string"},
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["project_path", "query"],
        },
    },
    {
        "name": "scholar_read_paper",
        "description": "Read a stored paper ID or project-local PDF and return evidence IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string"},
                "paper_id_or_pdf": {"type": "string"},
            },
            "required": ["project_path", "paper_id_or_pdf"],
        },
    },
    {
        "name": "scholar_compare_papers",
        "description": "Return metadata for selected stored paper IDs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string"},
                "paper_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["project_path", "paper_ids"],
        },
    },
    {
        "name": "scholar_find_evidence",
        "description": "Find evidence units for a stored work ID.",
        "inputSchema": {
            "type": "object",
            "properties": {"project_path": {"type": "string"}, "work_id": {"type": "string"}},
            "required": ["project_path", "work_id"],
        },
    },
    {
        "name": "scholar_get_claims",
        "description": "Return persisted claims.",
        "inputSchema": {
            "type": "object",
            "properties": {"project_path": {"type": "string"}},
            "required": ["project_path"],
        },
    },
    {
        "name": "scholar_run_research",
        "description": "Run the full research workflow and return artifact paths.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string"},
                "question": {"type": "string"},
                "no_embeddings": {"type": "boolean", "default": False},
            },
            "required": ["project_path", "question"],
        },
    },
    {
        "name": "scholar_get_run_status",
        "description": "Return run status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string"},
                "run_id": {"type": "string"},
            },
            "required": ["project_path", "run_id"],
        },
    },
    {
        "name": "scholar_export_report",
        "description": "Export a run report to a target path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_path": {"type": "string"},
                "run_id": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["project_path", "run_id", "output_path"],
        },
    },
]


def list_mcp_tools() -> list[dict[str, Any]]:
    return TOOL_SCHEMAS


def run_mcp_server() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        msg = "The `mcp` package is not installed. Run `uv sync` before starting the server."
        raise RuntimeError(msg) from exc

    mcp = FastMCP("Scholar Assistant")

    @mcp.tool()
    async def scholar_search_papers(
        project_path: str,
        query: str,
        max_results: int = 10,
    ) -> dict[str, Any]:
        project = Path(project_path).resolve()
        ensure_project_layout(project)
        run_id = "mcp_search"
        with Database(project / ".scholar" / "state.db") as connection:
            repository = ScholarRepository(connection)
            sink = EventSink(project / ".scholar" / "runs" / run_id / "events.jsonl")
            settings = ResearchOrchestrator(project).settings
            searcher = Searcher(repository, settings, project, sink, run_id=run_id)
            result = await searcher.search(
                query, max_results_per_query=max_results, no_embeddings=True
            )
        return {
            "retrieval_mode": result.retrieval_mode,
            "warnings": result.warnings,
            "papers": [paper.model_dump(mode="json") for paper in result.papers[:max_results]],
        }

    @mcp.tool()
    async def scholar_read_paper(project_path: str, paper_id_or_pdf: str) -> dict[str, Any]:
        project = Path(project_path).resolve()
        ensure_project_layout(project)
        run_id = "mcp_read"
        with Database(project / ".scholar" / "state.db") as connection:
            repository = ScholarRepository(connection)
            sink = EventSink(project / ".scholar" / "runs" / run_id / "events.jsonl")
            evidence = await Reader(repository, project, sink, run_id=run_id).read_paper(
                paper_id_or_pdf
            )
        return {"evidence": [item.model_dump(mode="json") for item in evidence]}

    @mcp.tool()
    async def scholar_compare_papers(project_path: str, paper_ids: list[str]) -> dict[str, Any]:
        project = Path(project_path).resolve()
        with Database(project / ".scholar" / "state.db") as connection:
            repository = ScholarRepository(connection)
            papers = [paper for paper_id in paper_ids if (paper := repository.get_paper(paper_id))]
        return {"papers": [paper.model_dump(mode="json") for paper in papers]}

    @mcp.tool()
    async def scholar_find_evidence(project_path: str, work_id: str) -> dict[str, Any]:
        project = Path(project_path).resolve()
        with Database(project / ".scholar" / "state.db") as connection:
            evidence = ScholarRepository(connection).list_evidence(work_id)
        return {"evidence": [item.model_dump(mode="json") for item in evidence]}

    @mcp.tool()
    async def scholar_get_claims(project_path: str) -> dict[str, Any]:
        project = Path(project_path).resolve()
        with Database(project / ".scholar" / "state.db") as connection:
            claims = ScholarRepository(connection).list_claims()
        return {"claims": [claim.model_dump(mode="json") for claim in claims]}

    @mcp.tool()
    async def scholar_run_research(
        project_path: str,
        question: str,
        no_embeddings: bool = False,
    ) -> dict[str, Any]:
        result = await ResearchOrchestrator(Path(project_path)).run_research(
            question,
            no_embeddings=no_embeddings,
        )
        return result.summary()

    @mcp.tool()
    async def scholar_get_run_status(project_path: str, run_id: str) -> dict[str, Any]:
        run = ResearchOrchestrator(Path(project_path)).get_run(run_id)
        return {"run": run}

    @mcp.tool()
    async def scholar_export_report(
        project_path: str, run_id: str, output_path: str
    ) -> dict[str, Any]:
        run = ResearchOrchestrator(Path(project_path)).get_run(run_id)
        if not run:
            return {"error": f"run not found: {run_id}"}
        source = Path(str(run["data"]["run_path"])) / "report.md"
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return {"output_path": str(target)}

    mcp.run()


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)

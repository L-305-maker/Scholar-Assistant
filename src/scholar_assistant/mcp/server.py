from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from scholar_assistant.agents.reader import Reader
from scholar_assistant.agents.searcher import Searcher
from scholar_assistant.core.budget import BudgetManager
from scholar_assistant.core.config import ScholarSettings
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
    _run_jsonrpc_stdio_server()


def _run_jsonrpc_stdio_server() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = _handle_jsonrpc_request(request)
        except Exception as exc:
            response = _jsonrpc_error(None, -32603, type(exc).__name__, str(exc))
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            sys.stdout.flush()


def _handle_jsonrpc_request(request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    if method == "initialize":
        return _jsonrpc_result(
            request_id,
            {
                "protocolVersion": params.get("protocolVersion", "2025-11-25"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "scholar-assistant", "version": "0.1.0"},
                "instructions": "Scholar-Assistant exposes project-scoped academic research tools.",
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return _jsonrpc_result(request_id, {"tools": list_mcp_tools()})
    if method == "tools/call":
        tool_name = str(params.get("name", ""))
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            return _jsonrpc_error(request_id, -32602, "Invalid arguments")
        result = asyncio.run(_dispatch_tool_call(tool_name, arguments))
        is_error = "error" in result
        return _jsonrpc_result(
            request_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, ensure_ascii=False, default=str),
                    }
                ],
                "structuredContent": result,
                "isError": is_error,
            },
        )
    return _jsonrpc_error(request_id, -32601, "Method not found", str(method))


def _jsonrpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(
    request_id: Any,
    code: int,
    message: str,
    data: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


async def _dispatch_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "scholar_search_papers":
            return await _tool_search_papers(
                str(arguments["project_path"]),
                str(arguments["query"]),
                int(arguments.get("max_results", 10)),
            )
        if name == "scholar_read_paper":
            return await _tool_read_paper(
                str(arguments["project_path"]),
                str(arguments["paper_id_or_pdf"]),
            )
        if name == "scholar_compare_papers":
            return await _tool_compare_papers(
                str(arguments["project_path"]),
                [str(paper_id) for paper_id in arguments.get("paper_ids", [])],
            )
        if name == "scholar_find_evidence":
            return await _tool_find_evidence(
                str(arguments["project_path"]),
                str(arguments["work_id"]),
            )
        if name == "scholar_get_claims":
            return await _tool_get_claims(str(arguments["project_path"]))
        if name == "scholar_run_research":
            return await _tool_run_research(
                str(arguments["project_path"]),
                str(arguments["question"]),
                bool(arguments.get("no_embeddings", False)),
            )
        if name == "scholar_get_run_status":
            return await _tool_get_run_status(
                str(arguments["project_path"]),
                str(arguments["run_id"]),
            )
        if name == "scholar_export_report":
            return await _tool_export_report(
                str(arguments["project_path"]),
                str(arguments["run_id"]),
                str(arguments["output_path"]),
            )
        return {"error": {"type": "UnknownTool", "message": f"unknown tool: {name}"}}
    except KeyError as exc:
        return {"error": {"type": "InvalidInput", "message": f"missing input: {exc.args[0]}"}}


async def _tool_search_papers(
    project_path: str,
    query: str,
    max_results: int = 10,
) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path, must_exist=False)
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
            "paper_count": len(result.papers),
            "papers": [_paper_summary(paper) for paper in result.papers[: min(max_results, 20)]],
        }
    except Exception as exc:
        return _tool_error(exc)


async def _tool_read_paper(project_path: str, paper_id_or_pdf: str) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path, must_exist=False)
        ensure_project_layout(project)
        target = _safe_project_reference(project, paper_id_or_pdf)
        run_id = "mcp_read"
        with Database(project / ".scholar" / "state.db") as connection:
            repository = ScholarRepository(connection)
            sink = EventSink(project / ".scholar" / "runs" / run_id / "events.jsonl")
            settings = ScholarSettings.load(project)
            evidence = await Reader(
                repository,
                project,
                sink,
                run_id=run_id,
                budget_manager=BudgetManager(settings.budget),
            ).read_paper(target)
        return {
            "evidence_count": len(evidence),
            "evidence": [item.model_dump(mode="json") for item in evidence[:20]],
        }
    except Exception as exc:
        return _tool_error(exc)


async def _tool_compare_papers(project_path: str, paper_ids: list[str]) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path)
        with Database(project / ".scholar" / "state.db") as connection:
            repository = ScholarRepository(connection)
            papers = [paper for paper_id in paper_ids if (paper := repository.get_paper(paper_id))]
        return {"papers": [paper.model_dump(mode="json") for paper in papers[:50]]}
    except Exception as exc:
        return _tool_error(exc)


async def _tool_find_evidence(project_path: str, work_id: str) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path)
        with Database(project / ".scholar" / "state.db") as connection:
            evidence = ScholarRepository(connection).list_evidence(work_id)
        return {
            "evidence_count": len(evidence),
            "evidence": [item.model_dump(mode="json") for item in evidence[:50]],
        }
    except Exception as exc:
        return _tool_error(exc)


async def _tool_get_claims(project_path: str) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path)
        with Database(project / ".scholar" / "state.db") as connection:
            claims = ScholarRepository(connection).list_claims()
        return {
            "claim_count": len(claims),
            "claims": [claim.model_dump(mode="json") for claim in claims[:50]],
        }
    except Exception as exc:
        return _tool_error(exc)


async def _tool_run_research(
    project_path: str,
    question: str,
    no_embeddings: bool = False,
) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path, must_exist=False)
        result = await ResearchOrchestrator(project).run_research(
            question,
            no_embeddings=no_embeddings,
        )
        summary = result.summary()
        summary["artifacts"] = {
            "run_path": str(result.run_path),
            "report": str(result.run_path / "report.md"),
        }
        return summary
    except Exception as exc:
        return _tool_error(exc)


async def _tool_get_run_status(project_path: str, run_id: str) -> dict[str, Any]:
    try:
        run = ResearchOrchestrator(_resolve_project_path(project_path)).get_run(run_id)
        return {"run": run}
    except Exception as exc:
        return _tool_error(exc)


async def _tool_export_report(
    project_path: str,
    run_id: str,
    output_path: str,
) -> dict[str, Any]:
    try:
        project = _resolve_project_path(project_path)
        run = ResearchOrchestrator(project).get_run(run_id)
        if not run:
            return {"error": {"type": "NotFound", "message": f"run not found: {run_id}"}}
        source = Path(str(run["data"]["run_path"])) / "report.md"
        target = _safe_output_path(project, output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return {"output_path": str(target)}
    except Exception as exc:
        return _tool_error(exc)


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


def _resolve_project_path(project_path: str, *, must_exist: bool = True) -> Path:
    project = Path(project_path).expanduser().resolve()
    if must_exist and not project.exists():
        msg = f"project path does not exist: {project}"
        raise FileNotFoundError(msg)
    if project.exists() and not project.is_dir():
        msg = f"project path is not a directory: {project}"
        raise NotADirectoryError(msg)
    return project


def _safe_project_reference(project: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists() or "/" in value or "\\" in value:
        resolved = (path if path.is_absolute() else project / path).resolve()
        if not resolved.is_relative_to(project):
            msg = f"path escapes project root: {value}"
            raise PermissionError(msg)
        return str(resolved)
    return value


def _safe_output_path(project: Path, value: str) -> Path:
    target = Path(value).expanduser()
    resolved = (target if target.is_absolute() else project / target).resolve()
    if not resolved.is_relative_to(project):
        msg = f"output path escapes project root: {value}"
        raise PermissionError(msg)
    return resolved


def _paper_summary(paper: Any) -> dict[str, Any]:
    return {
        "work_id": paper.work_id,
        "title": paper.title,
        "year": paper.year,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "sources": sorted(paper.source_ids),
        "relevance_score": paper.relevance_score,
    }


def _tool_error(exc: Exception) -> dict[str, Any]:
    return {"error": {"type": type(exc).__name__, "message": str(exc)}}

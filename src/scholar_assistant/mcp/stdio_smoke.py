from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout


async def run_smoke(project_path: Path | None = None) -> dict[str, Any]:
    created_tmp = False
    if project_path is None:
        project_path = Path(tempfile.mkdtemp(prefix="scholar-mcp-smoke-"))
        created_tmp = True
    ensure_project_layout(project_path)
    with Database(project_path / ".scholar" / "state.db"):
        pass

    env = dict(os.environ)
    env["SCHOLAR_DEMO_MODE"] = "1"
    server = StdioServerParameters(
        command="uv",
        args=["run", "scholar", "mcp-server"],
        cwd=str(Path(__file__).resolve().parents[3]),
        env=env,
    )
    result: dict[str, Any] = {"project_path": str(project_path)}
    try:
        async with stdio_client(server, errlog=sys.stderr) as (read, write), ClientSession(
            read, write
        ) as session:
            initialized = await session.initialize()
            tools_response = await session.list_tools()
            tool_names = [tool.name for tool in tools_response.tools]
            status = await session.call_tool(
                "scholar_get_run_status",
                {"project_path": str(project_path), "run_id": "missing"},
            )
            claims = await session.call_tool(
                "scholar_get_claims",
                {"project_path": str(project_path)},
            )
            research = await session.call_tool(
                "scholar_run_research",
                {
                    "project_path": str(project_path),
                    "question": "LLM agent memory retrieval noise",
                    "no_embeddings": True,
                },
            )
            result.update(
                {
                    "initialized": initialized.protocolVersion,
                    "tools_count": len(tool_names),
                    "tools": tool_names,
                    "status_call_serializable": _jsonable(status.model_dump(mode="json")),
                    "claims_call_serializable": _jsonable(claims.model_dump(mode="json")),
                    "research_call_serializable": _jsonable(research.model_dump(mode="json")),
                }
            )
    finally:
        if created_tmp:
            shutil.rmtree(project_path, ignore_errors=True)
    return result


def _jsonable(value: Any) -> bool:
    json.dumps(value, ensure_ascii=False, default=str)
    return True


def run_sync(project_path: Path | None = None) -> dict[str, Any]:
    return asyncio.run(run_smoke(project_path))

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

import typer

from scholar_assistant.agents.reader import Reader
from scholar_assistant.agents.searcher import Searcher
from scholar_assistant.cli.exec_command import execute_task
from scholar_assistant.cli.rendering import (
    print_papers,
    print_provider_table,
    stderr_console,
    stdout_console,
)
from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.core.events import EventSink
from scholar_assistant.core.orchestrator import ResearchOrchestrator
from scholar_assistant.providers.base import ModelMessage, ModelRequest, ProviderError
from scholar_assistant.providers.router import ModelRouter
from scholar_assistant.retrieval.embeddings import BGEM3Embedder
from scholar_assistant.retrieval.reranker import BGEReranker
from scholar_assistant.schemas.events import RunEvent, RunEventType
from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout
from scholar_assistant.storage.repositories import ScholarRepository

app = typer.Typer(help="Scholar-Assistant local academic research agent.")
config_app = typer.Typer(help="Configuration commands.")
providers_app = typer.Typer(help="Model provider commands.")
app.add_typer(config_app, name="config")
app.add_typer(providers_app, name="providers")


def _project_path(path: Path | None = None) -> Path:
    return (path or Path.cwd()).resolve()


@app.command()
def init(project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd()) -> None:
    """Initialize local Scholar project directories and SQLite state."""
    project = _project_path(project_path)
    paths = ensure_project_layout(project)
    with Database(project / ".scholar" / "state.db"):
        pass
    stdout_console.print(f"Initialized Scholar project at {project}")
    stdout_console.print(f"Config: {project / '.scholar' / 'config.toml'}")
    stdout_console.print(f"State DB: {project / '.scholar' / 'state.db'}")
    stdout_console.print(f"Created/verified {len(paths)} directories")


@app.command()
def doctor(
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Check local runtime, optional ML dependencies, and SQLite FTS5."""
    project = _project_path(project_path)
    ensure_project_layout(project)
    settings = ScholarSettings.load(project)
    checks: list[tuple[str, str]] = []
    checks.append(("python", sys.version.split()[0]))
    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        checks.append(("sqlite_fts5", "ok"))
    except sqlite3.Error as exc:
        checks.append(("sqlite_fts5", f"failed: {exc}"))
    checks.append(("bge_m3", "available" if BGEM3Embedder().available else "not installed"))
    checks.append(("bge_reranker", "available" if BGEReranker().available else "not installed"))
    try:
        import fitz  # noqa: F401

        checks.append(("pymupdf", "ok"))
    except ImportError:
        checks.append(("pymupdf", "missing"))
    try:
        import mcp  # noqa: F401

        checks.append(("mcp", "ok"))
    except ImportError:
        checks.append(("mcp", "missing"))
    for name, status in checks:
        stdout_console.print(f"{name}: {status}")
    missing_keys = [
        provider.api_key_env for provider in settings.providers.values() if not provider.has_api_key
    ]
    if missing_keys:
        stdout_console.print(f"missing_api_keys: {', '.join(sorted(missing_keys))}")


@config_app.command("show")
def config_show(
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Show resolved non-secret configuration."""
    settings = ScholarSettings.load(_project_path(project_path))
    stdout_console.print_json(settings.model_dump_json())


@providers_app.command("list")
def providers_list(
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """List configured providers without printing API keys."""
    settings = ScholarSettings.load(_project_path(project_path))
    print_provider_table(settings)


@providers_app.command("test")
def providers_test(
    provider: str,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Send a small test request to a configured provider."""
    project = _project_path(project_path)
    settings = ScholarSettings.load(project)
    if provider not in settings.providers:
        stderr_console.print(f"Unknown provider: {provider}")
        raise typer.Exit(1)
    provider_config = settings.providers[provider]
    if not provider_config.has_api_key:
        stderr_console.print(
            f"Configuration error: missing environment variable {provider_config.api_key_env}"
        )
        raise typer.Exit(2)
    model_alias = next(
        (name for name, model in settings.models.items() if model.provider == provider),
        settings.default_model,
    )

    async def _test() -> None:
        router = ModelRouter(settings)
        response = await router.complete(
            model_alias,
            ModelRequest(
                model="",
                messages=[ModelMessage(role="user", content="Return the word OK.")],
                max_output_tokens=16,
            ),
        )
        stdout_console.print_json(
            json.dumps(
                {
                    "provider": provider,
                    "model_alias": model_alias,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage.model_dump(),
                    "content_preview": response.content[:120],
                    "provider_state_keys": sorted(response.provider_state),
                }
            )
        )

    try:
        asyncio.run(_test())
    except ProviderError as exc:
        stderr_console.print(
            json.dumps(
                {
                    "error": exc.error_type.value,
                    "status_code": exc.status_code,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        raise typer.Exit(1) from exc


@app.command()
def search(
    question: str,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
    max_results: Annotated[int, typer.Option("--max-results")] = 20,
    no_embeddings: Annotated[bool, typer.Option("--no-embeddings")] = False,
) -> None:
    """Search arXiv and store selected papers."""
    project = _project_path(project_path)
    ensure_project_layout(project)
    run_id = "search_cli"
    with Database(project / ".scholar" / "state.db") as connection:
        repository = ScholarRepository(connection)
        sink = EventSink(project / ".scholar" / "runs" / run_id / "events.jsonl")
        searcher = Searcher(repository, ScholarSettings.load(project), project, sink, run_id=run_id)
        result = asyncio.run(
            searcher.search(
                question, max_results_per_query=max_results, no_embeddings=no_embeddings
            )
        )
    for warning in result.warnings:
        stderr_console.print(f"warning: {warning}")
    stderr_console.print(f"retrieval_mode: {result.retrieval_mode}")
    print_papers(result.papers)


@app.command()
def read(
    paper_id_or_pdf: str,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Read a stored paper ID or local PDF and create evidence units."""
    project = _project_path(project_path)
    ensure_project_layout(project)
    run_id = "read_cli"
    with Database(project / ".scholar" / "state.db") as connection:
        repository = ScholarRepository(connection)
        sink = EventSink(project / ".scholar" / "runs" / run_id / "events.jsonl")
        reader = Reader(repository, project, sink, run_id=run_id)
        evidence = asyncio.run(reader.read_paper(paper_id_or_pdf))
    stdout_console.print_json(json.dumps([item.model_dump(mode="json") for item in evidence]))


@app.command()
def compare(
    paper_ids: list[str],
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Compare stored papers at a metadata level."""
    project = _project_path(project_path)
    with Database(project / ".scholar" / "state.db") as connection:
        repository = ScholarRepository(connection)
        papers = [paper for paper_id in paper_ids if (paper := repository.get_paper(paper_id))]
    stdout_console.print_json(
        json.dumps([paper.model_dump(mode="json") for paper in papers], default=str)
    )


@app.command()
def research(
    question: str,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
    no_embeddings: Annotated[bool, typer.Option("--no-embeddings")] = False,
) -> None:
    """Run Search -> Read -> Think -> Verify -> Report."""
    project = _project_path(project_path)
    stderr_console.print("running research workflow")
    result = asyncio.run(
        ResearchOrchestrator(project).run_research(question, no_embeddings=no_embeddings)
    )
    for warning in result.warnings:
        stderr_console.print(f"warning: {warning}")
    stdout_console.print_json(json.dumps(result.summary(), ensure_ascii=False, default=str))
    stderr_console.print(f"report: {result.run_path / 'report.md'}")


@app.command("exec")
def exec_command(
    task: str,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
    json_mode: Annotated[bool, typer.Option("--json")] = False,
    output_schema: Annotated[Path | None, typer.Option("--output-schema")] = None,
    ephemeral: Annotated[bool, typer.Option("--ephemeral")] = False,
    no_embeddings: Annotated[bool, typer.Option("--no-embeddings")] = False,
) -> None:
    """Run a non-interactive Codex-style Scholar task."""
    if task == "-":
        task = sys.stdin.read()
    if not json_mode:
        stderr_console.print("running exec task")
    try:
        code, output = asyncio.run(
            execute_task(
                task,
                project_path=_project_path(project_path),
                json_mode=json_mode,
                output_schema_path=output_schema,
                ephemeral=ephemeral,
                no_embeddings=no_embeddings,
            )
        )
    except Exception as exc:
        if json_mode:
            event = RunEvent.new(
                RunEventType.ERROR,
                payload={"error_type": type(exc).__name__, "message": str(exc)},
            )
            sys.stdout.write(event.model_dump_json() + "\n")
        else:
            stderr_console.print(f"error: {type(exc).__name__}: {exc}")
        raise typer.Exit(1) from exc
    sys.stdout.write(output)
    raise typer.Exit(code)


@app.command()
def resume(
    run_id: str,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Show persisted run metadata for a previous run."""
    run = ResearchOrchestrator(_project_path(project_path)).get_run(run_id)
    if not run:
        stderr_console.print(f"Run not found: {run_id}")
        raise typer.Exit(1)
    stdout_console.print_json(json.dumps(run, ensure_ascii=False, default=str))


@app.command()
def status(
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Show latest run status."""
    run = ResearchOrchestrator(_project_path(project_path)).latest_run()
    stdout_console.print_json(
        json.dumps(run or {"status": "no_runs"}, ensure_ascii=False, default=str)
    )


@app.command()
def export(
    output_path: Path,
    project_path: Annotated[Path, typer.Option("--project-path", "-p")] = Path.cwd(),
) -> None:
    """Export the latest Markdown report to a target path."""
    orchestrator = ResearchOrchestrator(_project_path(project_path))
    run = orchestrator.latest_run()
    if not run:
        stderr_console.print("No run available to export")
        raise typer.Exit(1)
    source = Path(str(run["data"]["run_path"])) / "report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output_path)
    stdout_console.print(str(output_path.resolve()))


@app.command("mcp-server")
def mcp_server(
    list_tools: Annotated[bool, typer.Option("--list-tools")] = False,
) -> None:
    """Start the Scholar MCP server over stdio."""
    from scholar_assistant.mcp.server import list_mcp_tools, run_mcp_server

    if list_tools:
        stdout_console.print_json(json.dumps(list_mcp_tools(), ensure_ascii=False))
        return
    run_mcp_server()


def main() -> None:
    app()


if __name__ == "__main__":
    main()

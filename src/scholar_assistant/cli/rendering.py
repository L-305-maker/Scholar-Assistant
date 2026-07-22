from __future__ import annotations

from rich.console import Console
from rich.table import Table

from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.schemas.paper import Paper

stdout_console = Console()
stderr_console = Console(stderr=True)


def print_provider_table(settings: ScholarSettings) -> None:
    table = Table(title="Providers")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Base URL")
    table.add_column("API Key Env")
    table.add_column("Key Set")
    for name, provider in sorted(settings.providers.items()):
        table.add_row(
            name,
            provider.type,
            provider.base_url,
            provider.api_key_env,
            "yes" if provider.has_api_key else "no",
        )
    stdout_console.print(table)


def print_papers(papers: list[Paper]) -> None:
    table = Table(title="Selected Papers")
    table.add_column("Work ID")
    table.add_column("Year")
    table.add_column("Title")
    table.add_column("Source")
    table.add_column("Score")
    for paper in papers:
        table.add_row(
            paper.work_id,
            str(paper.year or ""),
            paper.title[:80],
            paper.source,
            f"{paper.relevance_score:.4f}",
        )
    stdout_console.print(table)

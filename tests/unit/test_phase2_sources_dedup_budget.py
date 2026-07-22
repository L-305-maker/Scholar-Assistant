from __future__ import annotations

from pathlib import Path

import pytest

import scholar_assistant.agents.searcher as searcher_module
from scholar_assistant.agents.searcher import Searcher
from scholar_assistant.core.budget import BudgetLimitExceeded, BudgetManager
from scholar_assistant.core.config import BudgetConfig, ScholarSettings
from scholar_assistant.core.events import EventSink
from scholar_assistant.schemas.paper import Paper
from scholar_assistant.storage.canonicalization import (
    decide_duplicate,
    normalize_arxiv_base,
    normalize_doi,
    normalize_title,
)
from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout
from scholar_assistant.storage.repositories import ScholarRepository
from scholar_assistant.tools.registry import (
    ToolExecutionContext,
    ToolExecutor,
    ToolPermissionError,
    ToolPermissionLevel,
    ToolRegistry,
    ToolSpec,
)
from scholar_assistant.tools.sources import SourceHit, SourceSearchRequest, SourceSearchResponse


class FakeSource:
    def __init__(self, name: str, hits: list[SourceHit] | None = None, fail: bool = False) -> None:
        self.name = name
        self.hits = hits or []
        self.fail = fail

    async def search(self, request: SourceSearchRequest) -> SourceSearchResponse:
        if self.fail:
            raise RuntimeError("source down")
        return SourceSearchResponse(
            source=self.name,
            query_id=request.query_id,
            hits=[hit.model_copy(update={"query_id": request.query_id}) for hit in self.hits],
            result_count=len(self.hits),
        )


@pytest.mark.asyncio
async def test_multi_source_fusion_preserves_provenance_and_source_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_project_layout(tmp_path)
    duplicate_arxiv = SourceHit(
        source="arxiv",
        source_id="2401.12345v2",
        title="Memory Retrieval Noise in LLM Agents",
        authors=["Ada Lovelace", "Alan Turing"],
        abstract="LLM agent memory retrieval noise.",
        year=2024,
        arxiv_id="2401.12345v2",
        query_id="q1",
        raw_rank=1,
    )
    duplicate_openalex = SourceHit(
        source="openalex",
        source_id="https://openalex.org/W123",
        title="Memory retrieval noise in LLM agents",
        authors=["A. Lovelace", "Alan Turing"],
        abstract="Noisy memory retrieval in agents.",
        year=2024,
        doi="https://doi.org/10.1000/Agent.Memory",
        query_id="q1",
        raw_rank=2,
    )
    distinct = SourceHit(
        source="semantic_scholar",
        source_id="S2-1",
        title="Vision Classification Benchmarks",
        authors=["Grace Hopper"],
        abstract="Image classification.",
        year=2024,
        query_id="q1",
        raw_rank=3,
    )
    fake_sources = {
        "arxiv": FakeSource("arxiv", [duplicate_arxiv]),
        "openalex": FakeSource("openalex", [duplicate_openalex]),
        "crossref": FakeSource("crossref", fail=True),
        "semantic_scholar": FakeSource("semantic_scholar", [distinct]),
    }
    monkeypatch.setattr(
        searcher_module,
        "build_literature_sources",
        lambda *_args, **_kw: fake_sources,
    )

    with Database(tmp_path / ".scholar" / "state.db") as connection:
        repository = ScholarRepository(connection)
        searcher = Searcher(
            repository,
            ScholarSettings.defaults(),
            tmp_path,
            EventSink(),
            run_id="phase2_multi",
        )
        result = await searcher.search(
            "LLM agent memory retrieval noise",
            no_embeddings=True,
            sources=["arxiv", "openalex", "crossref", "semantic-scholar"],
        )
        provenance = repository.list_retrieval_provenance("phase2_multi")

    assert result.papers
    assert any("crossref search failed" in warning for warning in result.warnings)
    assert len({paper.work_id for paper in result.papers}) == len(result.papers)
    assert {row["source"] for row in provenance} >= {"arxiv", "openalex", "semantic_scholar"}
    assert result.source_stats["crossref"]["failures"] >= 1


def test_canonicalization_and_conservative_dedup(tmp_path: Path) -> None:
    assert normalize_doi("https://doi.org/10.1000/ABC ") == "10.1000/abc"
    assert normalize_doi("doi:10.1000/ABC") == "10.1000/abc"
    assert normalize_arxiv_base("arXiv:2401.12345v2") == "2401.12345"
    assert normalize_title("Memory--Retrieval: Noise\nin {LLM} Agents") == (
        "memory retrieval noise in llm agents"
    )

    with Database(tmp_path / "state.db") as connection:
        repository = ScholarRepository(connection)
        first = repository.upsert_paper(
            Paper(
                title="Memory Retrieval Noise in LLM Agents",
                authors=["Ada Lovelace"],
                year=2024,
                doi="https://doi.org/10.1000/ABC",
                arxiv_id="2401.12345v1",
                source_ids={"arxiv": "2401.12345v1"},
            )
        )
        doi_duplicate = repository.upsert_paper(
            Paper(
                title="Memory retrieval noise in LLM agents: a study",
                authors=["A. Lovelace"],
                year=2025,
                doi="doi:10.1000/abc",
            )
        )
        arxiv_duplicate = repository.upsert_paper(
            Paper(title="Memory Retrieval Noise", arxiv_id="2401.12345v2")
        )
        source_duplicate = repository.upsert_paper(
            Paper(
                title="A source-ID duplicate",
                source="openalex",
                source_ids={"openalex": "W123"},
            )
        )
        source_duplicate_2 = repository.upsert_paper(
            Paper(
                title="A source ID duplicate with punctuation",
                source="openalex",
                source_ids={"openalex": "W123"},
            )
        )
        assert first.work_id == doi_duplicate.work_id == arxiv_duplicate.work_id
        assert source_duplicate.work_id == source_duplicate_2.work_id

        possible_left = repository.upsert_paper(
            Paper(title="Agent Memory Retrieval Benchmarks", authors=["A. Smith"], year=2024)
        )
        possible_right = repository.upsert_paper(
            Paper(title="Agent Memory Retrieval Benchmark", authors=["B. Smith"], year=2025)
        )
        assert possible_left.work_id != possible_right.work_id

    conflict = decide_duplicate(
        Paper(title="Same Title", authors=["Ada Lovelace"], doi="10.1/a"),
        Paper(title="Same Title", authors=["Ada Lovelace"], doi="10.1/b"),
    )
    assert conflict.action == "never_merge"


@pytest.mark.asyncio
async def test_tool_registry_permission_and_budget() -> None:
    async def handler() -> dict[str, str]:
        return {"ok": "yes"}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="source.demo",
            description="demo",
            input_schema={},
            handler=handler,
            category="source",
            permission_level=ToolPermissionLevel.T1_EXTERNAL_READ,
        )
    )
    denied = ToolExecutor(
        registry,
        ToolExecutionContext(allowed_permissions={ToolPermissionLevel.T0_LOCAL_READ}),
    )
    with pytest.raises(ToolPermissionError):
        await denied.execute("source.demo")

    budget = BudgetManager(BudgetConfig(max_search_calls=1))
    executor = ToolExecutor(registry, ToolExecutionContext(budget=budget))
    assert await executor.execute("source.demo") == {"ok": "yes"}
    with pytest.raises(BudgetLimitExceeded):
        await executor.execute("source.demo")
    budget.record_cache_hit("source.demo")
    assert budget.snapshot()["cache_hits"]["source.demo"] == 1

from __future__ import annotations

from pathlib import Path

import numpy as np

import scholar_assistant.agents.searcher as searcher_module
from scholar_assistant.agents.searcher import Searcher
from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.core.events import EventSink
from scholar_assistant.retrieval.bm25 import BM25Retriever
from scholar_assistant.retrieval.diversity import diverse_select
from scholar_assistant.retrieval.fusion import RankedItem, reciprocal_rank_fusion
from scholar_assistant.retrieval.reranker import fallback_rerank
from scholar_assistant.schemas.paper import Paper, PaperVersion, VersionType
from scholar_assistant.schemas.research import QueryPlan
from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout
from scholar_assistant.storage.repositories import ScholarRepository
from scholar_assistant.tools.arxiv import parse_arxiv_atom


def test_arxiv_atom_parse() -> None:
    xml = Path("tests/fixtures/arxiv_sample.xml").read_text(encoding="utf-8")
    result = parse_arxiv_atom(xml)
    assert len(result.papers) == 2
    assert result.papers[0].arxiv_id == "2401.00001v2"
    assert result.papers[0].doi == "10.1234/example"
    assert result.versions[0].access_type.value == "fulltext"


def test_dedup_doi_arxiv_and_title_merge(tmp_path: Path) -> None:
    with Database(tmp_path / "state.db") as connection:
        repository = ScholarRepository(connection)
        paper = Paper(title="Memory Retrieval Noise in LLM Agents", doi="10.1234/example")
        version = PaperVersion(work_id=paper.work_id, version_type=VersionType.ARXIV)
        stored = repository.upsert_paper(paper, version)
        duplicate = Paper(
            title="Memory Retrieval Noise in LLM Agents",
            doi="10.1234/example",
            arxiv_id="2401.00001v2",
        )
        stored_duplicate = repository.upsert_paper(duplicate)
        assert stored.work_id == stored_duplicate.work_id
        assert len(repository.list_versions(stored.work_id)) == 1


def test_fts5_bm25_search(tmp_path: Path) -> None:
    with Database(tmp_path / "state.db") as connection:
        repository = ScholarRepository(connection)
        paper = Paper(
            title="Memory Retrieval Noise in LLM Agents",
            abstract="Noisy retrieval injects irrelevant memories into agents.",
        )
        repository.upsert_paper(paper)
        hits = BM25Retriever(connection).search("retrieval noise agents")
        assert hits
        assert hits[0].work_id == paper.work_id


def test_rrf_orders_shared_items_first() -> None:
    fused = reciprocal_rank_fusion(
        [
            [RankedItem("a", 1.0, "bm25"), RankedItem("b", 0.5, "bm25")],
            [RankedItem("b", 0.9, "dense"), RankedItem("a", 0.3, "dense")],
        ],
        k=10,
    )
    assert [item.item_id for item in fused] == ["a", "b"]
    assert fused[0].source == "bm25+dense"


def test_reranker_fallback_and_diversity() -> None:
    papers = [
        Paper(title="Old Foundation", year=2019, relevance_score=0.9),
        Paper(title="Recent Memory Noise", year=2024, relevance_score=0.8),
        Paper(title="Benchmark for Agent Memory", year=2024, relevance_score=0.7),
    ]
    reranked = fallback_rerank(papers)
    selected = diverse_select(papers, max_count=2)
    assert reranked[0].score == 0.9
    assert len(selected) == 2
    assert len({paper.paper_role for paper in selected}) >= 2


def test_searcher_dense_retrieval_path_persists_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeEmbedder:
        available = True

        def encode(self, texts: list[str]) -> np.ndarray:
            vectors = np.zeros((len(texts), 2), dtype=np.float32)
            vectors[0] = [1.0, 0.0]
            if len(texts) > 1:
                vectors[1] = [1.0, 0.0]
            if len(texts) > 2:
                vectors[2:] = [0.0, 1.0]
            return vectors

    class FakeReranker:
        available = False

    ensure_project_layout(tmp_path)
    monkeypatch.setattr(searcher_module, "BGEM3Embedder", FakeEmbedder)
    monkeypatch.setattr(searcher_module, "BGEReranker", FakeReranker)
    with Database(tmp_path / ".scholar" / "state.db") as connection:
        repository = ScholarRepository(connection)
        papers = [
            repository.upsert_paper(
                Paper(title="Dense Memory Retrieval", abstract="LLM agent memory retrieval")
            ),
            repository.upsert_paper(Paper(title="Unrelated Topic", abstract="vision data")),
        ]
        searcher = Searcher(
            repository,
            ScholarSettings.defaults(),
            tmp_path,
            EventSink(),
            run_id="dense_test",
        )
        selected, mode = searcher._retrieve_and_select(
            QueryPlan(user_question="memory", core_query="memory retrieval"),
            papers,
            no_embeddings=False,
        )
    assert any(paper.title == "Dense Memory Retrieval" for paper in selected)
    assert "bge-m3-dense" in mode
    assert list((tmp_path / ".scholar" / "index").glob("dense-*.npz"))

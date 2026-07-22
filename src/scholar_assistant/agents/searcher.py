from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx

from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.core.events import EventSink
from scholar_assistant.core.planner import TaskPlanner
from scholar_assistant.retrieval.bm25 import BM25Retriever
from scholar_assistant.retrieval.diversity import diverse_select
from scholar_assistant.retrieval.embeddings import BGEM3Embedder, cosine_search, save_vectors
from scholar_assistant.retrieval.fusion import RankedItem, reciprocal_rank_fusion
from scholar_assistant.retrieval.reranker import BGEReranker, fallback_rerank
from scholar_assistant.schemas.events import RunEvent, RunEventType
from scholar_assistant.schemas.paper import Paper, PaperVersion
from scholar_assistant.schemas.research import QueryPlan
from scholar_assistant.storage.files import sha256_text, write_json
from scholar_assistant.storage.repositories import ScholarRepository
from scholar_assistant.tools.arxiv import ArxivClient, ArxivSearchResult, merge_results
from scholar_assistant.tools.demo_data import offline_demo_papers


@dataclass(slots=True)
class SearchResult:
    query_plan: QueryPlan
    papers: list[Paper]
    versions: list[PaperVersion]
    retrieval_mode: str
    warnings: list[str] = field(default_factory=list)


class Searcher:
    def __init__(
        self,
        repository: ScholarRepository,
        settings: ScholarSettings,
        project_path: Path,
        event_sink: EventSink,
        *,
        run_id: str,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.project_path = project_path
        self.event_sink = event_sink
        self.run_id = run_id
        self.planner = TaskPlanner()
        self.arxiv = ArxivClient()

    async def search(
        self,
        question: str,
        *,
        max_results_per_query: int = 20,
        no_embeddings: bool = False,
    ) -> SearchResult:
        query_plan = self.planner.plan_search(question)
        query_results: list[ArxivSearchResult] = []
        warnings: list[str] = []
        if self.settings.demo_mode:
            warnings.append("demo_mode enabled; skipping live arXiv search.")
            for query in query_plan.all_queries()[:2]:
                self.event_sink.emit(
                    RunEvent.new(
                        RunEventType.SEARCH_QUERY,
                        run_id=self.run_id,
                        payload={"query": query, "source": "offline_demo"},
                        tool="offline_demo",
                    )
                )
        else:
            for query in query_plan.all_queries()[: self.settings.budget.main_search_loops + 4]:
                self.event_sink.emit(
                    RunEvent.new(
                        RunEventType.SEARCH_QUERY,
                        run_id=self.run_id,
                        payload={"query": query, "source": "arxiv"},
                        tool="arxiv",
                    )
                )
                try:
                    query_results.append(
                        await self.arxiv.search(query, max_results=max_results_per_query)
                    )
                except (TimeoutError, httpx.HTTPError, OSError) as exc:
                    warnings.append(
                        f"arXiv search failed for query '{query}': {type(exc).__name__}"
                    )

        if query_results:
            merged = merge_results(query_results)
            papers, versions = merged.papers, merged.versions
            raw_path = self.project_path / ".scholar" / "cache" / f"{self.run_id}-arxiv.xml"
            raw_path.write_text(merged.raw_xml, encoding="utf-8")
        else:
            papers, versions = offline_demo_papers(question)
            warnings.append(
                "No live arXiv results were available; using marked offline demo metadata."
            )

        stored: list[Paper] = []
        version_by_initial_work = {version.work_id: version for version in versions}
        for paper in papers[: self.settings.budget.max_raw_candidates]:
            version = version_by_initial_work.get(paper.work_id)
            stored_paper = self.repository.upsert_paper(paper, version)
            stored.append(stored_paper)
            self.event_sink.emit(
                RunEvent.new(
                    RunEventType.PAPER_DISCOVERED,
                    run_id=self.run_id,
                    payload={
                        "work_id": stored_paper.work_id,
                        "title": stored_paper.title,
                        "source": stored_paper.source,
                    },
                    tool=stored_paper.source,
                )
            )

        selected, retrieval_mode = self._retrieve_and_select(
            query_plan,
            stored,
            no_embeddings=no_embeddings,
        )
        for paper in selected:
            self.event_sink.emit(
                RunEvent.new(
                    RunEventType.PAPER_SELECTED,
                    run_id=self.run_id,
                    payload={
                        "work_id": paper.work_id,
                        "title": paper.title,
                        "score": paper.relevance_score,
                        "role": paper.paper_role.value,
                    },
                )
            )
        write_json(
            self.project_path / ".scholar" / "runs" / self.run_id / "search-plan.json",
            query_plan.model_dump(mode="json"),
        )
        return SearchResult(
            query_plan=query_plan,
            papers=selected,
            versions=versions,
            retrieval_mode=retrieval_mode,
            warnings=warnings,
        )

    def _retrieve_and_select(
        self,
        query_plan: QueryPlan,
        candidates: list[Paper],
        *,
        no_embeddings: bool,
    ) -> tuple[list[Paper], str]:
        bm25 = BM25Retriever(self.repository.connection)
        rankings: list[list[RankedItem]] = [
            bm25.ranked_items(query, limit=self.settings.budget.max_rerank_candidates)
            for query in query_plan.all_queries()
        ]
        dense_mode = "dense-disabled" if no_embeddings else "dense-unavailable"
        if not no_embeddings and candidates:
            dense_ranking, dense_mode = self._dense_rank(query_plan.core_query, candidates)
            if dense_ranking:
                rankings.append(dense_ranking)

        fused = reciprocal_rank_fusion([ranking for ranking in rankings if ranking])
        score_by_work = {item.item_id: item.score for item in fused}

        if not score_by_work:
            score_by_work = {
                paper.work_id: lexical_overlap(
                    query_plan.core_query, f"{paper.title} {paper.abstract or ''}"
                )
                for paper in candidates
            }

        for paper in candidates:
            paper.relevance_score = score_by_work.get(paper.work_id, 0.0)

        base_mode = "bm25"
        if dense_mode == "bge-m3-dense":
            base_mode = "bm25+bge-m3-dense"
        elif no_embeddings:
            base_mode = "bm25-only"
        elif dense_mode != "bge-m3-dense":
            base_mode = "bm25+optional-dense-unavailable"

        rerank_mode = base_mode
        reranker = BGEReranker()
        ranked_candidates = sorted(candidates, key=lambda item: item.relevance_score, reverse=True)[
            : self.settings.budget.max_rerank_candidates
        ]
        if not no_embeddings and reranker.available and ranked_candidates:
            try:
                reranked = reranker.rerank(query_plan.core_query, ranked_candidates)
                score_by_work = {item.work_id: item.score for item in reranked}
                rerank_mode = f"{base_mode}+bge-reranker-v2-m3"
            except RuntimeError:
                reranked = fallback_rerank(ranked_candidates)
                score_by_work = {item.work_id: item.score for item in reranked}
                rerank_mode = f"{base_mode}+reranker-unavailable"
        else:
            reranked = fallback_rerank(ranked_candidates)
            score_by_work = {item.work_id: item.score for item in reranked}
            if no_embeddings:
                rerank_mode = "bm25-only"
            elif dense_mode == "bge-m3-dense":
                rerank_mode = "bm25+bge-m3-dense+reranker-unavailable"
            else:
                rerank_mode = "bm25+optional-ml-unavailable"

        for paper in ranked_candidates:
            paper.relevance_score = score_by_work.get(paper.work_id, paper.relevance_score)
        selected = diverse_select(
            ranked_candidates,
            max_count=min(self.settings.budget.max_core_papers, len(ranked_candidates)),
        )
        return selected, rerank_mode

    def _dense_rank(self, query: str, candidates: list[Paper]) -> tuple[list[RankedItem], str]:
        embedder = BGEM3Embedder()
        if not embedder.available:
            return [], "dense-unavailable"
        item_ids = [paper.work_id for paper in candidates]
        texts = [paper_text(paper) for paper in candidates]
        try:
            vectors = embedder.encode([query, *texts])
        except RuntimeError:
            return [], "dense-unavailable"
        query_vector = vectors[0]
        paper_vectors = vectors[1:]
        index_key = sha256_text("\n".join([query, *item_ids]))[:16]
        save_vectors(
            self.project_path / ".scholar" / "index" / f"dense-{index_key}.npz",
            item_ids,
            paper_vectors,
        )
        dense_hits = cosine_search(
            query_vector,
            item_ids,
            paper_vectors,
            limit=self.settings.budget.max_rerank_candidates,
        )
        return [
            RankedItem(item_id=hit.item_id, score=hit.score, source="bge-m3-dense")
            for hit in dense_hits
        ], "bge-m3-dense"


def lexical_overlap(query: str, text: str) -> float:
    query_terms = {term.lower() for term in query.split() if len(term) > 2}
    text_terms = {term.lower() for term in text.split() if len(term) > 2}
    if not query_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def paper_text(paper: Paper) -> str:
    authors = ", ".join(paper.authors[:6])
    return "\n".join(
        part
        for part in [
            paper.title,
            authors,
            paper.abstract or "",
            " ".join(paper.categories),
        ]
        if part
    )

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from scholar_assistant.core.budget import BudgetLimitExceeded, BudgetManager
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
from scholar_assistant.tools.demo_data import offline_demo_papers
from scholar_assistant.tools.registry import (
    RetryPolicy,
    ToolExecutionContext,
    ToolExecutor,
    ToolPermissionLevel,
    ToolRegistry,
    ToolSpec,
)
from scholar_assistant.tools.sources import (
    SourceSearchRequest,
    SourceSearchResponse,
    build_literature_sources,
    hit_to_paper_and_version,
)


@dataclass(slots=True)
class SearchResult:
    query_plan: QueryPlan
    papers: list[Paper]
    versions: list[PaperVersion]
    retrieval_mode: str
    warnings: list[str] = field(default_factory=list)
    source_stats: dict[str, dict[str, object]] = field(default_factory=dict)
    retrieval_provenance: list[dict[str, object]] = field(default_factory=list)


class Searcher:
    def __init__(
        self,
        repository: ScholarRepository,
        settings: ScholarSettings,
        project_path: Path,
        event_sink: EventSink,
        *,
        run_id: str,
        budget_manager: BudgetManager | None = None,
        enabled_sources: list[str] | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.project_path = project_path
        self.event_sink = event_sink
        self.run_id = run_id
        self.planner = TaskPlanner()
        self.budget_manager = budget_manager or BudgetManager(settings.budget)
        self.enabled_sources = enabled_sources
        self.sources = build_literature_sources(settings, enabled_names=enabled_sources)
        self.registry = self._build_registry()
        self.executor = ToolExecutor(
            self.registry,
            ToolExecutionContext(run_id=run_id, budget=self.budget_manager),
        )

    async def search(
        self,
        question: str,
        *,
        max_results_per_query: int = 20,
        no_embeddings: bool = False,
        sources: list[str] | None = None,
    ) -> SearchResult:
        if sources is not None:
            self.enabled_sources = sources
            self.sources = build_literature_sources(self.settings, enabled_names=sources)
            self.registry = self._build_registry()
            self.executor = ToolExecutor(
                self.registry,
                ToolExecutionContext(run_id=self.run_id, budget=self.budget_manager),
            )
        query_plan = self.planner.plan_search(question)
        warnings: list[str] = []
        responses: list[SourceSearchResponse] = []
        source_stats: dict[str, dict[str, object]] = {}
        if self.settings.demo_mode:
            warnings.append("demo_mode enabled; skipping live multi-source search.")
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
            responses = await self._search_sources(
                query_plan.all_queries()[: self.settings.budget.main_search_loops + 4],
                max_results_per_query=max_results_per_query,
            )
            for execution in self.executor.executions:
                self.repository.save_tool_execution(execution)
            for response in responses:
                stats = source_stats.setdefault(
                    response.source,
                    {
                        "result_count": 0,
                        "failures": 0,
                        "latency_ms": 0.0,
                        "warnings": [],
                    },
                )
                stats["result_count"] = int(stats["result_count"]) + response.result_count
                stats["latency_ms"] = float(stats["latency_ms"]) + response.latency_ms
                if response.error_type:
                    stats["failures"] = int(stats["failures"]) + 1
                stats["warnings"] = [*stats["warnings"], *response.warnings]
                warnings.extend(response.warnings)

        if responses and any(response.hits for response in responses):
            papers, versions, source_rankings, provenance = self._normalize_and_store_hits(
                responses,
                source_stats=source_stats,
            )
        else:
            papers, versions = offline_demo_papers(question)
            source_rankings = []
            provenance = []
            warnings.append(
                "No live source results were available; using marked offline demo metadata."
            )

        stored: list[Paper] = []
        version_by_initial_work = {version.work_id: version for version in versions}
        accepted_count = self.budget_manager.within_candidate_budget(len(papers))
        for paper in papers[:accepted_count]:
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
                        "retrieval_provenance": stored_paper.retrieval_provenance,
                    },
                    tool=stored_paper.source,
                )
            )

        selected, retrieval_mode = self._retrieve_and_select(
            query_plan,
            stored,
            no_embeddings=no_embeddings,
            source_rankings=source_rankings,
            source_weights=self._source_weights(source_rankings),
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
            source_stats=source_stats,
            retrieval_provenance=provenance,
        )

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for name, source in self.sources.items():
            config = self.settings.sources[name]

            async def handler(
                request: SourceSearchRequest,
                *,
                source=source,
            ) -> SourceSearchResponse:
                started = datetime.now(UTC)
                response = await source.search(request)
                response.latency_ms = (datetime.now(UTC) - started).total_seconds() * 1000
                return response

            registry.register(
                ToolSpec(
                    name=f"source.{name}",
                    description=f"Search {name} literature source",
                    input_schema=SourceSearchRequest.model_json_schema(),
                    output_schema=SourceSearchResponse.model_json_schema(),
                    handler=handler,
                    version="1",
                    category="source",
                    permission_level=ToolPermissionLevel.T1_EXTERNAL_READ,
                    timeout_seconds=config.timeout_seconds,
                    retry_policy=RetryPolicy(max_retries=config.max_retries),
                    network_domains=[name],
                )
            )
        return registry

    async def _search_sources(
        self,
        queries: list[str],
        *,
        max_results_per_query: int,
    ) -> list[SourceSearchResponse]:
        if not self.sources:
            return [
                SourceSearchResponse(
                    source="none",
                    query_id="no_sources",
                    warnings=["No literature sources are enabled."],
                    error_type="NoSourcesEnabled",
                )
            ]
        tasks = []
        for query_index, query in enumerate(queries):
            query_id = f"q{query_index + 1}"
            for source_name in self.sources:
                config = self.settings.sources[source_name]
                self.event_sink.emit(
                    RunEvent.new(
                        RunEventType.SEARCH_QUERY,
                        run_id=self.run_id,
                        payload={
                            "query_id": query_id,
                            "query": query,
                            "source": source_name,
                        },
                        tool=f"source.{source_name}",
                    )
                )
                request = SourceSearchRequest(
                    query_id=query_id,
                    query=query,
                    max_results=min(max_results_per_query, config.max_results),
                    timeout_seconds=config.timeout_seconds,
                    run_id=self.run_id,
                )
                tasks.append(self._execute_source(source_name, request))
        return await asyncio.gather(*tasks)

    async def _execute_source(
        self, source_name: str, request: SourceSearchRequest
    ) -> SourceSearchResponse:
        try:
            return await self.executor.execute(f"source.{source_name}", request=request)
        except BudgetLimitExceeded:
            raise
        except Exception as exc:
            warning = (
                f"{source_name} search failed for query '{request.query}': {type(exc).__name__}"
            )
            return SourceSearchResponse(
                source=source_name,
                query_id=request.query_id,
                warnings=[warning],
                error_type=type(exc).__name__,
            )

    def _normalize_and_store_hits(
        self,
        responses: list[SourceSearchResponse],
        *,
        source_stats: dict[str, dict[str, object]],
    ) -> tuple[list[Paper], list[PaperVersion], list[list[RankedItem]], list[dict[str, object]]]:
        papers_by_work: dict[str, Paper] = {}
        versions: list[PaperVersion] = []
        source_rankings_by_key: dict[tuple[str, str], list[RankedItem]] = {}
        provenance: list[dict[str, object]] = []
        for response in responses:
            config = self.settings.sources.get(response.source)
            source_weight = config.weight if config else 1.0
            for hit in response.hits:
                paper, version = hit_to_paper_and_version(hit)
                hit_provenance = hit.provenance(weight=source_weight)
                paper.retrieval_provenance = [hit_provenance]
                paper.metadata["retrieval_provenance"] = [hit_provenance]
                stored_paper = self.repository.upsert_paper(paper, version)
                version.work_id = stored_paper.work_id
                hit_provenance["work_id"] = stored_paper.work_id
                hit_data = hit.model_dump(mode="json")
                hit_data.update({"run_id": self.run_id, "work_id": stored_paper.work_id})
                self.repository.save_source_hit(hit_data)
                self.repository.save_retrieval_provenance(
                    run_id=self.run_id,
                    work_id=stored_paper.work_id,
                    provenance=hit_provenance,
                )
                existing_provenance = list(stored_paper.retrieval_provenance)
                if hit_provenance not in existing_provenance:
                    existing_provenance.append(hit_provenance)
                stored_paper.retrieval_provenance = existing_provenance
                stored_paper.metadata["retrieval_provenance"] = stored_paper.retrieval_provenance
                papers_by_work[stored_paper.work_id] = stored_paper
                versions.append(version)
                source_rankings_by_key.setdefault((response.source, response.query_id), []).append(
                    RankedItem(
                        item_id=stored_paper.work_id,
                        score=float(hit.raw_score or 1.0 / max(hit.raw_rank or 1, 1)),
                        source=f"{response.source}:{response.query_id}",
                    )
                )
                provenance.append(hit_provenance | {"hit_id": hit.hit_id})
        write_json(
            self.project_path / ".scholar" / "runs" / self.run_id / "source-stats.json",
            source_stats,
        )
        return (
            list(papers_by_work.values()),
            versions,
            list(source_rankings_by_key.values()),
            provenance,
        )

    def _source_weights(self, source_rankings: list[list[RankedItem]]) -> list[float]:
        weights: list[float] = []
        for ranking in source_rankings:
            if not ranking:
                weights.append(1.0)
                continue
            source_name = ranking[0].source.split(":", 1)[0]
            config = self.settings.sources.get(source_name)
            weights.append(config.weight if config else 1.0)
        return weights

    def _retrieve_and_select(
        self,
        query_plan: QueryPlan,
        candidates: list[Paper],
        *,
        no_embeddings: bool,
        source_rankings: list[list[RankedItem]] | None = None,
        source_weights: list[float] | None = None,
    ) -> tuple[list[Paper], str]:
        bm25 = BM25Retriever(self.repository.connection)
        rankings: list[list[RankedItem]] = [
            bm25.ranked_items(query, limit=self.settings.budget.max_rerank_candidates)
            for query in query_plan.all_queries()
        ]
        weights: list[float] = [1.0] * len(rankings)
        if source_rankings:
            rankings.extend(source_rankings)
            weights.extend(source_weights or [1.0] * len(source_rankings))
        dense_mode = "dense-disabled" if no_embeddings else "dense-unavailable"
        if not no_embeddings and candidates:
            dense_ranking, dense_mode = self._dense_rank(query_plan.core_query, candidates)
            if dense_ranking:
                rankings.append(dense_ranking)
                weights.append(1.0)

        non_empty_rankings: list[list[RankedItem]] = []
        non_empty_weights: list[float] = []
        for ranking, weight in zip(rankings, weights, strict=False):
            if ranking:
                non_empty_rankings.append(ranking)
                non_empty_weights.append(weight)
        fused = reciprocal_rank_fusion(non_empty_rankings, weights=non_empty_weights)
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
        retrieval_config = self.settings.retrieval
        reranker = BGEReranker(
            retrieval_config.bge_reranker_model,
            device=retrieval_config.device,
            cache_dir=retrieval_config.cache_dir,
            batch_size=retrieval_config.batch_size,
            max_length=retrieval_config.max_length,
            revision=retrieval_config.model_revision,
            allow_cpu_fallback=retrieval_config.allow_cpu_fallback,
        )
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
        retrieval_config = self.settings.retrieval
        embedder = BGEM3Embedder(
            retrieval_config.bge_m3_model,
            device=retrieval_config.device,
            cache_dir=retrieval_config.cache_dir,
            batch_size=retrieval_config.batch_size,
            max_length=retrieval_config.max_length,
            revision=retrieval_config.model_revision,
            allow_cpu_fallback=retrieval_config.allow_cpu_fallback,
        )
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

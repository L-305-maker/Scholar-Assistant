from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from scholar_assistant.core.config import ScholarSettings, SourceConfig
from scholar_assistant.schemas.paper import AccessType, PaperVersion, VersionType
from scholar_assistant.tools.arxiv import ArxivClient
from scholar_assistant.tools.crossref import CrossrefClient
from scholar_assistant.tools.openalex import OpenAlexClient
from scholar_assistant.tools.semantic_scholar import SemanticScholarClient


class SourceSearchRequest(BaseModel):
    query_id: str
    query: str
    year_start: int | None = None
    year_end: int | None = None
    max_results: int = Field(default=25, ge=1)
    fields: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, gt=0)
    project_id: str | None = None
    run_id: str | None = None


class SourceHit(BaseModel):
    hit_id: str = Field(default_factory=lambda: f"hit_{uuid4().hex[:12]}")
    source: str
    source_id: str | None = None
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    pdf_url: str | None = None
    references: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    raw_rank: int | None = None
    raw_score: float | None = None
    query_id: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_response_ref: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def provenance(self, *, weight: float = 1.0) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "query_id": self.query_id,
            "rank": self.raw_rank,
            "raw_score": self.raw_score,
            "weight": weight,
            "retrieved_at": self.retrieved_at.isoformat(),
        }


class SourceSearchResponse(BaseModel):
    source: str
    query_id: str
    hits: list[SourceHit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    error_type: str | None = None
    result_count: int = 0


class LiteratureSource(Protocol):
    name: str

    async def search(self, request: SourceSearchRequest) -> SourceSearchResponse: ...


class ArxivSource:
    name = "arxiv"

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.client = ArxivClient(timeout_seconds=config.timeout_seconds)

    async def search(self, request: SourceSearchRequest) -> SourceSearchResponse:
        result = await self.client.search(request.query, max_results=request.max_results)
        hits = [
            SourceHit(
                source=self.name,
                source_id=paper.source_ids.get("arxiv") or paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                year=paper.year,
                venue=paper.venue,
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                url=version.source_url if index < len(result.versions) else None,
                pdf_url=paper.pdf_url,
                raw_rank=index + 1,
                query_id=request.query_id,
                raw_response_ref=f"{request.run_id or 'run'}-arxiv.xml",
                metadata={"categories": paper.categories},
            )
            for index, (paper, version) in enumerate(
                zip(result.papers, result.versions, strict=False)
            )
        ]
        return SourceSearchResponse(
            source=self.name,
            query_id=request.query_id,
            hits=hits,
            result_count=len(hits),
        )


class OpenAlexSource:
    name = "openalex"

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.client = OpenAlexClient(timeout_seconds=config.timeout_seconds)

    async def search(self, request: SourceSearchRequest) -> SourceSearchResponse:
        papers = await self.client.search(request.query, max_results=request.max_results)
        hits = [
            SourceHit(
                source=self.name,
                source_id=paper.source_ids.get("openalex"),
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                year=paper.year,
                venue=paper.venue,
                doi=paper.doi,
                url=paper.source_ids.get("openalex"),
                raw_rank=index + 1,
                query_id=request.query_id,
            )
            for index, paper in enumerate(papers)
        ]
        return SourceSearchResponse(
            source=self.name,
            query_id=request.query_id,
            hits=hits,
            result_count=len(hits),
        )


class CrossrefSource:
    name = "crossref"

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.client = CrossrefClient(timeout_seconds=config.timeout_seconds)

    async def search(self, request: SourceSearchRequest) -> SourceSearchResponse:
        papers = await self.client.search(request.query, max_results=request.max_results)
        hits = [
            SourceHit(
                source=self.name,
                source_id=paper.source_ids.get("crossref") or paper.doi,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                year=paper.year,
                venue=paper.venue,
                doi=paper.doi,
                url=f"https://doi.org/{paper.doi}" if paper.doi else None,
                raw_rank=index + 1,
                query_id=request.query_id,
            )
            for index, paper in enumerate(papers)
        ]
        return SourceSearchResponse(
            source=self.name,
            query_id=request.query_id,
            hits=hits,
            result_count=len(hits),
        )


class SemanticScholarSource:
    name = "semantic_scholar"

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self.client = SemanticScholarClient(
            timeout_seconds=config.timeout_seconds,
            api_key=os.environ.get(config.api_key_env or ""),
        )

    async def search(self, request: SourceSearchRequest) -> SourceSearchResponse:
        papers = await self.client.search(request.query, max_results=request.max_results)
        hits = [
            SourceHit(
                source=self.name,
                source_id=paper.source_ids.get("semantic_scholar"),
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                year=paper.year,
                venue=paper.venue,
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                url=paper.metadata.get("url"),
                pdf_url=paper.pdf_url,
                raw_rank=index + 1,
                query_id=request.query_id,
            )
            for index, paper in enumerate(papers)
        ]
        return SourceSearchResponse(
            source=self.name,
            query_id=request.query_id,
            hits=hits,
            result_count=len(hits),
        )


def build_literature_sources(
    settings: ScholarSettings,
    *,
    enabled_names: list[str] | None = None,
) -> dict[str, LiteratureSource]:
    aliases = {"semantic-scholar": "semantic_scholar", "semanticscholar": "semantic_scholar"}
    wanted = None
    if enabled_names:
        wanted = {
            aliases.get(name.strip(), name.strip()).replace("-", "_")
            for name in enabled_names
        }
    factories: dict[str, type] = {
        "arxiv": ArxivSource,
        "openalex": OpenAlexSource,
        "crossref": CrossrefSource,
        "semantic_scholar": SemanticScholarSource,
    }
    sources: dict[str, LiteratureSource] = {}
    for name, config in settings.sources.items():
        normalized = aliases.get(name, name).replace("-", "_")
        if wanted is not None and normalized not in wanted:
            continue
        if not config.enabled:
            continue
        factory = factories.get(normalized)
        if factory is None:
            continue
        sources[normalized] = factory(config)
    return sources


def hit_to_paper_and_version(hit: SourceHit) -> tuple[Any, PaperVersion]:
    from scholar_assistant.schemas.paper import Paper

    paper = Paper(
        title=hit.title,
        authors=hit.authors,
        abstract=hit.abstract,
        year=hit.year,
        venue=hit.venue,
        doi=hit.doi,
        arxiv_id=hit.arxiv_id,
        source_ids={hit.source: hit.source_id} if hit.source_id else {},
        source=hit.source,
        pdf_url=hit.pdf_url,
        categories=list(hit.metadata.get("categories", [])),
        retrieval_provenance=[hit.provenance()],
        metadata={
            "retrieval_provenance": [hit.provenance()],
            "source_url": hit.url,
            "references": hit.references,
            "citations": hit.citations,
            **hit.metadata,
        },
    )
    version_type = {
        "arxiv": VersionType.ARXIV,
        "crossref": VersionType.JOURNAL,
    }.get(hit.source, VersionType.UNKNOWN)
    version = PaperVersion(
        work_id=paper.work_id,
        version_type=version_type,
        source_url=hit.url,
        access_type=AccessType.FULLTEXT if hit.pdf_url else AccessType.ABSTRACT_ONLY,
        is_canonical=hit.source == "crossref",
        is_latest=hit.source in {"arxiv", "semantic_scholar"},
    )
    return paper, version

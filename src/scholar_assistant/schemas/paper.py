from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class PaperRole(StrEnum):
    FOUNDATION = "foundation"
    MAINSTREAM = "mainstream"
    RECENT = "recent"
    BENCHMARK = "benchmark"
    CRITICAL = "critical"
    OTHER = "other"


class VersionType(StrEnum):
    ARXIV = "arxiv"
    CONFERENCE = "conference"
    JOURNAL = "journal"
    LOCAL_PDF = "local_pdf"
    DEMO = "demo"
    UNKNOWN = "unknown"


class AccessType(StrEnum):
    FULLTEXT = "fulltext"
    ABSTRACT_ONLY = "abstract_only"
    METADATA_ONLY = "metadata_only"


class ParseStatus(StrEnum):
    NOT_PARSED = "not_parsed"
    PARSED = "parsed"
    ABSTRACT_ONLY = "abstract_only"
    FAILED = "failed"


class ScholarlyWork(BaseModel):
    work_id: str = Field(default_factory=lambda: f"wk_{uuid4().hex[:12]}")
    canonical_title: str
    normalized_title: str
    doi: str | None = None
    arxiv_id: str | None = None
    source_ids: dict[str, str] = Field(default_factory=dict)
    canonical_version_id: str | None = None
    latest_version_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str | None] = Field(default_factory=dict)
    metadata_provenance: dict[str, str] = Field(default_factory=dict)
    retrieval_provenance: list[dict[str, Any]] = Field(default_factory=list)
    merge_reasons: list[str] = Field(default_factory=list)
    merge_confidence: float = 1.0
    conflicts: list[str] = Field(default_factory=list)


class Paper(BaseModel):
    work_id: str = Field(default_factory=lambda: f"wk_{uuid4().hex[:12]}")
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_ids: dict[str, str] = Field(default_factory=dict)
    paper_role: PaperRole = PaperRole.OTHER
    relevance_score: float = 0.0
    categories: list[str] = Field(default_factory=list)
    source: str = "unknown"
    pdf_url: str | None = None
    retrieval_provenance: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperVersion(BaseModel):
    version_id: str = Field(default_factory=lambda: f"ver_{uuid4().hex[:12]}")
    work_id: str
    version_type: VersionType
    source_url: str | None = None
    local_path: Path | None = None
    content_hash: str | None = None
    access_type: AccessType = AccessType.METADATA_ONLY
    parse_status: ParseStatus = ParseStatus.NOT_PARSED
    version_label: str | None = None
    page_count: int | None = Field(default=None, ge=1)
    is_canonical: bool = False
    is_latest: bool = False

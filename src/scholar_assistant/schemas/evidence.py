from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class EvidenceType(StrEnum):
    ABSTRACT = "abstract"
    SECTION_TEXT = "section_text"
    TABLE = "table"
    FIGURE = "figure"
    METADATA = "metadata"


class SourceType(StrEnum):
    PDF_FULLTEXT = "pdf_fulltext"
    ABSTRACT_ONLY = "abstract_only"
    API_METADATA = "api_metadata"
    USER_IMPORT = "user_import"


class VerificationStatus(StrEnum):
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    ABSTRACT_ONLY = "abstract_only"


class ClaimType(StrEnum):
    PAPER_FACT = "paper_fact"
    AUTHOR_CLAIM = "author_claim"
    CROSS_PAPER_SYNTHESIS = "cross_paper_synthesis"
    AGENT_INFERENCE = "agent_inference"
    RESEARCH_HYPOTHESIS = "research_hypothesis"


class SupportStatus(StrEnum):
    VERIFIED = "verified"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTESTED = "contested"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REJECTED = "rejected"


class EvidenceUnit(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    work_id: str
    version_id: str
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    paragraph_index: int | None = Field(default=None, ge=0)
    table_figure_id: str | None = None
    content: str
    evidence_type: EvidenceType
    source_type: SourceType
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    verification_status: VerificationStatus = VerificationStatus.PENDING
    content_hash: str


class Claim(BaseModel):
    claim_id: str = Field(default_factory=lambda: f"cl_{uuid4().hex[:12]}")
    content: str
    type: ClaimType
    scope: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    support_status: SupportStatus = SupportStatus.INSUFFICIENT_EVIDENCE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def paper_fact_needs_evidence(self) -> Claim:
        if self.type == ClaimType.PAPER_FACT and not self.supporting_evidence_ids:
            msg = "paper_fact claims must cite at least one evidence unit"
            raise ValueError(msg)
        return self


class HypothesisStatus(StrEnum):
    CANDIDATE = "candidate"
    NEEDS_TEST = "needs_test"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: f"hy_{uuid4().hex[:12]}")
    content: str
    motivation: str
    supporting_claims: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    generated_queries: list[str] = Field(default_factory=list)
    falsification_condition: str
    minimum_experiment: str
    risks: list[str] = Field(default_factory=list)
    status: HypothesisStatus = HypothesisStatus.CANDIDATE

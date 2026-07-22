from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from scholar_assistant.agents.reader import Reader
from scholar_assistant.core.events import EventSink
from scholar_assistant.core.quality_gate import QualityGate, QualityGateError
from scholar_assistant.schemas.evidence import (
    Claim,
    ClaimType,
    EvidenceType,
    EvidenceUnit,
    SourceType,
)
from scholar_assistant.schemas.paper import Paper, PaperVersion, VersionType
from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout, sha256_text
from scholar_assistant.storage.repositories import ScholarRepository


def test_storage_init_crud_and_idempotent_migration(tmp_path: Path) -> None:
    ensure_project_layout(tmp_path)
    db_path = tmp_path / ".scholar" / "state.db"
    with Database(db_path) as connection:
        repository = ScholarRepository(connection)
        paper = repository.upsert_paper(Paper(title="A Paper", abstract="Evidence text"))
        assert repository.get_paper(paper.work_id) is not None
    with Database(db_path) as connection:
        repository = ScholarRepository(connection)
        assert repository.get_paper(paper.work_id) is not None


def test_evidence_claim_relation_and_quality_gate() -> None:
    evidence = EvidenceUnit(
        work_id="wk_1",
        version_id="ver_1",
        content="The method reports a retrieval filter.",
        evidence_type=EvidenceType.SECTION_TEXT,
        source_type=SourceType.PDF_FULLTEXT,
        content_hash=sha256_text("The method reports a retrieval filter."),
    )
    claim = Claim(
        content="The paper reports a retrieval filter.",
        type=ClaimType.PAPER_FACT,
        supporting_evidence_ids=[evidence.evidence_id],
    )
    assert QualityGate({evidence.evidence_id: evidence}).validate_claim(claim)
    missing = Claim(
        content="The paper reports a retrieval filter.",
        type=ClaimType.PAPER_FACT,
        supporting_evidence_ids=[evidence.evidence_id],
    )
    with pytest.raises(QualityGateError):
        QualityGate({}).validate_claim(missing)
    with pytest.raises(ValueError):
        Claim(content="Unsupported fact.", type=ClaimType.PAPER_FACT)


def test_quality_gate_validates_version_page_and_direct_ranking() -> None:
    evidence = EvidenceUnit(
        work_id="wk_1",
        version_id="ver_1",
        page=3,
        content="Page evidence.",
        evidence_type=EvidenceType.SECTION_TEXT,
        source_type=SourceType.PDF_FULLTEXT,
        content_hash=sha256_text("Page evidence."),
    )
    version = PaperVersion(
        version_id="ver_1",
        work_id="wk_1",
        version_type=VersionType.LOCAL_PDF,
        page_count=2,
    )
    claim = Claim(
        content="The paper contains page evidence.",
        type=ClaimType.PAPER_FACT,
        supporting_evidence_ids=[evidence.evidence_id],
    )
    with pytest.raises(QualityGateError):
        QualityGate({evidence.evidence_id: evidence}, {"ver_1": version}).validate_claim(claim)

    ranking = Claim(
        content="Method A outperforms Method B across papers.",
        type=ClaimType.CROSS_PAPER_SYNTHESIS,
        supporting_evidence_ids=[evidence.evidence_id],
    )
    evidence.page = 1
    with pytest.raises(QualityGateError):
        QualityGate({evidence.evidence_id: evidence}, {"ver_1": version}).validate_claim(ranking)


@pytest.mark.asyncio
async def test_reader_extracts_pdf_evidence_with_page(tmp_path: Path) -> None:
    ensure_project_layout(tmp_path)
    pdf_path = tmp_path / "papers" / "sample.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Introduction\n"
        "This paper studies memory retrieval noise in LLM agents with controlled evidence.",
    )
    document.save(pdf_path)
    document.close()

    with Database(tmp_path / ".scholar" / "state.db") as connection:
        repository = ScholarRepository(connection)
        reader = Reader(repository, tmp_path, EventSink(), run_id="test")
        evidence = await reader.read_paper(pdf_path)

    assert evidence
    assert evidence[0].page == 1
    assert evidence[0].source_type == SourceType.PDF_FULLTEXT

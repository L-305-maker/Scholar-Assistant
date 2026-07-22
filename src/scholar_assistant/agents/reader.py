from __future__ import annotations

from pathlib import Path

import httpx

from scholar_assistant.core.budget import BudgetLimitExceeded, BudgetManager
from scholar_assistant.core.events import EventSink
from scholar_assistant.schemas.events import RunEvent, RunEventType
from scholar_assistant.schemas.evidence import (
    EvidenceType,
    EvidenceUnit,
    SourceType,
    VerificationStatus,
)
from scholar_assistant.schemas.paper import (
    AccessType,
    Paper,
    PaperVersion,
    ParseStatus,
    VersionType,
)
from scholar_assistant.storage.files import sha256_text
from scholar_assistant.storage.repositories import ScholarRepository
from scholar_assistant.tools.pdf_parser import PyMuPDFParser
from scholar_assistant.tools.registry import (
    RetryPolicy,
    ToolExecutionContext,
    ToolExecutor,
    ToolPermissionLevel,
    ToolRegistry,
    ToolSpec,
)


class Reader:
    def __init__(
        self,
        repository: ScholarRepository,
        project_path: Path,
        event_sink: EventSink,
        *,
        run_id: str,
        budget_manager: BudgetManager | None = None,
    ) -> None:
        self.repository = repository
        self.project_path = project_path
        self.event_sink = event_sink
        self.run_id = run_id
        self.parser = PyMuPDFParser()
        self.tool_registry = ToolRegistry()
        self._register_tools()
        self.tool_executor = ToolExecutor(
            self.tool_registry,
            ToolExecutionContext(run_id=run_id, budget=budget_manager),
        )
        self._saved_execution_count = 0

    async def read_paper(
        self, paper_or_path: Paper | str | Path, *, max_paragraphs: int = 8
    ) -> list[EvidenceUnit]:
        if isinstance(paper_or_path, Paper):
            return await self._read_known_paper(paper_or_path, max_paragraphs=max_paragraphs)
        path = Path(paper_or_path)
        if path.exists():
            return await self._read_local_pdf(path, max_paragraphs=max_paragraphs)
        paper = self.repository.get_paper(str(paper_or_path))
        if paper is None:
            msg = f"Paper or PDF not found: {paper_or_path}"
            raise FileNotFoundError(msg)
        return await self._read_known_paper(paper, max_paragraphs=max_paragraphs)

    async def _read_known_paper(self, paper: Paper, *, max_paragraphs: int) -> list[EvidenceUnit]:
        versions = self.repository.list_versions(paper.work_id)
        version = versions[0] if versions else None
        if paper.pdf_url and version:
            try:
                path = self._pdf_path(paper)
                cache_hit = path.exists() and path.stat().st_size > 0
                path = await self._execute_tool(
                    "pdf.download",
                    paper=paper,
                    path=path,
                    cache_hit=cache_hit,
                )
                return await self._read_pdf_for_paper(
                    paper, version, path, max_paragraphs=max_paragraphs
                )
            except BudgetLimitExceeded:
                raise
            except (httpx.HTTPError, OSError, RuntimeError):
                pass
        if version is None:
            version = PaperVersion(
                work_id=paper.work_id,
                version_type=VersionType.UNKNOWN,
                access_type=AccessType.ABSTRACT_ONLY,
                parse_status=ParseStatus.ABSTRACT_ONLY,
            )
            self.repository.upsert_version(version)
        content = paper.abstract or f"Metadata-only record for {paper.title}."
        evidence = EvidenceUnit(
            work_id=paper.work_id,
            version_id=version.version_id,
            section="Abstract",
            page=None,
            paragraph_index=0,
            content=content,
            evidence_type=EvidenceType.ABSTRACT,
            source_type=SourceType.ABSTRACT_ONLY,
            extraction_confidence=0.7 if paper.abstract else 0.3,
            verification_status=VerificationStatus.ABSTRACT_ONLY,
            content_hash=sha256_text(content),
        )
        self.repository.upsert_evidence(evidence)
        self.event_sink.emit(
            RunEvent.new(
                RunEventType.EVIDENCE_CREATED,
                run_id=self.run_id,
                payload={
                    "evidence_id": evidence.evidence_id,
                    "work_id": paper.work_id,
                    "source_type": evidence.source_type.value,
                },
            )
        )
        self.event_sink.emit(
            RunEvent.new(
                RunEventType.PAPER_READ,
                run_id=self.run_id,
                payload={"work_id": paper.work_id, "parse_status": "abstract_only"},
            )
        )
        return [evidence]

    async def _read_local_pdf(self, path: Path, *, max_paragraphs: int) -> list[EvidenceUnit]:
        paper = Paper(title=path.stem, source="local_pdf")
        paper = self.repository.upsert_paper(paper)
        version = PaperVersion(
            work_id=paper.work_id,
            version_type=VersionType.LOCAL_PDF,
            local_path=path.resolve(),
            access_type=AccessType.FULLTEXT,
            parse_status=ParseStatus.PARSED,
            is_canonical=True,
            is_latest=True,
        )
        self.repository.upsert_version(version)
        return await self._read_pdf_for_paper(
            paper, version, path, max_paragraphs=max_paragraphs
        )

    async def _read_pdf_for_paper(
        self,
        paper: Paper,
        version: PaperVersion,
        path: Path,
        *,
        max_paragraphs: int,
    ) -> list[EvidenceUnit]:
        parsed = await self._execute_tool("pdf.parse", path=path)
        version.local_path = path.resolve()
        version.content_hash = parsed.content_hash
        version.parse_status = ParseStatus.PARSED
        version.access_type = AccessType.FULLTEXT
        version.page_count = len(parsed.pages)
        self.repository.upsert_version(version)
        evidence_units: list[EvidenceUnit] = []
        paragraphs = [
            para for para in parsed.paragraphs if not para.is_reference and len(para.text) > 40
        ]
        for paragraph in paragraphs[:max_paragraphs]:
            evidence = EvidenceUnit(
                work_id=paper.work_id,
                version_id=version.version_id,
                section=paragraph.section,
                page=paragraph.page,
                paragraph_index=paragraph.paragraph_index,
                table_figure_id=paragraph.caption_type,
                content=paragraph.text,
                evidence_type=EvidenceType.SECTION_TEXT,
                source_type=SourceType.PDF_FULLTEXT,
                extraction_confidence=0.85,
                verification_status=VerificationStatus.VERIFIED,
                content_hash=sha256_text(paragraph.text),
            )
            self.repository.upsert_evidence(evidence)
            evidence_units.append(evidence)
            self.event_sink.emit(
                RunEvent.new(
                    RunEventType.EVIDENCE_CREATED,
                    run_id=self.run_id,
                    payload={
                        "evidence_id": evidence.evidence_id,
                        "work_id": paper.work_id,
                        "page": evidence.page,
                        "section": evidence.section,
                    },
                )
            )
        self.event_sink.emit(
            RunEvent.new(
                RunEventType.PAPER_READ,
                run_id=self.run_id,
                payload={
                    "work_id": paper.work_id,
                    "parse_status": "parsed",
                    "evidence": len(evidence_units),
                    "reading_summary": build_reading_summary(evidence_units),
                },
            )
        )
        paper.metadata["reading_summary"] = build_reading_summary(evidence_units)
        self.repository.upsert_paper(paper)
        return evidence_units

    async def _download_pdf(self, paper: Paper, path: Path, cache_hit: bool = False) -> Path:
        if not paper.pdf_url:
            msg = "Paper has no PDF URL"
            raise RuntimeError(msg)
        if cache_hit and path.exists() and path.stat().st_size > 0:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(paper.pdf_url)
            response.raise_for_status()
        path.write_bytes(response.content)
        return path

    def _parse_pdf(self, path: Path):
        return self.parser.parse(path)

    def _pdf_path(self, paper: Paper) -> Path:
        filename = f"{paper.arxiv_id or paper.work_id}.pdf".replace("/", "_")
        return self.project_path / "papers" / filename

    async def _execute_tool(self, name: str, **kwargs):
        try:
            return await self.tool_executor.execute(name, **kwargs)
        finally:
            self._flush_tool_executions()

    def _flush_tool_executions(self) -> None:
        new_executions = self.tool_executor.executions[self._saved_execution_count :]
        for execution in new_executions:
            self.repository.save_tool_execution(execution)
        self._saved_execution_count += len(new_executions)

    def _register_tools(self) -> None:
        self.tool_registry.register(
            ToolSpec(
                name="pdf.download",
                description="Download an open-access PDF into the project papers directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "paper": {"type": "object"},
                        "path": {"type": "string"},
                        "cache_hit": {"type": "boolean"},
                    },
                    "required": ["paper", "path"],
                },
                handler=self._download_pdf,
                category="reader",
                permission_level=ToolPermissionLevel.T2_PROJECT_WRITE,
                timeout_seconds=60.0,
                retry_policy=RetryPolicy(max_retries=1, backoff_seconds=0.5),
                cost_unit="pdf",
                idempotent=True,
                network_domains=["paper.pdf_url"],
                side_effect_level="project_write",
                write_allowed=True,
            )
        )
        self.tool_registry.register(
            ToolSpec(
                name="pdf.parse",
                description="Parse a local PDF into page-level structured text.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                handler=self._parse_pdf,
                category="reader",
                permission_level=ToolPermissionLevel.T0_LOCAL_READ,
                timeout_seconds=90.0,
                retry_policy=RetryPolicy(max_retries=0),
                cost_unit="pdf",
                idempotent=True,
                network_domains=[],
                side_effect_level="none",
            )
        )


def build_reading_summary(evidence_units: list[EvidenceUnit]) -> dict[str, list[str]]:
    fields = {
        "research_questions": [],
        "methods": [],
        "datasets": [],
        "metrics": [],
        "baselines": [],
        "results": [],
        "limitations": [],
    }
    patterns = {
        "research_questions": ["research question", "we ask", "aim", "objective"],
        "methods": ["method", "approach", "framework", "model", "algorithm"],
        "datasets": ["dataset", "benchmark", "corpus", "data set"],
        "metrics": ["metric", "accuracy", "f1", "precision", "recall", "score"],
        "baselines": ["baseline", "compare", "comparison", "versus"],
        "results": ["result", "outperform", "improve", "performance", "achieve"],
        "limitations": ["limitation", "future work", "fail", "noise", "error"],
    }
    for evidence in evidence_units:
        text = evidence.content.strip()
        lower = text.lower()
        for field, keywords in patterns.items():
            if any(keyword in lower for keyword in keywords):
                fields[field].append(text[:400])
    return {field: values[:5] for field, values in fields.items()}

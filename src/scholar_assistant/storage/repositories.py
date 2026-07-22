from __future__ import annotations

import json
import re
import sqlite3
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from scholar_assistant.schemas.events import RunEvent
from scholar_assistant.schemas.evidence import Claim, EvidenceUnit, Hypothesis
from scholar_assistant.schemas.paper import Paper, PaperVersion, ScholarlyWork
from scholar_assistant.schemas.research import ResearchProject


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


class ScholarRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def upsert_project(self, project: ResearchProject) -> None:
        data = project.model_dump_json()
        self.connection.execute(
            """
            INSERT INTO projects(project_id, data, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(project_id) DO UPDATE SET
                data = excluded.data,
                updated_at = excluded.updated_at
            """,
            (
                project.project_id,
                data,
                project.created_at.isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )

    def upsert_paper(self, paper: Paper, version: PaperVersion | None = None) -> Paper:
        normalized = normalize_title(paper.title)
        existing = self._find_existing_work(paper, normalized)
        if existing:
            paper.work_id = existing["work_id"]
        work = ScholarlyWork(
            work_id=paper.work_id,
            canonical_title=paper.title,
            normalized_title=normalized,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            source_ids=paper.source_ids,
        )
        self.connection.execute(
            """
            INSERT INTO works(work_id, data, doi, arxiv_id, normalized_title)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(work_id) DO UPDATE SET
                data = excluded.data,
                doi = COALESCE(excluded.doi, works.doi),
                arxiv_id = COALESCE(excluded.arxiv_id, works.arxiv_id)
            ON CONFLICT(normalized_title) DO UPDATE SET
                data = excluded.data
            """,
            (work.work_id, work.model_dump_json(), work.doi, work.arxiv_id, work.normalized_title),
        )
        self.connection.execute(
            """
            INSERT INTO papers(work_id, data, title, abstract, year, doi, arxiv_id, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(work_id) DO UPDATE SET
                data = excluded.data,
                title = excluded.title,
                abstract = excluded.abstract,
                year = excluded.year,
                doi = COALESCE(excluded.doi, papers.doi),
                arxiv_id = COALESCE(excluded.arxiv_id, papers.arxiv_id),
                source = excluded.source
            """,
            (
                paper.work_id,
                paper.model_dump_json(),
                paper.title,
                paper.abstract,
                paper.year,
                paper.doi,
                paper.arxiv_id,
                paper.source,
            ),
        )
        self.connection.execute("DELETE FROM papers_fts WHERE work_id = ?", (paper.work_id,))
        self.connection.execute(
            "INSERT INTO papers_fts(work_id, title, abstract, authors) VALUES (?, ?, ?, ?)",
            (paper.work_id, paper.title, paper.abstract or "", " ".join(paper.authors)),
        )
        if version is not None:
            version.work_id = paper.work_id
            self.upsert_version(version)
        self.connection.commit()
        return paper

    def _find_existing_work(self, paper: Paper, normalized: str) -> sqlite3.Row | None:
        if paper.doi:
            row = self.connection.execute(
                "SELECT * FROM works WHERE doi = ?", (paper.doi,)
            ).fetchone()
            if row:
                return row
        if paper.arxiv_id:
            row = self.connection.execute(
                "SELECT * FROM works WHERE arxiv_id = ?", (paper.arxiv_id,)
            ).fetchone()
            if row:
                return row
        row = self.connection.execute(
            "SELECT * FROM works WHERE normalized_title = ?", (normalized,)
        ).fetchone()
        if row:
            return row
        for candidate in self.connection.execute("SELECT * FROM works").fetchall():
            if title_similarity(normalized, candidate["normalized_title"]) > 0.94:
                return candidate
        return None

    def upsert_version(self, version: PaperVersion) -> None:
        self.connection.execute(
            """
            INSERT INTO paper_versions(version_id, work_id, data, source_url, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(version_id) DO UPDATE SET
                data = excluded.data,
                source_url = excluded.source_url,
                content_hash = excluded.content_hash
            """,
            (
                version.version_id,
                version.work_id,
                version.model_dump_json(),
                version.source_url,
                version.content_hash,
            ),
        )
        self.connection.commit()

    def list_papers(self) -> list[Paper]:
        rows = self.connection.execute(
            "SELECT data FROM papers ORDER BY year DESC NULLS LAST"
        ).fetchall()
        return [Paper.model_validate_json(row["data"]) for row in rows]

    def get_paper(self, work_id: str) -> Paper | None:
        row = self.connection.execute(
            "SELECT data FROM papers WHERE work_id = ?", (work_id,)
        ).fetchone()
        return Paper.model_validate_json(row["data"]) if row else None

    def list_versions(self, work_id: str) -> list[PaperVersion]:
        rows = self.connection.execute(
            "SELECT data FROM paper_versions WHERE work_id = ?", (work_id,)
        ).fetchall()
        return [PaperVersion.model_validate_json(row["data"]) for row in rows]

    def upsert_evidence(self, evidence: EvidenceUnit) -> EvidenceUnit:
        self.connection.execute(
            """
            INSERT INTO evidence(evidence_id, work_id, version_id, data, content_hash)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(evidence_id) DO UPDATE SET data = excluded.data
            """,
            (
                evidence.evidence_id,
                evidence.work_id,
                evidence.version_id,
                evidence.model_dump_json(),
                evidence.content_hash,
            ),
        )
        self.connection.commit()
        return evidence

    def list_evidence(self, work_id: str | None = None) -> list[EvidenceUnit]:
        if work_id:
            rows = self.connection.execute(
                "SELECT data FROM evidence WHERE work_id = ?", (work_id,)
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT data FROM evidence").fetchall()
        return [EvidenceUnit.model_validate_json(row["data"]) for row in rows]

    def upsert_claim(self, claim: Claim) -> Claim:
        self.connection.execute(
            """
            INSERT INTO claims(claim_id, data, type, support_status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(claim_id) DO UPDATE SET
                data = excluded.data,
                support_status = excluded.support_status
            """,
            (
                claim.claim_id,
                claim.model_dump_json(),
                claim.type.value,
                claim.support_status.value,
            ),
        )
        self.connection.commit()
        return claim

    def list_claims(self) -> list[Claim]:
        rows = self.connection.execute("SELECT data FROM claims").fetchall()
        return [Claim.model_validate_json(row["data"]) for row in rows]

    def upsert_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        self.connection.execute(
            """
            INSERT INTO hypotheses(hypothesis_id, data, status)
            VALUES (?, ?, ?)
            ON CONFLICT(hypothesis_id) DO UPDATE SET data = excluded.data, status = excluded.status
            """,
            (hypothesis.hypothesis_id, hypothesis.model_dump_json(), hypothesis.status.value),
        )
        self.connection.commit()
        return hypothesis

    def list_hypotheses(self) -> list[Hypothesis]:
        rows = self.connection.execute("SELECT data FROM hypotheses").fetchall()
        return [Hypothesis.model_validate_json(row["data"]) for row in rows]

    def save_event(self, event: RunEvent) -> None:
        self.connection.execute(
            """
            INSERT INTO run_events(run_id, event_type, timestamp, data)
            VALUES (?, ?, ?, ?)
            """,
            (
                event.run_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.model_dump_json(),
            ),
        )
        self.connection.commit()

    def save_run(self, run_id: str, status: str, question: str, data: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO runs(run_id, status, question, data)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                status = excluded.status,
                data = excluded.data,
                updated_at = CURRENT_TIMESTAMP
            """,
            (run_id, status, question, json.dumps(data, ensure_ascii=False, default=str)),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return {
            "run_id": row["run_id"],
            "status": row["status"],
            "question": row["question"],
            "data": json.loads(row["data"]),
            "updated_at": row["updated_at"],
        }

    def latest_run(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM runs ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return self.get_run(row["run_id"])

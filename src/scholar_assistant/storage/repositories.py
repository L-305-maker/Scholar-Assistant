from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

from scholar_assistant.schemas.events import RunEvent
from scholar_assistant.schemas.evidence import Claim, EvidenceUnit, Hypothesis
from scholar_assistant.schemas.paper import Paper, PaperVersion, ScholarlyWork
from scholar_assistant.schemas.research import ResearchProject
from scholar_assistant.storage.canonicalization import (
    decide_duplicate,
    normalize_arxiv_base,
    normalize_doi,
    normalize_source_id,
    normalize_title,
    title_similarity,
)


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
        paper.doi = normalize_doi(paper.doi)
        arxiv_base_id = normalize_arxiv_base(paper.arxiv_id)
        if arxiv_base_id:
            paper.metadata["arxiv_base_id"] = arxiv_base_id
        normalized = normalize_title(paper.title)
        existing = self._find_existing_work(paper, normalized)
        if existing:
            paper.work_id = existing["work_id"]
            paper = self._merge_existing_paper(paper)
        work = ScholarlyWork(
            work_id=paper.work_id,
            canonical_title=paper.title,
            normalized_title=normalized,
            doi=paper.doi,
            arxiv_id=arxiv_base_id or paper.arxiv_id,
            source_ids=paper.source_ids,
            aliases=[paper.title],
            identifiers={
                "doi": paper.doi,
                "arxiv_base_id": arxiv_base_id,
                **paper.source_ids,
            },
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
        self._upsert_identifiers(paper, arxiv_base_id)
        self._upsert_alias(paper.work_id, "title", paper.title, normalized, paper.source)
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
        doi = normalize_doi(paper.doi)
        arxiv_base = normalize_arxiv_base(paper.arxiv_id)
        if doi:
            row = self.connection.execute(
                "SELECT * FROM works WHERE doi = ?", (doi,)
            ).fetchone()
            if row:
                return row
        if arxiv_base:
            row = self.connection.execute(
                "SELECT * FROM works WHERE arxiv_id = ?", (arxiv_base,)
            ).fetchone()
            if row:
                return row
        for source, source_id in paper.source_ids.items():
            namespaced = normalize_source_id(source, source_id)
            if not namespaced:
                continue
            row = self.connection.execute(
                """
                SELECT works.*
                FROM work_identifiers
                JOIN works USING(work_id)
                WHERE work_identifiers.normalized_identifier = ?
                """,
                (namespaced,),
            ).fetchone()
            if row:
                return row
        row = self.connection.execute(
            "SELECT * FROM works WHERE normalized_title = ?", (normalized,)
        ).fetchone()
        if row:
            return row
        for candidate in self.connection.execute("SELECT * FROM works").fetchall():
            candidate_paper = self.get_paper(candidate["work_id"])
            if candidate_paper is None:
                continue
            decision = decide_duplicate(paper, candidate_paper)
            if decision.action in {"exact_merge", "strong_fuzzy_merge"}:
                return candidate
            if decision.action == "possible_duplicate":
                self.save_duplicate_candidate(
                    paper.work_id,
                    candidate["work_id"],
                    decision.action,
                    decision.reason,
                    decision.confidence,
                    decision.signals,
                )
            elif (
                title_similarity(normalized, candidate["normalized_title"]) > 0.985
                and decision.action != "never_merge"
            ):
                return candidate
        return None

    def _merge_existing_paper(self, incoming: Paper) -> Paper:
        existing = self.get_paper(incoming.work_id)
        if existing is None:
            return incoming
        incoming.source_ids = {**existing.source_ids, **incoming.source_ids}
        aliases = {existing.title, incoming.title, *existing.metadata.get("aliases", [])}
        incoming.metadata = {
            **existing.metadata,
            **incoming.metadata,
            "aliases": sorted(aliases),
        }
        if existing.doi and not incoming.doi:
            incoming.doi = existing.doi
        if existing.arxiv_id and not incoming.arxiv_id:
            incoming.arxiv_id = existing.arxiv_id
        if existing.abstract and (
            not incoming.abstract or len(existing.abstract) > len(incoming.abstract)
        ):
            incoming.abstract = existing.abstract
        if existing.venue and not incoming.venue:
            incoming.venue = existing.venue
        if existing.year and not incoming.year:
            incoming.year = existing.year
        if existing.pdf_url and not incoming.pdf_url:
            incoming.pdf_url = existing.pdf_url
        return incoming

    def _upsert_identifiers(self, paper: Paper, arxiv_base_id: str | None) -> None:
        identifiers: list[tuple[str, str, str]] = []
        if paper.doi:
            identifiers.append(("doi", paper.doi, f"doi:{paper.doi}"))
        if arxiv_base_id:
            identifiers.append(("arxiv", arxiv_base_id, f"arxiv:{arxiv_base_id}"))
        for source, source_id in paper.source_ids.items():
            normalized_source_id = normalize_source_id(source, source_id)
            if normalized_source_id:
                identifiers.append((source, source_id, normalized_source_id))
        for namespace, identifier, normalized_identifier in identifiers:
            self.connection.execute(
                """
                INSERT INTO work_identifiers(
                    work_id, namespace, identifier, normalized_identifier
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, normalized_identifier) DO UPDATE SET
                    work_id = excluded.work_id
                """,
                (paper.work_id, namespace, identifier, normalized_identifier),
            )

    def _upsert_alias(
        self, work_id: str, alias_type: str, value: str, normalized_value: str, source: str | None
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO paper_aliases(work_id, alias_type, value, normalized_value, source)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(work_id, alias_type, normalized_value) DO NOTHING
            """,
            (work_id, alias_type, value, normalized_value, source),
        )

    def save_duplicate_candidate(
        self,
        left_work_id: str,
        right_work_id: str,
        decision: str,
        reason: str,
        confidence: float,
        signals: dict[str, object],
    ) -> None:
        left, right = sorted([left_work_id, right_work_id])
        self.connection.execute(
            """
            INSERT INTO duplicate_candidates(
                left_work_id, right_work_id, decision, reason, confidence, signals
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(left_work_id, right_work_id, decision) DO UPDATE SET
                reason = excluded.reason,
                confidence = excluded.confidence,
                signals = excluded.signals
            """,
            (left, right, decision, reason, confidence, json.dumps(signals, ensure_ascii=False)),
        )

    def list_duplicate_candidates(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM duplicate_candidates").fetchall()
        return [dict(row) for row in rows]

    def save_source_hit(self, hit: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO source_hits(
                hit_id, run_id, query_id, source, source_id, work_id, data, retrieved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hit_id) DO UPDATE SET
                work_id = excluded.work_id,
                data = excluded.data
            """,
            (
                hit["hit_id"],
                hit["run_id"],
                hit["query_id"],
                hit["source"],
                hit.get("source_id"),
                hit.get("work_id"),
                json.dumps(hit, ensure_ascii=False, default=str),
                hit["retrieved_at"],
            ),
        )

    def save_retrieval_provenance(
        self,
        *,
        run_id: str,
        work_id: str,
        provenance: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO retrieval_provenance(
                run_id, work_id, source, source_id, query_id, rank, raw_score, weight, retrieved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, work_id, source, source_id, query_id) DO UPDATE SET
                rank = excluded.rank,
                raw_score = excluded.raw_score,
                weight = excluded.weight,
                retrieved_at = excluded.retrieved_at
            """,
            (
                run_id,
                work_id,
                provenance["source"],
                provenance.get("source_id"),
                provenance["query_id"],
                provenance.get("rank"),
                provenance.get("raw_score"),
                provenance.get("weight", 1.0),
                provenance["retrieved_at"],
            ),
        )

    def list_retrieval_provenance(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM retrieval_provenance WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def save_tool_execution(self, execution: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO tool_executions(
                execution_id, run_id, tool_name, status, started_at, completed_at,
                latency_ms, attempts, error_type, data
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(execution_id) DO UPDATE SET
                status = excluded.status,
                completed_at = excluded.completed_at,
                latency_ms = excluded.latency_ms,
                attempts = excluded.attempts,
                error_type = excluded.error_type,
                data = excluded.data
            """,
            (
                execution["execution_id"],
                execution.get("run_id"),
                execution["tool_name"],
                execution["status"],
                execution["started_at"],
                execution.get("completed_at"),
                execution.get("latency_ms"),
                execution.get("attempts", 1),
                execution.get("error_type"),
                json.dumps(execution, ensure_ascii=False, default=str),
            ),
        )

    def save_budget_usage(self, run_id: str, usage: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO budget_usage(run_id, data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                data = excluded.data,
                updated_at = CURRENT_TIMESTAMP
            """,
            (run_id, json.dumps(usage, ensure_ascii=False, default=str)),
        )

    def get_budget_usage(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT data FROM budget_usage WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

    def save_run_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT INTO run_manifests(run_id, data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(run_id) DO UPDATE SET
                data = excluded.data,
                updated_at = CURRENT_TIMESTAMP
            """,
            (run_id, json.dumps(manifest, ensure_ascii=False, default=str)),
        )

    def get_run_manifest(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT data FROM run_manifests WHERE run_id = ?", (run_id,)
        ).fetchone()
        return json.loads(row["data"]) if row else None

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

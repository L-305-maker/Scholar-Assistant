from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from scholar_assistant.agents.analyst import Analyst
from scholar_assistant.agents.reader import Reader
from scholar_assistant.agents.searcher import Searcher
from scholar_assistant.agents.verifier import Verifier
from scholar_assistant.core.budget import BudgetLimitExceeded, BudgetManager
from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.core.events import EventSink
from scholar_assistant.core.reporting import render_report
from scholar_assistant.core.state_machine import ResearchState, ResearchStateMachine
from scholar_assistant.schemas.events import RunEvent, RunEventType
from scholar_assistant.schemas.evidence import Claim, EvidenceUnit, Hypothesis
from scholar_assistant.schemas.paper import Paper
from scholar_assistant.storage.database import Database
from scholar_assistant.storage.files import ensure_project_layout, run_dir, write_json
from scholar_assistant.storage.repositories import ScholarRepository


@dataclass(slots=True)
class ResearchRunResult:
    run_id: str
    status: ResearchState
    question: str
    run_path: Path
    papers: list[Paper] = field(default_factory=list)
    evidence_units: list[EvidenceUnit] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retrieval_mode: str = "unknown"

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "question": self.question,
            "run_path": str(self.run_path),
            "papers": len(self.papers),
            "evidence": len(self.evidence_units),
            "claims": len(self.claims),
            "hypotheses": len(self.hypotheses),
            "retrieval_mode": self.retrieval_mode,
            "warnings": self.warnings,
        }


class ResearchOrchestrator:
    def __init__(self, project_path: Path) -> None:
        self.project_path = project_path.resolve()
        ensure_project_layout(self.project_path)
        self.settings = ScholarSettings.load(self.project_path)
        self.database = Database(self.project_path / ".scholar" / "state.db")

    async def run_research(
        self,
        question: str,
        *,
        run_id: str | None = None,
        no_embeddings: bool = False,
        sources: list[str] | None = None,
        max_candidates: int | None = None,
        max_deep_reads: int | None = None,
    ) -> ResearchRunResult:
        run_id = run_id or f"run_{uuid4().hex[:12]}"
        path = run_dir(self.project_path, run_id)
        event_sink = EventSink(path / "events.jsonl")
        machine = ResearchStateMachine()
        warnings: list[str] = []
        settings = self.settings.model_copy(deep=True)
        if max_candidates is not None:
            settings.budget.max_raw_candidates = max_candidates
        if max_deep_reads is not None:
            settings.budget.max_deep_read = max_deep_reads
        budget_manager = BudgetManager(settings.budget)
        with self.database as connection:
            repository = ScholarRepository(connection)
            repository.save_run(
                run_id, machine.state.value, question, {"state": machine.state.value}
            )
            event = event_sink.emit(
                RunEvent.new(
                    RunEventType.RUN_STARTED,
                    run_id=run_id,
                    payload={"question": question, "project_path": str(self.project_path)},
                )
            )
            repository.save_event(event)
            saved_event_count = len(event_sink.events)

            def flush_events() -> None:
                nonlocal saved_event_count
                for pending_event in event_sink.events[saved_event_count:]:
                    repository.save_event(pending_event)
                saved_event_count = len(event_sink.events)

            def emit_task_event(
                event_type: RunEventType,
                task_id: str,
                role: str,
                payload: dict[str, object] | None = None,
            ) -> None:
                event_sink.emit(
                    RunEvent.new(
                        event_type,
                        run_id=run_id,
                        task_id=task_id,
                        payload={
                            "task_type": task_id,
                            "assigned_role": role,
                            **(payload or {}),
                        },
                    )
                )
                flush_events()

            try:
                machine.transition(ResearchState.SCOPING)
                machine.transition(ResearchState.SEARCH_PLANNING)
                machine.transition(ResearchState.SEARCHING)
                emit_task_event(RunEventType.TASK_STARTED, "search", "Searcher")
                searcher = Searcher(
                    repository,
                    settings,
                    self.project_path,
                    event_sink,
                    run_id=run_id,
                    budget_manager=budget_manager,
                    enabled_sources=sources,
                )
                search_result = await searcher.search(
                    question, no_embeddings=no_embeddings, sources=sources
                )
                warnings.extend(search_result.warnings)
                for warning in search_result.warnings:
                    event_sink.emit(
                        RunEvent.new(
                            RunEventType.WARNING,
                            run_id=run_id,
                            task_id="search",
                            payload={"message": warning},
                        )
                    )
                emit_task_event(
                    RunEventType.TASK_COMPLETED,
                    "search",
                    "Searcher",
                    {
                        "papers": len(search_result.papers),
                        "retrieval_mode": search_result.retrieval_mode,
                    },
                )
                flush_events()

                if not search_result.papers:
                    machine.transition(ResearchState.PARTIALLY_COMPLETED)
                    return self._finalize(
                        repository=repository,
                        event_sink=event_sink,
                        machine=machine,
                        run_id=run_id,
                        question=question,
                        run_path=path,
                        papers=[],
                        evidence_units=[],
                        claims=[],
                        hypotheses=[],
                        retrieval_mode=search_result.retrieval_mode,
                        warnings=[*warnings, "No papers were selected."],
                    )

                machine.transition(ResearchState.SCREENING)
                machine.transition(ResearchState.READING)
                emit_task_event(RunEventType.TASK_STARTED, "read", "Reader")
                reader = Reader(
                    repository,
                    self.project_path,
                    event_sink,
                    run_id=run_id,
                    budget_manager=budget_manager,
                )
                evidence_units: list[EvidenceUnit] = []
                for paper in search_result.papers[: settings.budget.max_deep_read]:
                    if budget_manager.deep_reads >= settings.budget.max_deep_read:
                        raise BudgetLimitExceeded(
                            "deep read budget exhausted",
                            counter="deep_reads",
                        )
                    budget_manager.deep_reads += 1
                    evidence_units.extend(await reader.read_paper(paper, max_paragraphs=3))
                emit_task_event(
                    RunEventType.TASK_COMPLETED,
                    "read",
                    "Reader",
                    {"evidence": len(evidence_units)},
                )

                machine.transition(ResearchState.ANALYZING)
                emit_task_event(RunEventType.TASK_STARTED, "analyze", "Analyst")
                analyst = Analyst(repository, event_sink, run_id=run_id)
                claims, hypotheses = analyst.analyze(
                    evidence_units,
                    question=question,
                    max_claims=settings.budget.max_core_claims,
                    max_hypotheses=settings.budget.max_hypotheses,
                )
                emit_task_event(
                    RunEventType.TASK_COMPLETED,
                    "analyze",
                    "Analyst",
                    {"claims": len(claims), "hypotheses": len(hypotheses)},
                )

                machine.transition(ResearchState.VERIFYING)
                emit_task_event(RunEventType.TASK_STARTED, "verify", "Verifier")
                verifier = Verifier()
                versions = [
                    version
                    for paper in search_result.papers
                    for version in repository.list_versions(paper.work_id)
                ]
                verified_claims = verifier.verify_claims(claims, evidence_units, versions)
                for claim in verified_claims:
                    repository.upsert_claim(claim)
                for hypothesis in hypotheses:
                    repository.upsert_hypothesis(hypothesis)
                emit_task_event(
                    RunEventType.TASK_COMPLETED,
                    "verify",
                    "Verifier",
                    {"verified_claims": len(verified_claims)},
                )

                machine.transition(ResearchState.REPORTING)
                emit_task_event(RunEventType.TASK_STARTED, "report", "Reporter")
                machine.transition(ResearchState.COMPLETED)
                emit_task_event(RunEventType.TASK_COMPLETED, "report", "Reporter")
                return self._finalize(
                    repository=repository,
                    event_sink=event_sink,
                    machine=machine,
                    run_id=run_id,
                    question=question,
                    run_path=path,
                    papers=search_result.papers,
                    evidence_units=evidence_units,
                    claims=verified_claims,
                    hypotheses=hypotheses,
                    retrieval_mode=search_result.retrieval_mode,
                    warnings=warnings,
                    query_plan=search_result.query_plan,
                    budget_manager=budget_manager,
                    source_stats=search_result.source_stats,
                )
            except BudgetLimitExceeded as exc:
                machine.state = ResearchState.BUDGET_EXHAUSTED
                warning = f"Budget exhausted: {exc}"
                warnings.append(warning)
                event_sink.emit(
                    RunEvent.new(
                        RunEventType.WARNING,
                        run_id=run_id,
                        payload={"message": warning, "counter": exc.counter},
                    )
                )
                flush_events()
                return self._finalize(
                    repository=repository,
                    event_sink=event_sink,
                    machine=machine,
                    run_id=run_id,
                    question=question,
                    run_path=path,
                    papers=[],
                    evidence_units=[],
                    claims=[],
                    hypotheses=[],
                    retrieval_mode="budget-exhausted",
                    warnings=warnings,
                    budget_manager=budget_manager,
                    source_stats={},
                )
            except Exception as exc:
                machine.state = ResearchState.FAILED
                error = event_sink.emit(
                    RunEvent.new(
                        RunEventType.ERROR,
                        run_id=run_id,
                        payload={"error_type": type(exc).__name__, "message": str(exc)},
                    )
                )
                repository.save_event(error)
                repository.save_run(
                    run_id,
                    machine.state.value,
                    question,
                    {"state": machine.state.value, "error": str(exc)},
                )
                raise

    def _finalize(
        self,
        *,
        repository: ScholarRepository,
        event_sink: EventSink,
        machine: ResearchStateMachine,
        run_id: str,
        question: str,
        run_path: Path,
        papers: list[Paper],
        evidence_units: list[EvidenceUnit],
        claims: list[Claim],
        hypotheses: list[Hypothesis],
        retrieval_mode: str,
        warnings: list[str],
        query_plan: object | None = None,
        budget_manager: BudgetManager | None = None,
        source_stats: dict[str, dict[str, object]] | None = None,
    ) -> ResearchRunResult:
        completed = event_sink.emit(
            RunEvent.new(
                RunEventType.RUN_COMPLETED,
                run_id=run_id,
                payload={
                    "status": machine.state.value,
                    "papers": len(papers),
                    "evidence": len(evidence_units),
                    "claims": len(claims),
                    "hypotheses": len(hypotheses),
                },
            )
        )
        repository.save_event(completed)

        write_json(
            run_path / "research-brief.json", {"question": question, "status": machine.state.value}
        )
        if query_plan is not None and not (run_path / "search-plan.json").exists():
            write_json(run_path / "search-plan.json", query_plan.model_dump(mode="json"))
        write_json(run_path / "papers.json", [paper.model_dump(mode="json") for paper in papers])
        write_json(
            run_path / "evidence.json",
            [evidence.model_dump(mode="json") for evidence in evidence_units],
        )
        write_json(run_path / "claims.json", [claim.model_dump(mode="json") for claim in claims])
        write_json(
            run_path / "hypotheses.json",
            [hypothesis.model_dump(mode="json") for hypothesis in hypotheses],
        )
        manifest = {
            "run_id": run_id,
            "project_path": str(self.project_path),
            "question": question,
            "status": machine.state.value,
            "retrieval_mode": retrieval_mode,
            "enabled_sources": sorted((source_stats or {}).keys()),
            "source_stats": source_stats or {},
            "budget_usage": budget_manager.snapshot() if budget_manager else {},
            "warnings": warnings,
            "artifacts": {
                "events": str(run_path / "events.jsonl"),
                "report": str(run_path / "report.md"),
                "papers": str(run_path / "papers.json"),
                "evidence": str(run_path / "evidence.json"),
                "claims": str(run_path / "claims.json"),
                "hypotheses": str(run_path / "hypotheses.json"),
            },
        }
        write_json(run_path / "run-manifest.json", manifest)
        repository.save_run_manifest(run_id, manifest)
        if budget_manager:
            repository.save_budget_usage(run_id, budget_manager.snapshot())
        if query_plan is not None:
            report = render_report(
                question=question,
                query_plan=query_plan,
                papers=papers,
                evidence_units=evidence_units,
                claims=claims,
                hypotheses=hypotheses,
                retrieval_mode=retrieval_mode,
                warnings=warnings,
            )
        else:
            report = f"# Research Brief: {question}\n\n[证据不足] No search plan was generated.\n"
        (run_path / "report.md").write_text(report, encoding="utf-8")
        repository.save_run(
            run_id,
            machine.state.value,
            question,
            {
                "state": machine.state.value,
                "run_path": str(run_path),
                "papers": [paper.work_id for paper in papers],
                "retrieval_mode": retrieval_mode,
                "warnings": warnings,
            },
        )
        return ResearchRunResult(
            run_id=run_id,
            status=machine.state,
            question=question,
            run_path=run_path,
            papers=papers,
            evidence_units=evidence_units,
            claims=claims,
            hypotheses=hypotheses,
            warnings=warnings,
            retrieval_mode=retrieval_mode,
        )

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with self.database as connection:
            return ScholarRepository(connection).get_run(run_id)

    def latest_run(self) -> dict[str, object] | None:
        with self.database as connection:
            return ScholarRepository(connection).latest_run()

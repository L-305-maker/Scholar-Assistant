from __future__ import annotations

from scholar_assistant.core.events import EventSink
from scholar_assistant.schemas.events import RunEvent, RunEventType
from scholar_assistant.schemas.evidence import (
    Claim,
    ClaimType,
    EvidenceUnit,
    Hypothesis,
    SupportStatus,
)
from scholar_assistant.storage.repositories import ScholarRepository


class Analyst:
    def __init__(
        self, repository: ScholarRepository, event_sink: EventSink, *, run_id: str
    ) -> None:
        self.repository = repository
        self.event_sink = event_sink
        self.run_id = run_id

    def analyze(
        self,
        evidence_units: list[EvidenceUnit],
        *,
        question: str,
        max_claims: int = 30,
        max_hypotheses: int = 5,
    ) -> tuple[list[Claim], list[Hypothesis]]:
        claims: list[Claim] = []
        corpus_signals = extract_corpus_signals(evidence_units)
        for evidence in evidence_units[:max_claims]:
            content = _claim_from_evidence(evidence)
            claim_type = (
                ClaimType.PAPER_FACT
                if evidence.source_type.value == "pdf_fulltext"
                else ClaimType.AUTHOR_CLAIM
            )
            claim = Claim(
                content=content,
                type=claim_type,
                scope=evidence.work_id,
                supporting_evidence_ids=[evidence.evidence_id],
                support_status=SupportStatus.VERIFIED
                if claim_type == ClaimType.PAPER_FACT
                else SupportStatus.PARTIALLY_SUPPORTED,
                confidence=0.75 if claim_type == ClaimType.PAPER_FACT else 0.55,
                metadata={
                    "structured_signals": extract_corpus_signals([evidence]),
                },
            )
            claims.append(claim)

        if len(evidence_units) >= 2 and len(claims) < max_claims:
            ev_ids = [
                evidence.evidence_id for evidence in evidence_units[: min(4, len(evidence_units))]
            ]
            claims.append(
                Claim(
                    content=(
                        "The selected papers discuss memory or retrieval as a system component, "
                        "but deeper full-text evidence is required before method ranking."
                    ),
                    type=ClaimType.CROSS_PAPER_SYNTHESIS,
                    scope="selected_corpus",
                    supporting_evidence_ids=ev_ids,
                    support_status=SupportStatus.PARTIALLY_SUPPORTED,
                    confidence=0.5,
                    metadata={
                        "experiment_conditions": "not_established",
                        "comparability": "needs_manual_check",
                        "corpus_signals": corpus_signals,
                    },
                )
            )

        hypotheses: list[Hypothesis] = []
        if evidence_units and max_hypotheses > 0:
            supporting_claims = [claim.claim_id for claim in claims[: min(3, len(claims))]]
            hypotheses.append(
                Hypothesis(
                    content=(
                        "A memory retrieval filter that combines recency, semantic relevance, and "
                        "contradiction checks may reduce noisy memory injections in LLM agents."
                    ),
                    motivation=f"Generated from the research question: {question}",
                    supporting_claims=supporting_claims,
                    generated_queries=[
                        "LLM agent memory retrieval noise filtering evaluation",
                        "long-term memory agent retrieval benchmark irrelevant memories",
                    ],
                    falsification_condition=(
                        "On a fixed agent-memory benchmark, the filter does not reduce irrelevant "
                        "retrieved memories or harms task success beyond a predefined margin."
                    ),
                    minimum_experiment=(
                        "Build a corpus with relevant, stale, and conflicting memories; "
                        "compare baseline retrieval with the proposed filter on answer accuracy "
                        "and irrelevant-memory rate."
                    ),
                    risks=[
                        "Current MVP may rely on abstracts if PDFs cannot be downloaded.",
                        "Cross-paper comparability needs manual review of datasets and metrics.",
                    ],
                )
            )

        for claim in claims:
            self.event_sink.emit(
                RunEvent.new(
                    RunEventType.CLAIM_CREATED,
                    run_id=self.run_id,
                    payload={"claim_id": claim.claim_id, "type": claim.type.value},
                )
            )
        for hypothesis in hypotheses:
            self.event_sink.emit(
                RunEvent.new(
                    RunEventType.HYPOTHESIS_CREATED,
                    run_id=self.run_id,
                    payload={"hypothesis_id": hypothesis.hypothesis_id},
                )
            )
        return claims, hypotheses[:max_hypotheses]


def _claim_from_evidence(evidence: EvidenceUnit) -> str:
    content = evidence.content.strip()
    if len(content) > 260:
        content = content[:257].rstrip() + "..."
    location = f"page {evidence.page}" if evidence.page else evidence.section or "abstract"
    return f"Evidence from {evidence.work_id} ({location}) states: {content}"


def extract_corpus_signals(evidence_units: list[EvidenceUnit]) -> dict[str, list[str]]:
    patterns = {
        "datasets": ["dataset", "benchmark", "corpus", "data set"],
        "metrics": ["metric", "accuracy", "f1", "precision", "recall", "score"],
        "baselines": ["baseline", "compare", "comparison", "versus"],
        "results": ["result", "outperform", "improve", "performance", "achieve"],
        "limitations": ["limitation", "future work", "fail", "noise", "error"],
    }
    signals = {field: [] for field in patterns}
    for evidence in evidence_units:
        lower = evidence.content.lower()
        for field, keywords in patterns.items():
            if any(keyword in lower for keyword in keywords):
                signals[field].append(evidence.evidence_id)
    return {field: ids[:10] for field, ids in signals.items()}

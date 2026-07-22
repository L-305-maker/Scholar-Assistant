from __future__ import annotations

import os

import pytest

from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.core.quality_gate import QualityGate, QualityGateError
from scholar_assistant.retrieval.model_smoke import run_retrieval_smoke
from scholar_assistant.schemas.evidence import (
    Claim,
    ClaimType,
    EvidenceType,
    EvidenceUnit,
    SourceType,
    SupportStatus,
)
from scholar_assistant.storage.files import sha256_text


def _evidence(work_id: str, version_id: str = "ver") -> EvidenceUnit:
    return EvidenceUnit(
        work_id=work_id,
        version_id=f"{version_id}_{work_id}",
        content=f"Evidence for {work_id}",
        evidence_type=EvidenceType.SECTION_TEXT,
        source_type=SourceType.PDF_FULLTEXT,
        content_hash=sha256_text(work_id),
    )


def test_cross_paper_synthesis_requires_two_independent_works() -> None:
    evidence = [_evidence("wk_1", "a"), _evidence("wk_1", "b")]
    evidence_map = {item.evidence_id: item for item in evidence}
    claim = Claim(
        content="The corpus shares a memory retrieval concern.",
        type=ClaimType.CROSS_PAPER_SYNTHESIS,
        scope="selected_corpus",
        supporting_evidence_ids=[item.evidence_id for item in evidence],
    )
    with pytest.raises(QualityGateError):
        QualityGate(evidence_map).validate_claim(claim)

    second = _evidence("wk_2")
    evidence_map[second.evidence_id] = second
    claim.supporting_evidence_ids.append(second.evidence_id)
    assert QualityGate(evidence_map).validate_claim(claim)


def test_hypothesis_claim_and_incomparable_experiment_rules() -> None:
    evidence = _evidence("wk_1")
    evidence_map = {evidence.evidence_id: evidence}
    hypothesis = Claim(
        content="A retrieval filter may reduce memory noise.",
        type=ClaimType.RESEARCH_HYPOTHESIS,
        supporting_evidence_ids=[evidence.evidence_id],
    )
    with pytest.raises(QualityGateError):
        QualityGate(evidence_map).validate_claim(hypothesis)

    ranking = Claim(
        content="Method A is better than Method B.",
        type=ClaimType.AGENT_INFERENCE,
        supporting_evidence_ids=[evidence.evidence_id],
        support_status=SupportStatus.PARTIALLY_SUPPORTED,
        metadata={
            "comparability": {
                "base_model": "different",
                "dataset": "same",
                "metric": "same",
                "inference_budget": "same",
                "training_data": "same",
            }
        },
    )
    with pytest.raises(QualityGateError):
        QualityGate(evidence_map).validate_claim(ranking)


@pytest.mark.model
def test_model_smoke_is_marker_gated_without_env() -> None:
    if os.environ.get("SCHOLAR_RUN_MODEL_TESTS") != "1":
        pytest.skip("set SCHOLAR_RUN_MODEL_TESTS=1 to run real retrieval model smoke")
    result = run_retrieval_smoke(ScholarSettings.defaults(), allow_model_download=False)
    assert result["status"] in {"ok", "degraded", "failed"}

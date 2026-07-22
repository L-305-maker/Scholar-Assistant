from __future__ import annotations

from dataclasses import dataclass

from scholar_assistant.schemas.evidence import Claim, ClaimType, EvidenceUnit, SupportStatus
from scholar_assistant.schemas.paper import PaperVersion


class QualityGateError(ValueError):
    pass


@dataclass(slots=True)
class QualityGate:
    evidence: dict[str, EvidenceUnit]
    versions: dict[str, PaperVersion] | None = None

    def validate_claim(self, claim: Claim) -> Claim:
        missing = [
            evidence_id
            for evidence_id in claim.supporting_evidence_ids + claim.counter_evidence_ids
            if evidence_id not in self.evidence
        ]
        if missing:
            msg = f"Claim references missing evidence IDs: {', '.join(missing)}"
            raise QualityGateError(msg)

        if claim.type == ClaimType.PAPER_FACT and not claim.supporting_evidence_ids:
            msg = "paper_fact claims require at least one supporting evidence ID"
            raise QualityGateError(msg)

        for evidence_id in claim.supporting_evidence_ids:
            self._validate_evidence_location(self.evidence[evidence_id])

        if claim.type == ClaimType.CROSS_PAPER_SYNTHESIS:
            work_ids = {
                self.evidence[evidence_id].work_id for evidence_id in claim.supporting_evidence_ids
            }
            if len(work_ids) < 2:
                msg = "cross_paper_synthesis requires evidence from at least two independent works"
                raise QualityGateError(msg)
            if not claim.scope:
                msg = "cross_paper_synthesis requires an explicit scope"
                raise QualityGateError(msg)

        if claim.type == ClaimType.AGENT_INFERENCE:
            claim.metadata["explicit_inference"] = True
            if not claim.supporting_evidence_ids and not claim.counter_evidence_ids:
                claim.support_status = SupportStatus.INSUFFICIENT_EVIDENCE

        if claim.type == ClaimType.RESEARCH_HYPOTHESIS:
            if not claim.metadata.get("falsification_condition"):
                msg = "research_hypothesis claims require a falsification condition"
                raise QualityGateError(msg)
            if not claim.metadata.get("generated_queries"):
                msg = "research_hypothesis claims require generated verification queries"
                raise QualityGateError(msg)

        if (
            _is_direct_ranking_claim(claim)
            and not _has_comparable_experiment_metadata(claim)
        ):
            msg = "Direct superiority ranking requires comparable experiment conditions"
            raise QualityGateError(msg)

        if claim.supporting_evidence_ids and all(
            self.evidence[evidence_id].source_type.value == "abstract_only"
            for evidence_id in claim.supporting_evidence_ids
        ):
            claim.confidence = min(claim.confidence, 0.55)
            if claim.support_status == SupportStatus.VERIFIED:
                claim.support_status = SupportStatus.PARTIALLY_SUPPORTED

        if (
            claim.type == ClaimType.RESEARCH_HYPOTHESIS
            and claim.support_status == SupportStatus.VERIFIED
        ):
            claim.support_status = SupportStatus.PARTIALLY_SUPPORTED

        return claim

    def validate_claims(self, claims: list[Claim]) -> list[Claim]:
        return [self.validate_claim(claim) for claim in claims]

    def _validate_evidence_location(self, evidence: EvidenceUnit) -> None:
        if not self.versions:
            return
        version = self.versions.get(evidence.version_id)
        if version is None:
            msg = f"Evidence references missing version ID: {evidence.version_id}"
            raise QualityGateError(msg)
        if evidence.work_id != version.work_id:
            msg = f"Evidence work/version mismatch: {evidence.evidence_id}"
            raise QualityGateError(msg)
        if (
            evidence.page is not None
            and version.page_count is not None
            and evidence.page > version.page_count
        ):
            msg = (
                f"Evidence page {evidence.page} exceeds version "
                f"{version.version_id} page count {version.page_count}"
            )
            raise QualityGateError(msg)


def _is_direct_ranking_claim(claim: Claim) -> bool:
    if claim.type not in {ClaimType.CROSS_PAPER_SYNTHESIS, ClaimType.AGENT_INFERENCE}:
        return False
    content = claim.content.lower()
    ranking_terms = [
        "better than",
        "superior",
        "outperform",
        "outperforms",
        "best",
        "优于",
        "超过",
        "最佳",
        "最优",
    ]
    return any(term in content for term in ranking_terms)


def _has_comparable_experiment_metadata(claim: Claim) -> bool:
    if claim.metadata.get("experiment_conditions") == "comparable":
        return True
    comparability = claim.metadata.get("comparability") or {}
    if not isinstance(comparability, dict):
        return False
    blocking_keys = [
        "base_model",
        "dataset",
        "metric",
        "inference_budget",
        "training_data",
    ]
    return bool(comparability) and all(comparability.get(key) == "same" for key in blocking_keys)

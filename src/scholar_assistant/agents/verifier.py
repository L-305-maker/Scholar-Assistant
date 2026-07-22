from __future__ import annotations

from scholar_assistant.core.quality_gate import QualityGate
from scholar_assistant.schemas.evidence import Claim, EvidenceUnit
from scholar_assistant.schemas.paper import PaperVersion


class Verifier:
    def verify_claims(
        self,
        claims: list[Claim],
        evidence_units: list[EvidenceUnit],
        versions: list[PaperVersion] | None = None,
    ) -> list[Claim]:
        evidence = {unit.evidence_id: unit for unit in evidence_units}
        version_map = {version.version_id: version for version in versions or []}
        gate = QualityGate(evidence=evidence, versions=version_map)
        return gate.validate_claims(claims)

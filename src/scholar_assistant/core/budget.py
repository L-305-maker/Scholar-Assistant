from __future__ import annotations

from dataclasses import dataclass

from scholar_assistant.core.config import BudgetConfig


@dataclass(slots=True)
class BudgetManager:
    config: BudgetConfig
    search_loops_used: int = 0
    verification_loops_used: int = 0
    raw_candidates: int = 0
    rerank_candidates: int = 0
    core_papers: int = 0
    deep_reads: int = 0
    core_claims: int = 0
    hypotheses: int = 0

    def can_search(self) -> bool:
        return self.search_loops_used < self.config.main_search_loops

    def can_verify_search(self) -> bool:
        return self.verification_loops_used < self.config.verification_search_loops

    def mark_search_loop(self) -> None:
        self.search_loops_used += 1

    def within_candidate_budget(self, count: int) -> int:
        remaining = max(self.config.max_raw_candidates - self.raw_candidates, 0)
        accepted = min(count, remaining)
        self.raw_candidates += accepted
        return accepted

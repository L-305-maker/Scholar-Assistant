from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scholar_assistant.core.config import BudgetConfig

if TYPE_CHECKING:
    from scholar_assistant.tools.registry import ToolSpec


class BudgetLimitExceeded(RuntimeError):
    def __init__(self, message: str, *, counter: str | None = None) -> None:
        super().__init__(message)
        self.counter = counter


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
    counters: Counter[str] = field(default_factory=Counter)
    per_source_calls: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: Counter[str] = field(default_factory=Counter)
    warnings: list[str] = field(default_factory=list)

    def can_search(self) -> bool:
        return self.search_loops_used < self.config.main_search_loops

    def can_verify_search(self) -> bool:
        return self.verification_loops_used < self.config.verification_search_loops

    def mark_search_loop(self) -> None:
        if not self.can_search():
            raise BudgetLimitExceeded("main search loop budget exhausted", counter="search_loops")
        self.search_loops_used += 1

    def within_candidate_budget(self, count: int) -> int:
        remaining = max(self.config.max_raw_candidates - self.raw_candidates, 0)
        if count > 0 and remaining <= 0:
            raise BudgetLimitExceeded("raw candidate budget exhausted", counter="raw_candidates")
        accepted = min(count, remaining)
        self.raw_candidates += accepted
        return accepted

    def check_tool(self, spec: ToolSpec, inputs: dict[str, Any] | None = None) -> None:
        inputs = inputs or {}
        if _is_cache_hit(inputs):
            return
        if spec.category == "source":
            self._check_less_than("search_calls", self.config.max_search_calls)
            self._check_less_than(
                spec.name,
                self.config.max_source_calls_per_source,
                per_source=True,
            )
        elif spec.name == "pdf.download":
            self._check_less_than("pdf_downloads", self.config.max_pdf_downloads)
        elif spec.name == "pdf.parse":
            self._check_less_than("pdf_parses", self.config.max_pdf_parses)
        elif spec.name == "retrieval.dense":
            self._check_less_than("dense_retrievals", self.config.max_dense_retrievals)
        elif spec.name == "retrieval.reranker":
            documents = int(inputs.get("document_count", 1))
            if self.counters["reranker_documents"] + documents > self.config.max_reranker_documents:
                raise BudgetLimitExceeded(
                    "budget exhausted for reranker_documents",
                    counter="reranker_documents",
                )
        elif spec.category == "mcp":
            self._check_less_than("mcp_tool_calls", self.config.max_mcp_tool_calls)
        elif spec.category == "model":
            self._check_less_than("model_requests", self.config.max_model_requests)

    def record_tool(self, spec: ToolSpec, inputs: dict[str, Any], result: Any) -> None:
        if _is_cache_hit(inputs):
            self.record_cache_hit(spec.name)
            return
        if spec.category == "source":
            self.counters["search_calls"] += 1
            self.per_source_calls[spec.name] += 1
        elif spec.name == "pdf.download":
            self.counters["pdf_downloads"] += 1
        elif spec.name == "pdf.parse":
            self.counters["pdf_parses"] += 1
        elif spec.name == "retrieval.dense":
            self.counters["dense_retrievals"] += 1
        elif spec.name == "retrieval.reranker":
            documents = int(inputs.get("document_count", 1))
            self.counters["reranker_documents"] += documents
        elif spec.category == "mcp":
            self.counters["mcp_tool_calls"] += 1
        elif spec.category == "model":
            self.counters["model_requests"] += 1
            usage = getattr(result, "usage", None)
            if usage is not None:
                self.counters["input_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                self.counters["output_tokens"] += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )

    def record_retry(self, tool_name: str) -> None:
        if self.counters["retries"] >= self.config.max_retries:
            raise BudgetLimitExceeded("retry budget exhausted", counter="retries")
        self.counters["retries"] += 1
        self.counters[f"retries:{tool_name}"] += 1

    def record_cache_hit(self, tool_name: str) -> None:
        self.cache_hits[tool_name] += 1

    def _check_less_than(self, counter: str, limit: int, *, per_source: bool = False) -> None:
        current = self.per_source_calls[counter] if per_source else self.counters[counter]
        if current >= limit:
            raise BudgetLimitExceeded(f"budget exhausted for {counter}", counter=counter)

    def snapshot(self) -> dict[str, Any]:
        return {
            "search_loops_used": self.search_loops_used,
            "verification_loops_used": self.verification_loops_used,
            "raw_candidates": self.raw_candidates,
            "rerank_candidates": self.rerank_candidates,
            "core_papers": self.core_papers,
            "deep_reads": self.deep_reads,
            "core_claims": self.core_claims,
            "hypotheses": self.hypotheses,
            "counters": dict(self.counters),
            "per_source_calls": dict(self.per_source_calls),
            "cache_hits": dict(self.cache_hits),
            "warnings": list(self.warnings),
        }


def _is_cache_hit(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get("cache_hit"))

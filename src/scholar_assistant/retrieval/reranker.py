from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scholar_assistant.schemas.paper import Paper


@dataclass(frozen=True)
class RerankResult:
    work_id: str
    score: float
    mode: str


class BGEReranker:
    def __init__(
        self, model_name: str = "BAAI/bge-reranker-v2-m3", *, use_fp16: bool = False
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model: Any | None = None

    @property
    def available(self) -> bool:
        try:
            import FlagEmbedding  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self) -> Any:
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                msg = "FlagEmbedding is not installed; reranker is disabled"
                raise RuntimeError(msg) from exc
            self._model = FlagReranker(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def rerank(self, query: str, papers: list[Paper]) -> list[RerankResult]:
        if not papers:
            return []
        model = self._load()
        pairs = [[query, f"{paper.title}\n{paper.abstract or ''}"] for paper in papers]
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        return [
            RerankResult(work_id=paper.work_id, score=float(score), mode="bge-reranker-v2-m3")
            for paper, score in sorted(
                zip(papers, scores, strict=False), key=lambda pair: pair[1], reverse=True
            )
        ]


def fallback_rerank(papers: list[Paper]) -> list[RerankResult]:
    return [
        RerankResult(work_id=paper.work_id, score=paper.relevance_score, mode="fallback")
        for paper in sorted(papers, key=lambda item: item.relevance_score, reverse=True)
    ]

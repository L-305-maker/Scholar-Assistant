from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scholar_assistant.retrieval.device import select_flagembedding_device
from scholar_assistant.schemas.paper import Paper


@dataclass(frozen=True)
class RerankResult:
    work_id: str
    score: float
    mode: str


class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        use_fp16: bool = False,
        device: str = "auto",
        cache_dir: Path | None = None,
        batch_size: int = 8,
        max_length: int = 8192,
        revision: str | None = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.revision = revision
        self.allow_cpu_fallback = allow_cpu_fallback
        self._model: Any | None = None
        self.metadata: dict[str, Any] = {
            "model_name": model_name,
            "revision": revision,
            "device": device,
            "batch_size": batch_size,
            "max_length": max_length,
            "loaded": False,
        }

    @property
    def available(self) -> bool:
        try:
            import FlagEmbedding  # noqa: F401
        except Exception as exc:
            self.metadata["availability_error"] = f"{type(exc).__name__}: {exc}"
            return False
        return True

    def _load(self) -> Any:
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                msg = "FlagEmbedding is not installed; reranker is disabled"
                raise RuntimeError(msg) from exc
            if self.cache_dir:
                os.environ.setdefault("HF_HOME", str(self.cache_dir))
            start = time.perf_counter()
            kwargs: dict[str, Any] = {"use_fp16": self.use_fp16}
            selected_device = select_flagembedding_device(
                self.device,
                allow_cpu_fallback=self.allow_cpu_fallback,
                metadata=self.metadata,
            )
            if selected_device:
                kwargs["devices"] = selected_device
                self.device = selected_device
            if self.cache_dir:
                kwargs["cache_dir"] = str(self.cache_dir)
            if self.revision:
                kwargs["revision"] = self.revision
            try:
                self._model = FlagReranker(self.model_name, **kwargs)
            except RuntimeError as exc:
                if (
                    self.allow_cpu_fallback
                    and self.device not in {"cpu", "auto"}
                    and "out of memory" in str(exc).lower()
                ):
                    self.device = "cpu"
                    kwargs["devices"] = "cpu"
                    self._model = FlagReranker(self.model_name, **kwargs)
                    self.metadata["warning"] = "CUDA OOM; retried on CPU"
                else:
                    raise RuntimeError(f"BGE reranker load failed: {type(exc).__name__}") from exc
            self.metadata.update(
                {
                    "loaded": True,
                    "device": self.device,
                    "load_time_ms": (time.perf_counter() - start) * 1000,
                }
            )
        return self._model

    def rerank(self, query: str, papers: list[Paper]) -> list[RerankResult]:
        if not papers:
            return []
        model = self._load()
        pairs = [[query, f"{paper.title}\n{paper.abstract or ''}"] for paper in papers]
        start = time.perf_counter()
        scores = model.compute_score(
            pairs,
            normalize=True,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        self.metadata["last_inference_time_ms"] = (time.perf_counter() - start) * 1000
        self.metadata["last_pair_count"] = len(pairs)
        if isinstance(scores, float):
            scores = [scores]
        return [
            RerankResult(work_id=paper.work_id, score=float(score), mode="bge-reranker-v2-m3")
            for paper, score in sorted(
                zip(papers, scores, strict=False), key=lambda pair: pair[1], reverse=True
            )
        ]

    async def rerank_async(self, query: str, papers: list[Paper]) -> list[RerankResult]:
        return await asyncio.to_thread(self.rerank, query, papers)


def fallback_rerank(papers: list[Paper]) -> list[RerankResult]:
    return [
        RerankResult(work_id=paper.work_id, score=paper.relevance_score, mode="fallback")
        for paper in sorted(papers, key=lambda item: item.relevance_score, reverse=True)
    ]

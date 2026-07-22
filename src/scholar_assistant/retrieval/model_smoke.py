from __future__ import annotations

import math
import os
from typing import Any

from scholar_assistant.core.config import ScholarSettings
from scholar_assistant.retrieval.embeddings import BGEM3Embedder, cosine_search
from scholar_assistant.retrieval.reranker import BGEReranker
from scholar_assistant.schemas.paper import Paper


def run_retrieval_smoke(
    settings: ScholarSettings,
    *,
    allow_model_download: bool = True,
) -> dict[str, Any]:
    if not allow_model_download and not os.environ.get("SCHOLAR_RUN_MODEL_TESTS"):
        return {
            "status": "skipped",
            "reason": "model smoke requires SCHOLAR_RUN_MODEL_TESTS=1",
            "retrieval_mode": "not-run",
        }
    query = "long-term memory retrieval for LLM agents"
    documents = [
        Paper(
            title="LLM Agent Long-Term Memory Retrieval",
            abstract="Long-term memory retrieval for LLM agents can suffer from noisy memories.",
        ),
        Paper(
            title="Image Classification With Vision Transformers",
            abstract="This document discusses image classification datasets and visual features.",
        ),
        Paper(
            title="Serializable Database Transactions",
            abstract="This document discusses database isolation levels and transactions.",
        ),
    ]
    config = settings.retrieval
    embedder = BGEM3Embedder(
        config.bge_m3_model,
        device=config.device,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        max_length=config.max_length,
        revision=config.model_revision,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    if not embedder.available:
        return {
            "status": "degraded",
            "reason": embedder.metadata.get(
                "availability_error",
                "FlagEmbedding is not installed",
            ),
            "retrieval_mode": "bm25-only",
            "model": embedder.metadata,
        }
    texts = [query, *[f"{paper.title}\n{paper.abstract}" for paper in documents]]
    try:
        vectors = embedder.encode(texts)
        hits = cosine_search(vectors[0], [paper.work_id for paper in documents], vectors[1:])
        first_work_id = hits[0].item_id
        finite_scores = all(math.isfinite(hit.score) for hit in hits)
    except RuntimeError as exc:
        return {
            "status": "degraded",
            "reason": f"BGE-M3 failed: {type(exc).__name__}",
            "retrieval_mode": "bm25-only",
            "model": embedder.metadata,
        }
    reranker = BGEReranker(
        config.bge_reranker_model,
        device=config.device,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
        max_length=config.max_length,
        revision=config.model_revision,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    rerank_status = "skipped"
    reranked_first = None
    if reranker.available:
        try:
            reranked = reranker.rerank(query, documents)
            rerank_status = "ok"
            reranked_first = reranked[0].work_id if reranked else None
        except RuntimeError as exc:
            rerank_status = f"degraded:{type(exc).__name__}"
    return {
        "status": "ok" if first_work_id == documents[0].work_id and finite_scores else "failed",
        "dense_first_relevant": first_work_id == documents[0].work_id,
        "reranker_status": rerank_status,
        "reranked_first_relevant": reranked_first in {None, documents[0].work_id},
        "vector_dim": int(vectors.shape[1]) if vectors.ndim == 2 else None,
        "finite_scores": finite_scores,
        "embedding_model": embedder.metadata,
        "reranker_model": reranker.metadata,
        "retrieval_mode": "bge-m3+dense",
    }

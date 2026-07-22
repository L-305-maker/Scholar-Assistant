from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scholar_assistant.retrieval.device import select_flagembedding_device


@dataclass(frozen=True)
class DenseSearchResult:
    item_id: str
    score: float


class EmbeddingUnavailable(RuntimeError):
    pass


class BGEM3Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
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
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise EmbeddingUnavailable(
                    "FlagEmbedding is not installed; dense retrieval is disabled"
                ) from exc
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
                self._model = BGEM3FlagModel(self.model_name, **kwargs)
            except RuntimeError as exc:
                if (
                    self.allow_cpu_fallback
                    and self.device not in {"cpu", "auto"}
                    and "out of memory" in str(exc).lower()
                ):
                    self.device = "cpu"
                    kwargs["devices"] = "cpu"
                    self._model = BGEM3FlagModel(self.model_name, **kwargs)
                    self.metadata["warning"] = "CUDA OOM; retried on CPU"
                else:
                    raise EmbeddingUnavailable(f"BGE-M3 load failed: {type(exc).__name__}") from exc
            self.metadata.update(
                {
                    "loaded": True,
                    "device": self.device,
                    "load_time_ms": (time.perf_counter() - start) * 1000,
                }
            )
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        start = time.perf_counter()
        output = model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        dense = output["dense_vecs"] if isinstance(output, dict) else output
        vectors = np.asarray(dense, dtype=np.float32)
        self.metadata["last_inference_time_ms"] = (time.perf_counter() - start) * 1000
        self.metadata["last_vector_dim"] = int(vectors.shape[1]) if vectors.ndim == 2 else None
        return normalize(vectors)

    async def encode_async(self, texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(self.encode, texts)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def save_vectors(path: Path, item_ids: list[str], vectors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, item_ids=np.asarray(item_ids), vectors=vectors)


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return [str(item_id) for item_id in data["item_ids"].tolist()], np.asarray(data["vectors"])


def cosine_search(
    query_vector: np.ndarray,
    item_ids: list[str],
    vectors: np.ndarray,
    *,
    limit: int = 50,
) -> list[DenseSearchResult]:
    query = normalize(query_vector.reshape(1, -1))[0]
    scores = vectors @ query
    order = np.argsort(scores)[::-1][:limit]
    return [
        DenseSearchResult(item_id=item_ids[index], score=float(scores[index])) for index in order
    ]

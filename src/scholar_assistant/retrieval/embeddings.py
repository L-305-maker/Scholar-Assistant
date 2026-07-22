from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DenseSearchResult:
    item_id: str
    score: float


class EmbeddingUnavailable(RuntimeError):
    pass


class BGEM3Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", *, use_fp16: bool = False) -> None:
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
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise EmbeddingUnavailable(
                    "FlagEmbedding is not installed; dense retrieval is disabled"
                ) from exc
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        output = model.encode(
            texts, return_dense=True, return_sparse=False, return_colbert_vecs=False
        )
        dense = output["dense_vecs"] if isinstance(output, dict) else output
        vectors = np.asarray(dense, dtype=np.float32)
        return normalize(vectors)


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

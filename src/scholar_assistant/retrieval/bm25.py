from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from scholar_assistant.retrieval.fusion import RankedItem


@dataclass(frozen=True)
class RetrievalHit:
    work_id: str
    score: float
    mode: str


class BM25Retriever:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def search(self, query: str, *, limit: int = 50) -> list[RetrievalHit]:
        fts_query = build_fts_query(query)
        if not fts_query:
            return []
        rows = self.connection.execute(
            """
            SELECT work_id, bm25(papers_fts) AS bm25_score
            FROM papers_fts
            WHERE papers_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        hits = [
            RetrievalHit(
                work_id=row["work_id"],
                score=1.0 / (1.0 + abs(float(row["bm25_score"]))),
                mode="bm25",
            )
            for row in rows
        ]
        return hits

    def ranked_items(self, query: str, *, limit: int = 50) -> list[RankedItem]:
        return [
            RankedItem(item_id=hit.work_id, score=hit.score, source=hit.mode)
            for hit in self.search(query, limit=limit)
        ]


def build_fts_query(query: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9_]+", query.lower()) if len(token) > 1]
    return " OR ".join(tokens[:32])

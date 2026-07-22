from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedItem:
    item_id: str
    score: float
    source: str


def reciprocal_rank_fusion(
    rankings: list[list[RankedItem]],
    *,
    k: int = 60,
) -> list[RankedItem]:
    fused: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item.item_id] = fused.get(item.item_id, 0.0) + 1.0 / (k + rank)
            sources.setdefault(item.item_id, []).append(item.source)
    return [
        RankedItem(item_id=item_id, score=score, source="+".join(sorted(set(sources[item_id]))))
        for item_id, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
    ]

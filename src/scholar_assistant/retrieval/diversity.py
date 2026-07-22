from __future__ import annotations

from collections import defaultdict, deque

from scholar_assistant.schemas.paper import Paper, PaperRole

ROLE_ORDER = [
    PaperRole.FOUNDATION,
    PaperRole.MAINSTREAM,
    PaperRole.RECENT,
    PaperRole.BENCHMARK,
    PaperRole.CRITICAL,
    PaperRole.OTHER,
]


def assign_role(paper: Paper) -> PaperRole:
    text = f"{paper.title} {paper.abstract or ''}".lower()
    recent = bool(paper.year and paper.year >= 2023)
    if any(term in text for term in ["survey", "benchmark", "evaluation"]):
        return PaperRole.BENCHMARK if "benchmark" in text else PaperRole.MAINSTREAM
    if any(term in text for term in ["limitations", "noise", "negative", "failure"]):
        return PaperRole.CRITICAL
    if recent:
        return PaperRole.RECENT
    if paper.year and paper.year <= 2020:
        return PaperRole.FOUNDATION
    return paper.paper_role or PaperRole.OTHER


def diverse_select(papers: list[Paper], *, max_count: int) -> list[Paper]:
    if max_count <= 0:
        return []
    buckets: dict[PaperRole, deque[Paper]] = defaultdict(deque)
    for paper in sorted(papers, key=lambda item: item.relevance_score, reverse=True):
        role = assign_role(paper)
        paper.paper_role = role
        buckets[role].append(paper)

    selected: list[Paper] = []
    seen: set[str] = set()
    while len(selected) < max_count:
        progressed = False
        for role in ROLE_ORDER:
            bucket = buckets.get(role)
            while bucket:
                candidate = bucket.popleft()
                if candidate.work_id in seen:
                    continue
                selected.append(candidate)
                seen.add(candidate.work_id)
                progressed = True
                break
            if len(selected) >= max_count:
                break
        if not progressed:
            break
    return selected

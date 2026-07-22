from __future__ import annotations

import re

import httpx

from scholar_assistant.schemas.paper import Paper


class CrossrefClient:
    base_url = "https://api.crossref.org/works"

    async def search(self, query: str, *, max_results: int = 20) -> list[Paper]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.base_url,
                params={"query": query, "rows": max_results},
                headers={"User-Agent": "scholar-assistant/0.1 (mailto:example@example.com)"},
            )
            response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        papers: list[Paper] = []
        for item in items:
            title = (item.get("title") or ["Untitled"])[0]
            authors = [
                " ".join(filter(None, [author.get("given"), author.get("family")]))
                for author in item.get("author", [])
            ]
            year = _year_from_parts(item.get("published-print") or item.get("published-online"))
            papers.append(
                Paper(
                    title=title,
                    authors=[author for author in authors if author],
                    abstract=_strip_tags(item.get("abstract")),
                    year=year,
                    venue=(item.get("container-title") or [None])[0],
                    doi=item.get("DOI"),
                    source_ids={"crossref": item.get("DOI", "")},
                    source="crossref",
                )
            )
        return papers


def _year_from_parts(value: dict | None) -> int | None:
    try:
        return int(value["date-parts"][0][0]) if value else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _strip_tags(value: str | None) -> str | None:
    if not value:
        return None
    return re.sub(r"<[^>]+>", "", value).strip()

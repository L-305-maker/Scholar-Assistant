from __future__ import annotations

import httpx

from scholar_assistant.schemas.paper import Paper


class OpenAlexClient:
    base_url = "https://api.openalex.org/works"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(self, query: str, *, max_results: int = 20) -> list[Paper]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(
                self.base_url, params={"search": query, "per-page": max_results}
            )
            response.raise_for_status()
        data = response.json()
        papers: list[Paper] = []
        for item in data.get("results", []):
            authorships = item.get("authorships") or []
            papers.append(
                Paper(
                    title=item.get("title") or "Untitled",
                    authors=[
                        (authorship.get("author") or {}).get("display_name", "")
                        for authorship in authorships
                        if (authorship.get("author") or {}).get("display_name")
                    ],
                    abstract=_inverted_abstract(item.get("abstract_inverted_index")),
                    year=item.get("publication_year"),
                    venue=(item.get("primary_location") or {})
                    .get("source", {})
                    .get("display_name"),
                    doi=(item.get("doi") or "").removeprefix("https://doi.org/") or None,
                    source_ids={"openalex": item.get("id", "")},
                    source="openalex",
                )
            )
        return papers


def _inverted_abstract(index: dict[str, list[int]] | None) -> str | None:
    if not index:
        return None
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        words.extend((position, word) for position in positions)
    return " ".join(word for _, word in sorted(words))

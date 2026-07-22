from __future__ import annotations

import httpx

from scholar_assistant.schemas.paper import Paper


class SemanticScholarClient:
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, timeout_seconds: float = 30.0, api_key: str | None = None) -> None:
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    async def search(self, query: str, *, max_results: int = 20) -> list[Paper]:
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,authors,abstract,year,venue,externalIds,url,openAccessPdf",
        }
        headers = {"x-api-key": self.api_key} if self.api_key else None
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(self.base_url, params=params, headers=headers)
            response.raise_for_status()
        papers: list[Paper] = []
        for item in response.json().get("data", []):
            external_ids = item.get("externalIds") or {}
            open_pdf = item.get("openAccessPdf") or {}
            papers.append(
                Paper(
                    title=item.get("title") or "Untitled",
                    authors=[author.get("name", "") for author in item.get("authors", [])],
                    abstract=item.get("abstract"),
                    year=item.get("year"),
                    venue=item.get("venue"),
                    doi=external_ids.get("DOI"),
                    arxiv_id=external_ids.get("ArXiv"),
                    source_ids={"semantic_scholar": item.get("paperId", "")},
                    source="semantic_scholar",
                    pdf_url=open_pdf.get("url"),
                    metadata={"url": item.get("url")},
                )
            )
        return papers

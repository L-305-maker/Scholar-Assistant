from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import httpx

from scholar_assistant.schemas.paper import AccessType, Paper, PaperVersion, VersionType

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


@dataclass(slots=True)
class ArxivSearchResult:
    papers: list[Paper]
    versions: list[PaperVersion]
    raw_xml: str


class ArxivClient:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def search(
        self,
        query: str,
        *,
        max_results: int = 25,
        start: int = 0,
    ) -> ArxivSearchResult:
        params = {
            "search_query": _to_arxiv_query(query),
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
        return parse_arxiv_atom(response.text)


def _to_arxiv_query(query: str) -> str:
    if any(prefix in query for prefix in ["ti:", "abs:", "au:", "cat:", "all:"]):
        return query
    return f"all:{query}"


def parse_arxiv_atom(xml_text: str) -> ArxivSearchResult:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    versions: list[PaperVersion] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = _clean_text(_find_text(entry, "atom:title"))
        abstract = _clean_text(_find_text(entry, "atom:summary"))
        entry_id = _find_text(entry, "atom:id").strip()
        arxiv_id = _extract_arxiv_id(entry_id)
        published = _find_text(entry, "atom:published")
        year = _parse_year(published)
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        doi = _find_text(entry, "arxiv:doi") or None
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        pdf_url = _extract_pdf_url(entry)
        paper = Paper(
            title=title,
            authors=[author for author in authors if author],
            abstract=abstract,
            year=year,
            venue="arXiv",
            doi=doi,
            arxiv_id=arxiv_id,
            source_ids={"arxiv": arxiv_id} if arxiv_id else {},
            source="arxiv",
            pdf_url=pdf_url,
            categories=categories,
        )
        version = PaperVersion(
            work_id=paper.work_id,
            version_type=VersionType.ARXIV,
            source_url=entry_id,
            access_type=AccessType.FULLTEXT if pdf_url else AccessType.ABSTRACT_ONLY,
            version_label=_extract_version(arxiv_id),
            is_canonical=True,
            is_latest=True,
        )
        papers.append(paper)
        versions.append(version)
    return ArxivSearchResult(papers=papers, versions=versions, raw_xml=xml_text)


def merge_results(results: Iterable[ArxivSearchResult]) -> ArxivSearchResult:
    papers: list[Paper] = []
    versions: list[PaperVersion] = []
    raw = []
    for result in results:
        papers.extend(result.papers)
        versions.extend(result.versions)
        raw.append(result.raw_xml)
    return ArxivSearchResult(papers=papers, versions=versions, raw_xml="\n".join(raw))


def _find_text(entry: ET.Element, path: str) -> str:
    return entry.findtext(path, default="", namespaces=ATOM_NS)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_arxiv_id(entry_id: str) -> str | None:
    if not entry_id:
        return None
    return entry_id.rstrip("/").split("/")[-1]


def _extract_version(arxiv_id: str | None) -> str | None:
    if not arxiv_id:
        return None
    match = re.search(r"v\d+$", arxiv_id)
    return match.group(0) if match else None


def _parse_year(value: str) -> int | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).year
    except ValueError:
        return None


def _extract_pdf_url(entry: ET.Element) -> str | None:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            return link.attrib.get("href")
    arxiv_id = _extract_arxiv_id(_find_text(entry, "atom:id"))
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else None

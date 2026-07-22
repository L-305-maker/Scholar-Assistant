from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from scholar_assistant.schemas.paper import Paper

DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
ARXIV_RE = re.compile(
    r"^(?P<base>(?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?P<version>v\d+)?$",
    re.IGNORECASE,
)
LATEX_MARKUP_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?")
PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class ArxivIdentifier:
    raw: str
    base_id: str
    version: int | None = None


@dataclass(frozen=True)
class NormalizedAuthor:
    raw_name: str
    normalized_name: str
    surname: str
    initials: str
    orcid: str | None = None
    order: int = 0


@dataclass(frozen=True)
class DuplicateDecision:
    action: str
    reason: str
    confidence: float
    signals: dict[str, object] = field(default_factory=dict)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = DOI_PREFIX_RE.sub("", value.strip()).strip()
    normalized = normalized.rstrip(" .")
    return normalized.casefold() or None


def parse_arxiv_id(value: str | None) -> ArxivIdentifier | None:
    if not value:
        return None
    cleaned = value.strip().removeprefix("arXiv:").removeprefix("arxiv:")
    cleaned = cleaned.rstrip("/")
    match = ARXIV_RE.match(cleaned)
    if not match:
        return None
    version_text = match.group("version")
    version = int(version_text[1:]) if version_text else None
    return ArxivIdentifier(raw=value, base_id=match.group("base"), version=version)


def normalize_arxiv_base(value: str | None) -> str | None:
    parsed = parse_arxiv_id(value)
    return parsed.base_id if parsed else None


def normalize_source_id(source: str, source_id: str | None) -> str | None:
    if not source_id:
        return None
    source_name = source.replace("-", "_").casefold().strip()
    identifier = source_id.strip().rstrip("/")
    if not identifier:
        return None
    return f"{source_name}:{identifier}"


def normalize_title(title: str) -> str:
    text = html.unescape(title or "")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    text = _strip_latex(text)
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    text = text.casefold()
    text = re.sub(r"\s*:\s*", " ", text)
    text = PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_latex(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return match.group(1) or ""

    return LATEX_MARKUP_RE.sub(replace, text)


def title_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_title(left), normalize_title(right)).ratio()


def normalize_author(name: str, *, order: int = 0, orcid: str | None = None) -> NormalizedAuthor:
    normalized = unicodedata.normalize("NFKC", name).casefold()
    normalized = re.sub(r"[^\w\s-]", " ", normalized)
    parts = [part for part in re.sub(r"\s+", " ", normalized).strip().split(" ") if part]
    if not parts:
        return NormalizedAuthor(name, "", "", "", orcid, order)
    surname = parts[-1]
    initials = "".join(part[0] for part in parts[:-1] if part)
    return NormalizedAuthor(
        raw_name=name,
        normalized_name=" ".join(parts),
        surname=surname,
        initials=initials,
        orcid=orcid,
        order=order,
    )


def normalized_authors(authors: list[str]) -> list[NormalizedAuthor]:
    return [normalize_author(author, order=index) for index, author in enumerate(authors)]


def author_overlap(left: list[str], right: list[str]) -> float:
    left_authors = normalized_authors(left)
    right_authors = normalized_authors(right)
    if not left_authors or not right_authors:
        return 0.0
    left_keys = {(author.surname, author.initials[:1]) for author in left_authors}
    right_keys = {(author.surname, author.initials[:1]) for author in right_authors}
    return len(left_keys & right_keys) / max(len(left_keys | right_keys), 1)


def first_author_matches(left: list[str], right: list[str]) -> bool:
    if not left or not right:
        return False
    left_author = normalize_author(left[0])
    right_author = normalize_author(right[0])
    if not left_author.surname or not right_author.surname:
        return False
    return (
        left_author.surname == right_author.surname
        and left_author.initials[:1] == right_author.initials[:1]
    )


def year_distance(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return abs(left - right)


def decide_duplicate(left: Paper, right: Paper) -> DuplicateDecision:
    left_doi = normalize_doi(left.doi)
    right_doi = normalize_doi(right.doi)
    if left_doi and right_doi:
        if left_doi == right_doi:
            return DuplicateDecision("exact_merge", "normalized DOI match", 1.0)
        return DuplicateDecision("never_merge", "conflicting DOI", 1.0)

    left_arxiv = normalize_arxiv_base(left.arxiv_id)
    right_arxiv = normalize_arxiv_base(right.arxiv_id)
    if left_arxiv and right_arxiv:
        if left_arxiv == right_arxiv:
            return DuplicateDecision("exact_merge", "arXiv base ID match", 1.0)
        return DuplicateDecision("never_merge", "conflicting arXiv IDs", 0.95)

    for source, source_id in left.source_ids.items():
        left_source_id = normalize_source_id(source, source_id)
        right_source_id = normalize_source_id(source, right.source_ids.get(source))
        if left_source_id and right_source_id and left_source_id == right_source_id:
            return DuplicateDecision("exact_merge", "same-source internal ID match", 1.0)

    title_score = title_similarity(left.title, right.title)
    author_score = author_overlap(left.authors, right.authors)
    first_author = first_author_matches(left.authors, right.authors)
    distance = year_distance(left.year, right.year)
    signals = {
        "title_similarity": title_score,
        "author_overlap": author_score,
        "first_author_matches": first_author,
        "year_distance": distance,
    }
    no_identifier_conflict = not (
        (left_doi and right_doi and left_doi != right_doi)
        or (left_arxiv and right_arxiv and left_arxiv != right_arxiv)
    )
    year_compatible = distance is None or distance <= 1
    if title_score >= 0.965 and (first_author or author_score >= 0.34) and year_compatible:
        return DuplicateDecision(
            "strong_fuzzy_merge",
            "high title similarity with author/year compatibility",
            min(0.98, 0.70 + title_score * 0.2 + author_score * 0.1),
            signals,
        )
    if (
        no_identifier_conflict
        and title_score >= 0.90
        and (first_author or author_score >= 0.20)
        and (distance is None or distance <= 2)
    ):
        return DuplicateDecision(
            "possible_duplicate",
            "similar title and author signals",
            0.65,
            signals,
        )
    return DuplicateDecision("distinct", "insufficient duplicate evidence", 0.0, signals)

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import fitz

from scholar_assistant.storage.files import sha256_file


@dataclass(slots=True)
class ParsedParagraph:
    page: int
    paragraph_index: int
    text: str
    section: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    is_reference: bool = False
    caption_type: str | None = None


@dataclass(slots=True)
class ParsedDocument:
    path: Path
    content_hash: str
    pages: list[str]
    paragraphs: list[ParsedParagraph] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


class PyMuPDFParser:
    section_pattern = re.compile(r"^\s*(\d+(\.\d+)*\s+)?[A-Z][A-Za-z0-9 ,:/()-]{3,80}$")
    caption_pattern = re.compile(r"^(Table|Figure|Fig\.)\s+\d+", re.IGNORECASE)

    def parse(self, path: Path) -> ParsedDocument:
        resolved = path.expanduser().resolve()
        content_hash = sha256_file(resolved)
        pages: list[str] = []
        paragraphs: list[ParsedParagraph] = []
        current_section: str | None = None
        in_references = False
        with fitz.open(resolved) as document:
            metadata = dict(document.metadata or {})
            for page_index, page in enumerate(document, start=1):
                page_text = page.get_text("text")
                pages.append(page_text)
                blocks = page.get_text("blocks")
                paragraph_index = 0
                for block in blocks:
                    if len(block) < 5:
                        continue
                    text = _clean_text(str(block[4]))
                    if not text:
                        continue
                    if text.lower() in {"references", "bibliography"}:
                        in_references = True
                        current_section = "References"
                    elif self._looks_like_section(text):
                        current_section = text[:120]
                    caption = self.caption_pattern.match(text)
                    paragraphs.append(
                        ParsedParagraph(
                            page=page_index,
                            paragraph_index=paragraph_index,
                            text=text,
                            section=current_section,
                            bbox=(
                                float(block[0]),
                                float(block[1]),
                                float(block[2]),
                                float(block[3]),
                            ),
                            is_reference=in_references,
                            caption_type=caption.group(1).lower() if caption else None,
                        )
                    )
                    paragraph_index += 1
        return ParsedDocument(
            path=resolved,
            content_hash=content_hash,
            pages=pages,
            paragraphs=paragraphs,
            metadata=metadata,
        )

    def _looks_like_section(self, text: str) -> bool:
        if len(text.split()) > 12:
            return False
        return bool(self.section_pattern.match(text)) or text.lower() in {
            "abstract",
            "introduction",
            "method",
            "methods",
            "experiments",
            "results",
            "discussion",
            "conclusion",
            "references",
        }


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

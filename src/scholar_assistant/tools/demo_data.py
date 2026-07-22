from __future__ import annotations

from scholar_assistant.schemas.paper import AccessType, Paper, PaperRole, PaperVersion, VersionType


def offline_demo_papers(question: str) -> tuple[list[Paper], list[PaperVersion]]:
    """Return explicitly marked demo metadata for offline smoke tests."""
    topics = [
        (
            "MemGPT: Towards LLMs as Operating Systems",
            2023,
            PaperRole.FOUNDATION,
            "Demo metadata: memory tiering for LLM agents; verify with live arXiv.",
        ),
        (
            "Generative Agents: Interactive Simulacra of Human Behavior",
            2023,
            PaperRole.FOUNDATION,
            "Demo metadata: agent memory, reflection, and planning; verify live.",
        ),
        (
            "A Survey on the Memory Mechanism of Large Language Model based Agents",
            2024,
            PaperRole.RECENT,
            "Demo metadata: memory mechanisms in LLM agents; verify live.",
        ),
        (
            "Retrieval-Augmented Generation for Large Language Models: A Survey",
            2023,
            PaperRole.MAINSTREAM,
            "Demo metadata: retrieval quality and noise in RAG; verify live.",
        ),
    ]
    papers: list[Paper] = []
    versions: list[PaperVersion] = []
    for title, year, role, abstract in topics:
        paper = Paper(
            title=title,
            authors=[],
            abstract=f"{abstract} Original question: {question}",
            year=year,
            venue="offline_demo",
            source="offline_demo",
            paper_role=role,
            metadata={"warning": "offline demo metadata, not a verified bibliographic record"},
        )
        version = PaperVersion(
            work_id=paper.work_id,
            version_type=VersionType.DEMO,
            access_type=AccessType.ABSTRACT_ONLY,
            source_url=None,
            is_canonical=True,
            is_latest=True,
        )
        papers.append(paper)
        versions.append(version)
    return papers, versions

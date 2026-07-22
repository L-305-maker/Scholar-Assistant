from __future__ import annotations

import os

import pytest

from scholar_assistant.tools.arxiv import ArxivClient


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_arxiv_search_smoke() -> None:
    if os.environ.get("SCHOLAR_RUN_LIVE_TESTS") != "1":
        pytest.skip("set SCHOLAR_RUN_LIVE_TESTS=1 to run live API smoke tests")
    result = await ArxivClient(timeout_seconds=20).search(
        "LLM agent memory retrieval",
        max_results=1,
    )
    assert result.papers

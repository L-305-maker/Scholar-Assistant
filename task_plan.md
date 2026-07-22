# Task Plan: Scholar-Assistant MVP

## Goal
Build a runnable local Python MVP for Scholar-Assistant with uv, CLI, SQLite storage, provider abstraction, arXiv search, retrieval fallback, PDF evidence extraction, research orchestration, MCP server entrypoint, docs, and tests.

## Scope Boundaries
- Implement a real end-to-end local path, not pseudocode.
- Keep API keys environment-only.
- Do not depend on Docker, PostgreSQL, Redis, Elasticsearch, Neo4j, cloud services, or Google Scholar scraping.
- Make BGE retrieval optional and lazy; BM25/FTS5 must work without ML models.
- Keep MVP compact enough to verify in this repository.

- [x] P0 Phase 1: Inspect empty repo and initialize uv Python package.
- [x] P0 Phase 2: Add schemas, settings, events, SQLite migrations, and repositories.
- [x] P0 Phase 3: Add provider layer with OpenAI-compatible DeepSeek/Kimi config presets and tests.
- [x] P0 Phase 4: Add arXiv search, dedup/version merge, FTS5 BM25, RRF, optional embedding/reranker fallback, and diversity selection.
- [x] P0 Phase 5: Add PDF parser, evidence extraction, claim quality gate, reader/searcher/analyst/verifier, and orchestrator state machine.
- [x] P0 Phase 6: Add Typer CLI commands, exec JSONL mode, run artifacts, status/resume/export, and MCP server.
- [x] P1 Phase 7: Add README, AGENTS, architecture/data-model docs, and env example.
- [x] P0 Phase 8: Run uv sync, tests, ruff, help, init, provider, exec, and research smoke checks.

## Decisions Made
- Empty project: initialize from scratch, preserving existing empty directories.
- MCP implementation: use the official `mcp` package if installed; provide a structured fallback error if unavailable.
- Codex client: add a disabled-by-default interface and controlled CLI adapter, without making it core research logic.
- Dense retrieval: lazy optional adapter using FlagEmbedding when installed; otherwise BM25-only with explicit retrieval mode.

## Errors Encountered
- `python -m py_compile ...` failed because the environment has no `python` executable. Use `python3` or `uv run python`.
- First `uv sync` failed because `/home/cforever/.cache/uv` is read-only. Resolved by using `UV_CACHE_DIR=.uv-cache`.
- Sandbox DNS blocked PyPI on the first dependency install. Resolved by rerunning `UV_CACHE_DIR=.uv-cache uv sync` with approved network access.
- Added `uv.toml` with `cache-dir = ".uv-cache"` so bare `uv sync` works in this read-only-home environment.
- Initial Claim schema validation did not reject default empty evidence lists for `paper_fact`. Resolved by replacing the field validator with a model-level validator.
- Initial `ruff check .` found import, line-length, and Chinese punctuation issues. Resolved by running `ruff format`, ignoring `RUF001` for Chinese report text, and shortening long lines.

## Status
Completed - implementation, tests, lint, CLI smoke checks, docs, and cleanup are done.

# Supplement Plan

## Goal
Close the high-priority gaps from `requirement_audit.md` while keeping the MVP lightweight and runnable without API keys or BGE models.

- [x] P0: Integrate BGE-M3 dense retrieval into the Searcher path with persisted local vector index and BM25 fallback.
- [x] P0: Expand Quality Gate checks for version/page validity and unsafe direct ranking claims.
- [x] P0: Emit complete task/warning JSONL events in orchestrator and exec paths.
- [x] P1: Add Reader structured extraction fields and Analyst comparability/conflict scaffolding.
- [x] P1: Add tests for provider 429/403/timeout, dense retrieval path, quality gate page/version checks, task events, and CLI JSONL.
- [x] P0: Run `uv sync`, `uv run pytest`, `uv run ruff check .`, and CLI smoke checks.

## Constraints
- Keep API keys environment-only.
- Keep BGE optional and lazy-loaded.
- Do not add large frameworks or services.
- Do not execute arbitrary shell from CLI or MCP.

## Status
Completed - supplements implemented and verification passed.

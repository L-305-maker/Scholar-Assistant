# Notes: Phase 2 Optimization

## Baseline
- `uv sync`: passed before code changes.
- `uv run pytest`: 20 passed before code changes.
- `uv run ruff check .`: passed before code changes.
- `uv run scholar --help`: passed before code changes.

## Implementation Findings
- Searcher previously called arXiv directly and did not use OpenAlex, Crossref, or Semantic Scholar in the default recall path.
- Source clients returned `Paper` or arXiv-specific structures, so a normalization boundary was needed before fusion.
- SQLite schema version was `1`; provenance, source identifiers, duplicate candidates, tool executions, budget usage, and run manifests had no dedicated tables.
- `ToolRegistry` and `BudgetManager` existed but were narrow helper classes, not execution controls.
- BGE adapters were lazy but lacked configurable cache/device/batch/revision metadata and marker-gated smoke paths.
- MCP `--list-tools` worked, but no SDK client smoke path verified stdio initialize and repeated calls.

## Implemented Evidence
- Added unified literature source models and adapters in `src/scholar_assistant/tools/sources.py`.
- Searcher now concurrently searches enabled sources, isolates source failures, stores provenance, and fuses source/query rankings before BM25/dense/reranker selection.
- Added canonicalization and conservative duplicate decisions in `src/scholar_assistant/storage/canonicalization.py`.
- Added schema migration v2 for provenance, identifiers, duplicate candidates, tool executions, budget usage, and run manifests.
- Added package-level MCP stdio smoke helper and manual script.
- Added marker-gated model smoke path; default tests skip real model execution.

## Verification So Far
- `uv run ruff check .`: passed after first implementation round.
- `uv run pytest`: 27 passed, 1 skipped after first implementation round.

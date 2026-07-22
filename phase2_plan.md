# Task Plan: Phase 2 Optimization

## Goal
Deliver a second-stage incremental optimization of the runnable Scholar-Assistant MVP with code changes, tests, verification, and `phase2_optimization_report.md`.

## Phases
- [x] Phase 1: Read required docs, current test layout, and establish baseline scope.
- [x] Phase 2: Run baseline commands and record results.
- [x] Phase 3: Inspect source code around Searcher, sources, dedup, BGE, MCP, ToolRegistry, BudgetManager, migrations, and CLI.
- [x] Phase 4: Implement P0 changes with minimal compatible edits.
- [x] Phase 5: Add focused offline tests and marker-gated live/model tests.
- [x] Phase 6: Update docs, audit, and phase2 report.
- [x] Phase 7: Run required verification and smoke tests.

## P0 Work Items
- Multi-source Searcher fusion for arXiv, OpenAlex, Crossref, and Semantic Scholar with provenance and source failures isolated.
- Conservative canonicalization and deduplication with DOI, arXiv base ID, source ID, title/author/year fuzzy rules, and possible duplicate tracking.
- Real BGE/reranker execution path with optional dependency isolation, smoke script, marker-gated tests, and explicit degradation metadata.
- MCP stdio smoke script and integration coverage for initialize, tools/list, consecutive calls, and structured errors.
- ToolRegistry and BudgetManager as shared controls for external/data/model/tool paths used by CLI, Orchestrator, and MCP.

## Decisions Made
- Keep existing CLI and SQLite compatibility; add migrations instead of rebuilding `state.db`.
- Default test suite remains offline, no API keys, no model downloads.
- Use `retrieval` optional extra as the new explicit name while keeping existing `ml` extra compatible.

## Errors Encountered
- None yet.

## Status
**Completed** - Phase 2 code, tests, documentation, smoke verification, and report are complete with the BGE real-model limitation recorded.

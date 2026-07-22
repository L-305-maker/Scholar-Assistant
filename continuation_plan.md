# Task Plan: Continue Phase 2 Completion

## Goal
Close the highest-priority remaining gaps from `phase2_optimization_report.md` with incremental code, tests, verification, and a short follow-up report.

## Phases
- [x] Phase 1: Re-read communication/coding/planning instructions and inspect current state.
- [x] Phase 2: Stabilize retrieval optional dependency constraints and BGE availability reporting.
- [x] Phase 3: Route Reader PDF download/parse through ToolRegistry/ToolExecutor and BudgetManager.
- [x] Phase 4: Add MCP stdout/stderr protocol-level smoke coverage.
- [ ] Phase 5: Update docs/report and run verification.

## Decisions Made
- Keep default tests offline and model/live paths marker-gated.
- Do not remove or rewrite Phase 2 work; only add targeted fixes.
- Treat real BGE model download as environment-dependent. The immediate code fix is dependency pinning and clearer smoke diagnostics.

## Errors Encountered
- Initial `uv run pytest` / `uv run ruff` attempts failed inside the sandbox because DNS access to PyPI was blocked; reran `UV_CACHE_DIR=.uv-cache uv sync` with approved escalation and completed dependency sync.
- The MCP SDK `FastMCP` and low-level `Server` stdio server did not respond to `initialize` in this environment even for a minimal server. Replaced the runtime entrypoint with an explicit JSON-RPC stdio loop while preserving the MCP tool message shape expected by the Python client.

## Status
**Currently in Phase 5** - Running full verification and updating follow-up documentation.

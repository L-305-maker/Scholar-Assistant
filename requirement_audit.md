# Scholar-Assistant Requirement Audit

## Supplement Status

This audit was followed by a supplementation pass. The previous P0 findings for dense retrieval integration, Quality Gate depth, task/warning JSONL events, and missing provider/dense/quality/event tests have been addressed. See `supplement_report.md` for the current verification record.

Remaining gaps after supplementation:

- OpenAlex, Crossref, and Semantic Scholar clients are still not part of the default Searcher fusion workflow.
- Source-ID and author/year-based dedup are still not complete.
- ToolRegistry and BudgetManager exist but are still not central execution controls.
- Real installed-BGE execution and MCP long-lived stdio startup remain unverified in this environment.

## Original Audit Conclusion

The original codebase was a runnable MVP, but it did not fully satisfy the original requirement set. The largest gaps were dense retrieval integration, stricter evidence verification, required event coverage, richer Reader/Analyst behavior, and test coverage for several mandated edge cases.

## Verified Working Items

- `uv sync`: passed.
- `uv run scholar --help`: passed.
- `uv run pytest`: passed, 17 tests.
- `uv run ruff check .`: passed.
- `scholar init`: passed on `/tmp/scholar-assistant-audit`.
- `scholar providers list`: shows DeepSeek and Kimi with no secret leakage.
- `scholar providers test deepseek`: missing key exits 2 with `DEEPSEEK_API_KEY` error.
- `scholar providers test kimi`: missing key exits 2 with `MOONSHOT_API_KEY` error.
- `SCHOLAR_DEMO_MODE=1 scholar research --no-embeddings ...`: completed and generated report artifacts.
- `scholar exec --json --ephemeral ...`: outputs valid JSONL and avoids persistent run writes.
- `scholar mcp-server --list-tools`: returns 8 MCP tool schemas.

## Complete Or Mostly Complete

- Project skeleton and uv setup are present: `pyproject.toml`, `uv.toml`, `uv.lock`, `src/`, `tests/`, `docs/`.
- CLI commands requested by the spec exist in `src/scholar_assistant/cli/app.py`.
- OpenAI-compatible Provider abstraction exists and is SDK-free.
- DeepSeek/Kimi config presets and environment-only API key behavior exist.
- arXiv Atom client and parser are implemented.
- SQLite state, FTS5 table, migrations, and repositories are implemented.
- Pydantic models for project, paper, version, evidence, claim, hypothesis, task, and run event exist.
- PyMuPDF parser extracts pages, paragraph-like blocks, section heuristics, reference marker, captions, and bounding boxes.
- Markdown report has the required 13 sections plus Evidence Index and Claim Evidence Links.
- MCP exposes the required 8 tools and returns structured JSON-compatible data.

## Unfinished Or Partial Requirements

### P0: Dense Retrieval Is Not Integrated

Requirement: BM25 + BGE-M3 dense retrieval, RRF, then `BAAI/bge-reranker-v2-m3`.

Current state:
- `BGEM3Embedder`, `save_vectors`, `load_vectors`, and `cosine_search` exist in `src/scholar_assistant/retrieval/embeddings.py`.
- `Searcher._retrieve_and_select` only builds BM25 rankings, then optionally calls `BGEReranker`; it never calls `BGEM3Embedder` or `cosine_search`.
- Result: when BGE is installed, dense retrieval still will not run as required; only reranking may run.

Evidence:
- `src/scholar_assistant/agents/searcher.py:158-194`
- `src/scholar_assistant/retrieval/embeddings.py:20-83`

### P0: Quality Gate Is Too Shallow

Requirement: validate real Evidence IDs, `paper_fact` evidence, page belongs to paper version, hypotheses not stable facts, unsupported factual judgments rejected, and no direct superiority ranking under inconsistent experimental conditions.

Current state:
- It checks missing evidence IDs and `paper_fact` support.
- It downgrades verified research hypotheses.
- It does not check whether an evidence page belongs to the version, whether the version exists, or whether claims imply invalid experiment rankings.

Evidence:
- `src/scholar_assistant/core/quality_gate.py:16-39`

### P0: Required JSONL Event Stream Is Incomplete

Requirement: JSONL events include `run.started`, `task.started`, `task.completed`, search/read/evidence/claim/hypothesis events, warnings/errors, and `run.completed`.

Current state:
- Event enum defines `task.started` and `task.completed`.
- Orchestrator does not emit task events.
- `exec --ephemeral` returns only one `run.completed` event.
- Warnings are stored in result/report but not emitted as `warning` events in the orchestrator path.

Evidence:
- `src/scholar_assistant/schemas/events.py`
- `src/scholar_assistant/core/orchestrator.py:75-203`
- `src/scholar_assistant/cli/exec_command.py:28-40`

### P1: Reader Does Not Yet Extract Required Academic Fields

Requirement: Reader extracts research question, method, datasets, metrics, baselines, and results.

Current state:
- Reader extracts evidence paragraphs from PDF or abstract.
- It does not structure research question, method, dataset, metric, baseline, or result fields.
- PDF download/parse errors are swallowed and degraded to abstract without recording the exact failure stage as an event.

Evidence:
- `src/scholar_assistant/agents/reader.py:56-106`
- `src/scholar_assistant/agents/reader.py:123-181`

### P1: Analyst Is Deterministic And Minimal

Requirement: compare papers, identify incomparable experiments, conflicts, hidden assumptions, and generate hypotheses while separating fact/synthesis/inference.

Current state:
- Analyst creates one claim per evidence unit and a fixed cross-paper synthesis/hypothesis template.
- It does not yet inspect experimental comparability, conflicts, hidden assumptions, datasets, or metrics.

Evidence:
- `src/scholar_assistant/agents/analyst.py:23-116`

### P1: OpenAlex, Crossref, Semantic Scholar Are Not In Default Workflow

Requirement: reserve and as much as possible implement OpenAlex/Crossref/Semantic Scholar for metadata and citation relations.

Current state:
- API clients exist.
- Searcher/Orchestrator only call arXiv or offline demo; the extra clients are not registered into default recall/fusion.

Evidence:
- `src/scholar_assistant/tools/openalex.py`
- `src/scholar_assistant/tools/crossref.py`
- `src/scholar_assistant/tools/semantic_scholar.py`
- `src/scholar_assistant/agents/searcher.py:72-100`

### P1: Dedup Does Not Fully Match Required Priority

Requirement: DOI, arXiv ID, source internal ID, normalized title, author set/year/title similarity.

Current state:
- Dedup checks DOI, arXiv ID, normalized title, then title similarity.
- It does not check source internal IDs or author set/year similarity.

Evidence:
- `src/scholar_assistant/storage/repositories.py:107-128`

### P1: Tests Do Not Cover Several Explicit Requirements

Current tests cover real behavior, but several required cases are missing:

- Provider 429.
- Provider timeout.
- Provider 403.
- Reranker optional degradation through the Searcher path.
- Dense retrieval happy path.
- Budget exhaustion.
- Failed retry behavior in Orchestrator.
- Validation failure / partial completion paths.
- MCP real stdio startup and handler error handling.
- stdout/stderr separation for non-ephemeral `exec --json`.

Evidence:
- `tests/unit/test_provider.py:18-177`
- `tests/unit/test_retrieval_and_tools.py:15-75`
- `tests/integration/test_orchestrator_cli_mcp.py:18-70`

### P2: Tool Registry And Budget Manager Are Present But Weakly Integrated

Requirement architecture includes Tool Registry and Budget Manager as active controls.

Current state:
- `ToolRegistry` exists but the main Searcher/Reader/MCP paths do not route tool execution through it.
- `BudgetManager` exists but Orchestrator mostly uses raw `settings.budget` limits directly.

Evidence:
- `src/scholar_assistant/tools/registry.py`
- `src/scholar_assistant/core/budget.py`
- `src/scholar_assistant/core/orchestrator.py:118-127`

## Unverified Due To Environment

- Live arXiv network search was not verified in this audit. The code path exists, and XML parsing is fixture-tested, but current smoke used `SCHOLAR_DEMO_MODE=1`.
- Installed-BGE dense retrieval and reranking were not verified because `doctor` reports `bge_m3: not installed` and `bge_reranker: not installed`.
- Actual long-running `uv run scholar mcp-server` stdio session was not held open; only `--list-tools` was verified.

## Recommended Fix Order

1. Integrate BGE-M3 dense retrieval into `Searcher._retrieve_and_select`, persist/query vectors under `.scholar/index`, and add mocked dense retrieval tests.
2. Expand Quality Gate to validate version existence, page ranges, unsupported factual claims, and direct ranking claims under incomparable experiments.
3. Add task/warning events to Orchestrator and make `exec --json` produce the full required event set.
4. Add tests for provider 429/403/timeout, budget exhaustion, validation failure, partial completion, and MCP error handling.
5. Enrich Reader/Analyst outputs with structured method/dataset/metric/baseline/result fields and comparability checks.

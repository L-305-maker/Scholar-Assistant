# Scholar-Assistant Supplement Report

## Implemented Supplements

- Integrated BGE-M3 dense retrieval into the Searcher path.
  - `Searcher._retrieve_and_select` now adds dense ranking to RRF when embeddings are available.
  - Dense vectors are persisted under `.scholar/index/dense-*.npz`.
  - `--no-embeddings` and missing `FlagEmbedding` still fall back to BM25.
- Expanded Quality Gate.
  - Validates missing evidence IDs.
  - Validates evidence version existence when version context is supplied.
  - Validates evidence work/version match.
  - Validates evidence page does not exceed `PaperVersion.page_count`.
  - Rejects direct superiority/ranking claims unless `experiment_conditions = "comparable"`.
- Expanded run events.
  - Orchestrator now emits `task.started` and `task.completed` for search/read/analyze/verify/report.
  - Search warnings are emitted as `warning` events.
  - `scholar exec --json --ephemeral` emits `run.started`, `task.started`, `task.completed`, and `run.completed`.
- Added lightweight structured reading/analysis support.
  - Reader stores `reading_summary` in `Paper.metadata` and emits it in `paper.read`.
  - Analyst stores structured evidence signals and comparability metadata in Claim metadata.
- Expanded tests from 17 to 20.
  - Added provider 403, 429, and timeout coverage.
  - Added dense retrieval path and local vector-index persistence coverage.
  - Added Quality Gate version/page/direct-ranking coverage.
  - Added Orchestrator task/warning event coverage.

## Verification

- `uv sync`: passed.
- `uv run scholar --help`: passed.
- `uv run pytest`: 20 passed.
- `uv run ruff check .`: passed.
- `uv run scholar init --project-path /tmp/scholar-assistant-supplement`: passed.
- `uv run scholar providers list --project-path /tmp/scholar-assistant-supplement`: passed.
- `uv run scholar exec --json --ephemeral --project-path /tmp/scholar-assistant-supplement "搜索相关论文"`: passed, 4 JSONL events.
- `SCHOLAR_DEMO_MODE=1 uv run scholar research --no-embeddings --project-path /tmp/scholar-assistant-supplement "调研 LLM Agent 长期记忆中的检索噪声问题"`: passed.
- Demo run event counts included `task.started`, `task.completed`, and `warning`.
- `uv run scholar mcp-server --list-tools`: passed.

## Remaining Limits

- Live arXiv network search was not reverified in this sandbox; demo mode and arXiv fixture tests passed.
- Installed `FlagEmbedding` / real BGE model execution was not verified because local `doctor` reports BGE models as not installed.
- OpenAlex, Crossref, and Semantic Scholar clients still are not part of the default Searcher fusion workflow.
- Source-ID and author/year-based dedup are still not complete.
- ToolRegistry and BudgetManager exist but are still not the central execution controls.
- MCP stdio was verified via tool schema listing, not by holding a long-lived stdio session open.

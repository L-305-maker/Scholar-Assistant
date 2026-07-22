# Notes: Scholar-Assistant MVP

## Repository Inspection
- Working directory: `/home/cforever/project/Scholar-Assistant`.
- `rg --files -uu` found only `.gitignore`.
- Existing directories: `.agents`, `.codex`, `.git`, `agent_os`, `skills`, `tests`.
- `git status --short` failed because `.git` is an empty directory, not a valid repository.
- Existing `.gitignore` ignores `.vscode/`, `.venv/`, `.env`, `__pycache__/`, `*.cache`, and `data/`.

## Implementation Notes
- Use `src/scholar_assistant` package layout.
- Use Typer + Rich for CLI, Pydantic v2 for schemas/settings, httpx for API calls, SQLite FTS5 for local search, PyMuPDF for PDF parsing.
- Keep provider SDK-free by calling OpenAI-compatible `/chat/completions` with httpx.
- Network-dependent tests must use mocks/fixtures.

## Verification Log
- `python3 -m compileall -q src tests`: passed.
- `uv sync`: passed after adding project-level `uv.toml`; generated/checked `uv.lock`.
- `uv run pytest`: passed, 17 tests.
- `uv run ruff check .`: passed.
- `uv run scholar --help`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar --help`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar init --project-path /tmp/scholar-assistant-smoke`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar providers list --project-path /tmp/scholar-assistant-smoke`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar providers test deepseek --project-path /tmp/scholar-assistant-smoke`: exited 2 with clear missing `DEEPSEEK_API_KEY` error.
- `UV_CACHE_DIR=.uv-cache uv run scholar exec --json --ephemeral --project-path /tmp/scholar-assistant-smoke "搜索相关论文"`: passed, stdout was one JSONL event.
- `UV_CACHE_DIR=.uv-cache SCHOLAR_DEMO_MODE=1 uv run scholar research --no-embeddings --project-path /tmp/scholar-assistant-smoke "调研 LLM Agent 长期记忆中的检索噪声问题"`: passed, generated run `run_85a3340c288f`.
- Research artifacts present: `events.jsonl`, `research-brief.json`, `search-plan.json`, `papers.json`, `evidence.json`, `claims.json`, `hypotheses.json`, `report.md`.
- `UV_CACHE_DIR=.uv-cache uv run scholar status --project-path /tmp/scholar-assistant-smoke`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar resume run_85a3340c288f --project-path /tmp/scholar-assistant-smoke`: passed.
- `UV_CACHE_DIR=.uv-cache SCHOLAR_DEMO_MODE=1 uv run scholar search --no-embeddings --project-path /tmp/scholar-assistant-smoke "LLM agent memory retrieval noise"`: passed, retrieval mode `bm25-only`.
- `UV_CACHE_DIR=.uv-cache uv run scholar mcp-server --list-tools`: passed, returned 8 tool schemas.
- `UV_CACHE_DIR=.uv-cache uv run scholar doctor --project-path /tmp/scholar-assistant-smoke`: passed; SQLite FTS5, PyMuPDF, MCP ok; BGE models not installed; API keys missing.

## File Counts
- Source/test/doc files under `src`, `tests`, and `docs`: 53.
- Top-level generated lock file: `uv.lock`.
- Project uv config: `uv.toml` sets `cache-dir = ".uv-cache"` for this read-only-home environment.
- Generated Python bytecode caches were removed after verification.

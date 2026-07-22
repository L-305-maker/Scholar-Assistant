# Scholar-Assistant MVP Implementation Report

This file records the final implementation summary, verification commands, and known limits.

## Implemented Capabilities
- uv-managed Python project with `pyproject.toml`, `uv.lock`, Typer CLI, Rich output, Pydantic v2 schemas, SQLite migrations, and tests.
- Project-level `uv.toml` keeps uv cache in `.uv-cache` so `uv sync` works even when home cache is read-only.
- Local project layout: `.scholar/config.toml`, `.scholar/state.db`, runs/cache/index/logs, `papers`, `notes`, and `reports`.
- OpenAI-compatible provider abstraction with DeepSeek/Kimi presets, environment-only API keys, JSON schema output, tool calls, retry, timeout, error classification, token usage, raw response, and `reasoning_content` preservation.
- arXiv Atom search and parser, plus lightweight OpenAlex/Crossref/Semantic Scholar clients.
- Dedup and version merge by DOI/arXiv/title similarity, with `ScholarlyWork` and `PaperVersion`.
- SQLite FTS5 BM25 retrieval, RRF, optional lazy BGE-M3 and BGE reranker adapters, and diversity selection.
- PyMuPDF PDF parser with pages, paragraphs, section heuristics, references marker, captions, and block coordinates.
- Searcher, Reader, Analyst, Verifier, Quality Gate, state machine, and Research Orchestrator.
- Evidence-bound Claim and Hypothesis models. `paper_fact` without Evidence is rejected.
- `scholar` commands: `init`, `doctor`, `config show`, `providers list/test`, `search`, `read`, `compare`, `research`, `exec`, `resume`, `status`, `export`, `mcp-server`.
- Codex-style `exec` supports stdin `-`, `--json`, `--output-schema`, `--ephemeral`, stdout/stderr separation, and stable error events.
- MCP stdio server exposes 8 fixed tools and no arbitrary shell.
- README, architecture doc, data model doc, `.env.example`, and project `AGENTS.md`.

## Verification
- `uv sync`: passed.
- `uv run scholar --help`: passed.
- `uv run pytest`: 17 passed.
- `uv run ruff check .`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar init --project-path /tmp/scholar-assistant-smoke`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar providers list --project-path /tmp/scholar-assistant-smoke`: passed.
- `UV_CACHE_DIR=.uv-cache uv run scholar providers test deepseek --project-path /tmp/scholar-assistant-smoke`: clear missing key error, exit code 2.
- `UV_CACHE_DIR=.uv-cache uv run scholar exec --json --ephemeral --project-path /tmp/scholar-assistant-smoke "搜索相关论文"`: passed.
- `UV_CACHE_DIR=.uv-cache SCHOLAR_DEMO_MODE=1 uv run scholar research --no-embeddings --project-path /tmp/scholar-assistant-smoke "调研 LLM Agent 长期记忆中的检索噪声问题"`: passed with report output.
- `UV_CACHE_DIR=.uv-cache uv run scholar mcp-server --list-tools`: passed.

## Limits
- Live arXiv search is implemented, but smoke verification used `SCHOLAR_DEMO_MODE=1` to avoid sandbox DNS/network delays.
- BGE-M3 and `BAAI/bge-reranker-v2-m3` adapters are implemented but not installed in the default environment; `doctor` reports them as not installed and the CLI falls back to BM25.
- The Analyst is deterministic and evidence-constrained in the MVP; it does not yet call an LLM to produce richer structured comparisons.
- OpenAlex, Crossref, and Semantic Scholar clients are lightweight API adapters without full downstream integration into the default orchestrator path.
- MCP server startup is implemented through the official `mcp` package; smoke verification used `--list-tools` rather than holding an interactive stdio server open.

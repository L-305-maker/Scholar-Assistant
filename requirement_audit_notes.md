# Requirement Audit Notes

## Evidence Log
- Repository contains expected uv project files: `pyproject.toml`, `uv.toml`, `uv.lock`, `README.md`, `AGENTS.md`, `.env.example`, `docs/`, `src/scholar_assistant/`, and `tests/`.
- `implementation_report.md` claims the MVP passed `uv sync`, `uv run pytest`, `uv run ruff check .`, CLI smoke checks, and demo research.
- Current verification passed: `uv sync`, `uv run scholar --help`, `uv run pytest`, `uv run ruff check .`.
- Current smoke checks passed: `scholar init`, `providers list`, `doctor`, `mcp-server --list-tools`, `exec --json --ephemeral`, and demo `research --no-embeddings`.
- DeepSeek/Kimi provider tests without keys both exit with code 2 and print clear missing environment variable errors.

## Open Findings
- Dense retrieval helpers exist in `retrieval/embeddings.py`, but `Searcher._retrieve_and_select` only uses BM25 plus optional reranker; it does not build/query a dense vector index.
- `BGEReranker` availability controls the only optional ML scoring branch. `BGEM3Embedder` is not used in `Searcher`.
- Quality Gate validates evidence IDs and paper_fact evidence presence, but does not validate page belongs to version, version exists, or experimental comparability.
- Orchestrator does not emit `task.started` or `task.completed` events, despite those event types being required/listed.
- Tests do not cover provider 429 or timeout, dense retrieval happy path, MCP real stdio startup, budget exhausted, failed retry, validation failure, or partial completed behavior.
- `Reader._read_known_paper` swallows PDF download/parse exceptions and falls back to abstract without event-level failure detail.
- `exec --ephemeral` correctly avoids persistent writes but returns only a single `run.completed` event, not a full run/task event stream.
- OpenAlex/Crossref/Semantic Scholar clients exist, but the default orchestrator/search path does not call them.

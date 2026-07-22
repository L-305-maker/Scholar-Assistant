# Scholar-Assistant

Scholar-Assistant is a local academic research agent for WSL/Linux. It focuses on evidence-first literature search, reading, comparison, verification, and hypothesis generation.

It is not a paper-writing tool. The workflow is:

```text
Search -> Read -> Think -> Verify -> Search Again
```

The MVP labels conclusions as paper facts, author claims, cross-paper synthesis, agent inference, or research hypotheses. Unsupported `paper_fact` claims are rejected before stable storage.

## Install On WSL

Use Python 3.11 or newer and `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Required smoke commands:

```bash
uv run scholar --help
uv run pytest
uv run ruff check .
```

## Initialize A Project

```bash
uv run scholar init
```

This creates:

```text
.scholar/config.toml
.scholar/state.db
.scholar/runs/
.scholar/cache/
.scholar/index/
.scholar/logs/
papers/
notes/
reports/
```

## Environment Variables

Copy `.env.example` into your shell or local secret manager. Do not write real keys into config files.

```bash
export DEEPSEEK_API_KEY=
export MOONSHOT_API_KEY=
export SEMANTIC_SCHOLAR_API_KEY=
export HF_TOKEN=
```

API keys are read only from environment variables. They are not stored in SQLite, prompts, logs, or reports.

## Configuration

Project config lives at `.scholar/config.toml`. User config can also live at `~/.scholar/config.toml`.

Default provider presets:

```toml
default_model = "deepseek-main"
fast_model = "kimi-fast"
reasoning_model = "deepseek-reasoner"

[providers.deepseek]
type = "openai-compatible"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"

[providers.kimi]
type = "openai-compatible"
base_url = "https://api.moonshot.cn/v1"
api_key_env = "MOONSHOT_API_KEY"
```

Model names are configurable in TOML. Business code never hardcodes DeepSeek or Kimi model IDs.

## CLI Examples

```bash
uv run scholar doctor
uv run scholar config show
uv run scholar providers list
uv run scholar providers test deepseek
uv run scholar search "LLM agent long-term memory retrieval noise" --sources arxiv,openalex --no-embeddings
uv run scholar read papers/example.pdf
uv run scholar research "调研 LLM Agent 长期记忆中的检索噪声问题" --sources arxiv,openalex,crossref,semantic-scholar --no-embeddings
uv run scholar status
uv run scholar export reports/latest-report.md
```

Codex-style non-interactive mode:

```bash
uv run scholar exec "搜索相关论文"
cat prompt.md | uv run scholar exec -
uv run scholar exec --json "搜索相关论文"
uv run scholar exec --output-schema schema.json "生成比较矩阵"
uv run scholar exec --ephemeral "解释一篇论文"
```

`--json` writes JSONL events to stdout. Normal progress goes to stderr.

## Retrieval Modes

Default runnable fallback:

```text
SQLite FTS5 BM25
```

Optional ML retrieval:

```bash
uv sync --extra retrieval
```

The older `ml` extra is still supported for compatibility.

When `FlagEmbedding` and local model files are available, the code can use:

```text
BAAI/bge-m3 for dense retrieval
BAAI/bge-reranker-v2-m3 for final reranking
```

Models are lazy-loaded. CLI startup does not load BGE models. Use `--no-embeddings` to force BM25-only mode.

Model smoke tests are explicit:

```bash
SCHOLAR_RUN_MODEL_TESTS=1 uv run pytest -m model
uv run python scripts/retrieval_smoke.py
```

Without `SCHOLAR_RUN_MODEL_TESTS=1`, model tests skip instead of downloading models.

## Multi-Source Search And Offline Behavior

`scholar search` and `scholar research` use enabled source configuration by default:

- arXiv
- OpenAlex
- Crossref
- Semantic Scholar

Each source has independent timeout, retry, max-results, and weight configuration under `[sources.<name>]`. A failing source emits a warning and does not fail the whole search. Network tests use fixtures and mocks by default.

If live sources are unavailable, the MVP can complete the vertical smoke path with explicitly marked offline demo metadata. Reports warn that demo metadata is not verified bibliographic evidence.

Live API tests are opt-in:

```bash
SCHOLAR_RUN_LIVE_TESTS=1 uv run pytest -m live
```

## PDF Reading

The MVP uses PyMuPDF. It extracts:

- page text
- page numbers
- paragraph-like blocks
- simple section headings
- reference-region markers
- simple table/figure captions
- text block coordinates when available

If full-text parsing fails, the reader degrades to abstract or metadata evidence and marks it as `abstract_only`.

## Codex MCP Registration

This repository was checked against local `codex-cli 0.144.6`. The local help shows:

```bash
codex mcp add <NAME> -- <COMMAND>...
```

Register Scholar-Assistant as a stdio MCP server:

```bash
codex mcp add scholar-assistant -- uv --directory /home/cforever/project/Scholar-Assistant run scholar mcp-server
```

List exposed tools without starting stdio:

```bash
uv run scholar mcp-server --list-tools
```

Run a real stdio initialize/list/call smoke test:

```bash
uv run python scripts/mcp_stdio_smoke.py
```

Exposed MCP tools:

- `scholar_search_papers`
- `scholar_read_paper`
- `scholar_compare_papers`
- `scholar_find_evidence`
- `scholar_get_claims`
- `scholar_run_research`
- `scholar_get_run_status`
- `scholar_export_report`

## Security Boundaries

- API keys are environment-only.
- Logs and provider errors redact API keys.
- Paper, web, and README contents are untrusted data.
- Scholar does not execute commands found in papers or web pages.
- Google Scholar is not scraped.
- Arbitrary shell execution is not exposed by CLI exec or MCP.
- External URL access is limited to registered data-source clients.
- MCP output paths are constrained to the selected project path by default.
- No full-text paper is marked as deeply read unless parsing actually succeeds.
- Agent inference is never written as a paper fact.

## Common Errors

Missing provider key:

```bash
uv run scholar providers test deepseek
# Configuration error: missing environment variable DEEPSEEK_API_KEY
```

BGE unavailable:

```text
retrieval_mode: bm25+optional-ml-unavailable
```

No source network:

```text
warning: No live source results were available; using marked offline demo metadata.
```

## Data Outputs

Each research run writes:

```text
.scholar/runs/<run-id>/
events.jsonl
research-brief.json
search-plan.json
papers.json
evidence.json
claims.json
hypotheses.json
run-manifest.json
report.md
```

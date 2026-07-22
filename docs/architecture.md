# Scholar-Assistant Architecture

## Runtime Shape

```text
Scholar CLI
  -> Research Orchestrator
     -> Research State Machine
     -> Task Planner
     -> Budget Manager
     -> Model Router
     -> Tool Registry
     -> Quality Gate
        -> Searcher / Reader / Analyst / Verifier
           -> SQLite + Local Files + Local Retrieval Index
```

The LLM provider layer can generate or structure content, but the state machine controls workflow order. The Quality Gate decides whether a result can enter stable state.

## State Machine

Normal states:

```text
CREATED -> SCOPING -> SEARCH_PLANNING -> SEARCHING -> SCREENING -> READING -> ANALYZING -> VERIFYING -> REPORTING -> COMPLETED
```

Exceptional states:

```text
FAILED
PARTIALLY_COMPLETED
BUDGET_EXHAUSTED
BLOCKED
```

The MVP uses `ResearchStateMachine` to reject invalid transitions.

## Provider Layer

`ModelProvider` is an async protocol:

```python
async def complete(request: ModelRequest) -> ModelResponse: ...
```

`OpenAICompatibleProvider` handles:

- custom base URL
- custom model name
- API key environment variable
- text response
- tool calls
- JSON schema response format
- timeout
- exponential backoff retry
- 401, 403, 429, 5xx classification
- token usage
- raw response retention
- DeepSeek-style `reasoning_content` in `provider_state`

Business logic depends on the provider protocol, not vendor SDK objects.

## Retrieval Flow

```text
multi-query recall
  -> paper dedup and version merge
  -> SQLite FTS5 BM25
  -> optional BGE-M3 dense retrieval
  -> reciprocal rank fusion
  -> optional bge-reranker-v2-m3
  -> diversity selection
```

The default path is BM25-only. BGE-M3 and the reranker are lazy optional adapters. If `FlagEmbedding` is unavailable or `--no-embeddings` is used, the CLI reports the fallback retrieval mode.

## Evidence, Claims, Hypotheses

Stable conclusions are separated into:

- `paper_fact`
- `author_claim`
- `cross_paper_synthesis`
- `agent_inference`
- `research_hypothesis`

`paper_fact` must cite at least one real `EvidenceUnit`. A `research_hypothesis` cannot be upgraded to verified fact by the Quality Gate.

## Storage

Project data lives under:

```text
.scholar/state.db
.scholar/runs/
.scholar/cache/
.scholar/index/
papers/
notes/
reports/
```

SQLite stores structured state. Local files store raw API responses, parsed artifacts, JSONL events, and reports. Migrations are idempotent and tracked in `schema_migrations`.

## MCP

`uv run scholar mcp-server` starts a stdio MCP server using the official MCP Python package when installed.

The MCP server exposes fixed Scholar tools only. It does not expose arbitrary shell execution and does not return full PDFs through MCP responses.

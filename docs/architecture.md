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
  -> concurrent source search
  -> source normalization and provenance storage
  -> conservative dedup and version merge
  -> SQLite FTS5 BM25
  -> optional BGE-M3 dense retrieval
  -> reciprocal rank fusion
  -> optional bge-reranker-v2-m3
  -> diversity selection
```

Default source search includes arXiv, OpenAlex, Crossref, and Semantic Scholar when enabled in config. Each source has its own timeout, retry count, max-results, and weight. Source failures produce warning events and source statistics instead of failing the whole search.

The default runnable ranking fallback is BM25-only. BGE-M3 and the reranker are lazy optional adapters. If `FlagEmbedding` is unavailable or `--no-embeddings` is used, the CLI reports the fallback retrieval mode.

## Canonicalization

The repository uses a conservative merge pipeline:

- exact merge by normalized DOI, arXiv base ID, or same-source internal ID;
- strong fuzzy merge by high title similarity, author overlap, compatible year, and no identifier conflict;
- possible duplicates are recorded but not automatically merged;
- conflicting DOI or arXiv IDs are never merged automatically.

Source provenance is stored separately from canonical metadata so Crossref publication data does not overwrite stronger arXiv or Semantic Scholar relevance evidence.

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

Schema v2 adds source hits, retrieval provenance, work identifiers, paper aliases, duplicate candidates, tool executions, budget usage, and run manifests. Each run writes `run-manifest.json` with retrieval mode, source stats, budget usage, warnings, and artifact paths.

## Tool And Budget Controls

External and tool-like operations are registered with `ToolRegistry` metadata:

- permission level from T0 local read to T4 high risk;
- timeout and retry policy;
- network domains and side-effect level;
- input and output schemas.

`BudgetManager` checks budgets before tool execution and records actual consumption after execution. Search source calls, raw candidates, deep reads, retries, dense retrieval, reranker use, MCP calls, and cache hits are tracked in the run manifest.

## MCP

`uv run scholar mcp-server` starts a stdio MCP server using the official MCP Python package when installed.

The MCP server exposes fixed Scholar tools only. It does not expose arbitrary shell execution and does not return full PDFs through MCP responses.

`uv run python scripts/mcp_stdio_smoke.py` starts a real stdio session, initializes the protocol, lists tools, performs consecutive tool calls, and exits cleanly. MCP path arguments are resolved with `pathlib` and write outputs are constrained to the selected project path by default.

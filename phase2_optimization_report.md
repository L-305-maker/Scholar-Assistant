# Phase 2 Optimization Report

## 1. 修改摘要

本阶段在可运行 MVP 上做了增量优化，没有重写 CLI、Provider、Reader 或 Orchestrator 的基本结构。

主要完成：

- 默认 Searcher 接入 arXiv、OpenAlex、Crossref、Semantic Scholar 四来源。
- 增加统一 `LiteratureSource`/`SourceSearchRequest`/`SourceHit`/`SourceSearchResponse` 边界。
- 增加并发 source search、来源级 timeout/retry、失败隔离、source stats 和 warning。
- 增加 provenance 保存：`source_hits`、`retrieval_provenance`、paper metadata。
- 增加 DOI、arXiv base ID、source ID、标题、作者、年份的保守规范化与去重。
- 增加 SQLite schema v2 幂等迁移。
- 扩展 BGE-M3/reranker wrapper 的 device/cache/batch/revision/timing metadata 和 smoke path。
- 增加 MCP stdio SDK client smoke，验证 initialize、tools/list、连续工具调用和 demo research。
- 扩展 ToolRegistry、ToolExecutor、BudgetManager，并接入 source search、MCP/research 主路径和 run manifest。
- 增强 Quality Gate：跨论文归纳、hypothesis claim、agent inference、实验可比性、abstract-only 证据强度。
- 新增 11 个测试；默认测试从 20 个扩展为 31 个。

## 2. 基线检查结果

修改前基线：

- `uv sync`: passed.
- `uv run pytest`: 20 passed.
- `uv run ruff check .`: passed.
- `uv run scholar --help`: passed.

## 3. 架构变化

新增/扩展的关键文件：

- `src/scholar_assistant/tools/sources.py`: 统一文献 source 接口、四来源 adapter、source hit 到 Paper/Version 的 normalizer。
- `src/scholar_assistant/storage/canonicalization.py`: DOI/arXiv/source ID/title/author/year 规范化和 duplicate decision。
- `src/scholar_assistant/tools/registry.py`: 权限等级、tool metadata、async executor、timeout/retry、执行记录。
- `src/scholar_assistant/core/budget.py`: 预算执行前检查、调用后计数、重试、cache hit、snapshot。
- `src/scholar_assistant/retrieval/model_smoke.py`: 小语料 BGE/reranker smoke helper。
- `src/scholar_assistant/mcp/stdio_smoke.py`: MCP stdio SDK client smoke helper。
- `scripts/retrieval_smoke.py`: 手工 BGE smoke 入口。
- `scripts/mcp_stdio_smoke.py`: 手工 MCP stdio smoke 入口。

保留兼容：

- 原有 `scholar` 命令仍存在。
- 旧 `ml` optional extra 保留，同时新增 `retrieval` extra。
- schema v1 数据库通过 migration v2 升级，不删除旧表。

## 4. 多数据源融合实现

默认启用来源配置：

- `arxiv`
- `openalex`
- `crossref`
- `semantic_scholar`

CLI 新增：

```bash
uv run scholar search "LLM agent memory" --sources arxiv,openalex,crossref,semantic-scholar
uv run scholar research "LLM agent memory" --sources arxiv,openalex
```

融合流程：

```text
query plan
  -> query x source concurrent search
  -> SourceHit normalization
  -> source_hits / retrieval_provenance storage
  -> conservative dedup
  -> source/query weighted RRF
  -> BM25
  -> optional BGE-M3 dense
  -> optional reranker
  -> diversity selection
```

live 多源 CLI smoke：

```bash
uv run scholar search \
  --project-path /tmp/scholar-assistant-phase2 \
  --sources arxiv,openalex,crossref,semantic-scholar \
  --max-results 1 \
  --no-embeddings \
  "LLM agent memory retrieval noise"
```

结果：

- exit code 0.
- `retrieval_mode: bm25-only`.
- arXiv/OpenAlex/Crossref/Semantic Scholar 都有 provenance 记录。
- Semantic Scholar 有 4 个 query 失败，生成 warning，但整体 search 未失败。
- `source-stats.json` 记录：arXiv 6、OpenAlex 6、Crossref 6、Semantic Scholar 2，Semantic Scholar failures 4。

## 5. 去重与版本合并规则

实现规则：

- DOI normalization: 去除 `https://doi.org/`、`doi:`，casefold，去尾部空白和标点。
- arXiv normalization: 解析 base ID 与 version，例如 `2401.12345v2 -> 2401.12345`.
- source ID normalization: 使用 `source:id` 命名空间，避免不同来源同 ID 误合并。
- title normalization: Unicode NFKC、HTML entity、casefold、空白、标点、LaTeX 简单标记、连字符和冒号处理。
- author normalization: normalized name、surname、initials、order。

合并层级：

- exact merge: normalized DOI 相同、arXiv base ID 相同、同一来源 source ID 相同。
- strong fuzzy merge: 标题高度相似、作者重叠或第一作者匹配、年份差不超过 1、无 DOI/arXiv 冲突。
- possible duplicate: 记录候选，不自动合并。
- never merge: DOI/arXiv 冲突或证据不足。

测试覆盖：

- DOI URL 与裸 DOI。
- arXiv v1/v2。
- source ID 合并。
- 标题标点/副标题差异。
- 作者缩写和年份差。
- 冲突 DOI never merge。
- possible duplicate 不自动合并。

## 6. BGE 真实路径实现和验证结果

实现：

- `BGEM3Embedder` 支持 model name、device、cache_dir、batch_size、max_length、revision、CPU fallback。
- `BGEReranker` 支持同类配置。
- 记录 load/inference time、device、vector dim、pair count。
- `encode_async` / `rerank_async` 使用线程池包装阻塞推理。
- `doctor --deep` 和 `scripts/retrieval_smoke.py` 可显式验证。
- `pytest -m model` 需要 `SCHOLAR_RUN_MODEL_TESTS=1`。

实际验证：

```bash
uv sync --extra retrieval
```

结果：

- passed.
- 安装 69 个 optional retrieval 包。
- 包括 `flagembedding==1.4.0`、`torch==2.13.0`、`transformers==5.14.1`。
- 安装耗时约 8m49s。

随后运行：

```bash
SCHOLAR_RUN_MODEL_TESTS=1 uv run python scripts/retrieval_smoke.py
```

结果：

- exit code 0.
- status: `degraded`.
- 原因：`ModuleNotFoundError: Could not import module 'AutoModel'. Are this object's requirements defined correctly?`
- 未进入 BAAI/bge-m3 模型下载和真实向量推理阶段。

结论：

- 真实模型执行路径已实现。
- 当前环境的 FlagEmbedding/Transformers 导入链失败，真实 BGE-M3/reranker 排序未验证成功。
- 默认环境已通过 `uv sync` 恢复，optional retrieval 包已卸载。

## 7. MCP STDIO 验证结果

新增：

```bash
uv run python scripts/mcp_stdio_smoke.py
```

验证内容：

- 启动 `uv run scholar mcp-server`。
- MCP initialize。
- tools/list。
- 连续调用：
  - `scholar_get_run_status`
  - `scholar_get_claims`
  - `scholar_run_research`
- 返回结果 JSON 可序列化。
- 会话正常退出，临时项目清理。

结果：

- exit code 0.
- initialized protocol: `2025-11-25`.
- tools_count: 8.
- 三次 tool call 均可序列化。

路径安全：

- MCP read/export 路径使用 `pathlib.resolve()`。
- PDF/path 参数不得逃出 project root。
- export output 默认限制在 project root 内。
- 错误返回结构化 `{ "error": { "type": ..., "message": ... } }`。

## 8. ToolRegistry 和 BudgetManager 覆盖情况

已覆盖：

- source search 通过 `ToolRegistry`/`ToolExecutor`。
- source tool metadata 包含 permission、timeout、retry、network domains、schema、side effect。
- BudgetManager 对 source call、per-source call、raw candidates、deep reads、retries、MCP/model counters 做计数。
- Orchestrator 将 budget snapshot 写入 `run-manifest.json` 和 `budget_usage`。
- MCP 使用相同 Orchestrator/Repository/Searcher 路径。

仍有限制：

- PDF 下载和 PDF 解析仍是 Reader 内部直接调用，已有预算字段，但未全部改成 ToolExecutor wrapper。
- report 写入是本地受控文件写入，没有走 ToolRegistry。

## 9. 数据库迁移

schema version 从 1 升级为 2。

新增表：

- `work_identifiers`
- `paper_aliases`
- `source_hits`
- `retrieval_provenance`
- `duplicate_candidates`
- `tool_executions`
- `budget_usage`
- `run_manifests`

验证：

- 旧 schema v1 升级测试通过。
- `/tmp/scholar-assistant-phase2/.scholar/state.db` 中 `schema_migrations` 为 `[1, 2]`。

## 10. 新增测试

新增测试文件：

- `tests/unit/test_phase2_sources_dedup_budget.py`
- `tests/unit/test_phase2_quality_and_models.py`
- `tests/integration/test_phase2_mcp_stdio.py`
- `tests/integration/test_phase2_live_markers.py`

覆盖：

- 四来源 mock 搜索。
- 单来源失败。
- provenance 保存。
- DOI/arXiv/source ID/title/author/year 去重。
- possible duplicate 不自动合并。
- ToolRegistry 权限拒绝。
- BudgetManager 执行前检查和预算耗尽。
- Quality Gate 跨论文归纳、hypothesis claim、不可比较实验。
- MCP stdio initialize/list/call。
- live/model marker gate。
- schema v1 -> v2 迁移。

## 11. 所有运行命令及结果

基线：

- `uv sync`: passed.
- `uv run pytest`: 20 passed.
- `uv run ruff check .`: passed.
- `uv run scholar --help`: passed.

最终默认验证：

- `uv sync`: passed; restored default env and uninstalled 69 optional retrieval packages.
- `uv run ruff check .`: passed.
- `uv run pytest`: 29 passed, 2 skipped, 3 warnings.
- `uv run scholar --help`: passed.
- `uv run scholar doctor --project-path /tmp/scholar-assistant-phase2`: passed.
  - SQLite FTS5 ok.
  - sources enabled: arxiv, openalex, crossref, semantic_scholar.
  - BGE/reranker unavailable in default env because `FlagEmbedding` is not installed.
- `uv run scholar mcp-server --list-tools`: passed, 8 tools.
- `rg -n "sk-[A-Za-z0-9]" .`: no matches.
- `rg -n "API_KEY|api_key" src tests README.md .env.example pyproject.toml`: only environment variable names, test dummy values, and environment reads.

离线 demo：

```bash
uv run scholar init --project-path /tmp/scholar-assistant-phase2
SCHOLAR_DEMO_MODE=1 uv run scholar research \
  --project-path /tmp/scholar-assistant-phase2 \
  --no-embeddings \
  "调研 LLM Agent 长期记忆中的检索噪声问题"
```

结果：

- status: `COMPLETED`.
- papers: 4.
- evidence: 4.
- claims: 5.
- hypotheses: 1.
- retrieval_mode: `bm25-only`.
- run artifacts: `events.jsonl`, `research-brief.json`, `search-plan.json`, `papers.json`, `evidence.json`, `claims.json`, `hypotheses.json`, `run-manifest.json`, `report.md`.
- events: 38 lines.

MCP:

- `uv run python scripts/mcp_stdio_smoke.py`: passed.

Budget exhausted:

```bash
SCHOLAR_DEMO_MODE=1 uv run scholar research \
  --project-path /tmp/scholar-assistant-phase2 \
  --no-embeddings \
  --max-candidates 0 \
  "budget exhausted smoke"
```

结果：

- status: `BUDGET_EXHAUSTED`.
- warning: `Budget exhausted: raw candidate budget exhausted`.
- `run-manifest.json` exists.
- `resume` returns `BUDGET_EXHAUSTED`.

Live API:

- `SCHOLAR_RUN_LIVE_TESTS=1 uv run pytest -m live`: 1 passed, 29 deselected.
- live arXiv smoke succeeded.

Model:

- `uv sync --extra retrieval`: passed.
- `SCHOLAR_RUN_MODEL_TESTS=1 uv run pytest -m model`: 1 passed, 30 deselected.
- `SCHOLAR_RUN_MODEL_TESTS=1 uv run python scripts/retrieval_smoke.py`: degraded due FlagEmbedding import dependency error.

## 12. 已知限制

- 当前环境未完成真实 BGE-M3/reranker 推理验证；失败发生在 FlagEmbedding 导入依赖链，而不是模型排序断言。
- Semantic Scholar live smoke 中部分 query 返回 ToolExecutionError；系统按 warning 记录并继续融合其他来源。
- PDF 下载/解析还没有完全封装为 ToolExecutor tool。
- MCP server 的底层 FastMCP 会输出 INFO 日志；在 SDK smoke 中协议调用成功，命令输出捕获会同时显示 server 日志和脚本 JSON。
- Live 多源 API 排名质量仍是 MVP 级，Crossref 低相关结果已通过 source weight 降权，但还没有领域级 query refinement。

## 13. 未完成内容及准确原因

- 真实 BGE-M3/reranker 成功推理：未完成。原因是 `FlagEmbedding` import 失败：`AutoModel` 依赖无法导入；没有进入 Hugging Face 模型下载阶段。
- PDF 下载/解析 ToolExecutor 全覆盖：部分完成。Reader 仍直接调用 httpx/PyMuPDF；预算字段存在，但未全部路由到 ToolExecutor。
- MCP 协议 stdout/stderr 字节级断言：部分完成。SDK stdio 会话可初始化和连续调用；当前测试没有逐字节断言 server stdout 仅协议消息。

## 14. 后续三个最高优先级任务

1. 修复 retrieval extra 的 FlagEmbedding/Transformers 兼容版本，完成真实 BGE-M3 和 bge-reranker-v2-m3 排序断言。
2. 将 Reader 的 PDF download/parse 完全接入 ToolRegistry/ToolExecutor，并把 parse/download budget 写入 manifest。
3. 增加 MCP stdout/stderr 字节级协议测试，并将 FastMCP INFO 日志固定到 stderr。

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    type: str = "openai-compatible"
    base_url: str
    api_key_env: str
    timeout_seconds: float = 30.0
    max_retries: int = 2

    @property
    def has_api_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))


class ModelConfig(BaseModel):
    provider: str
    model: str
    tool_calling: bool = True
    structured_output: bool = True
    preserve_provider_state: bool = False


class SourceConfig(BaseModel):
    enabled: bool = True
    max_results: int = Field(default=100, ge=1)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=1, ge=0)
    weight: float = Field(default=1.0, gt=0)
    api_key_env: str | None = None
    ttl_seconds: int = Field(default=86_400, ge=0)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key_env and os.environ.get(self.api_key_env))


class RetrievalModelConfig(BaseModel):
    bge_m3_model: str = "BAAI/bge-m3"
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    model_revision: str | None = None
    device: str = "auto"
    cache_dir: Path | None = None
    batch_size: int = Field(default=8, ge=1)
    max_length: int = Field(default=8192, ge=1)
    allow_cpu_fallback: bool = True


class BudgetConfig(BaseModel):
    main_search_loops: int = 2
    verification_search_loops: int = 1
    max_raw_candidates: int = 300
    max_rerank_candidates: int = 100
    max_core_papers: int = 15
    max_deep_read: int = 10
    max_core_claims: int = 30
    max_hypotheses: int = 5
    max_search_calls: int = 64
    max_source_calls_per_source: int = 32
    max_pdf_downloads: int = 20
    max_pdf_parses: int = 20
    max_dense_retrievals: int = 8
    max_reranker_documents: int = 100
    max_mcp_tool_calls: int = 64
    max_model_requests: int = 32
    max_retries: int = 32
    max_input_tokens: int = 250_000
    max_output_tokens: int = 50_000


class ScholarSettings(BaseModel):
    default_model: str = "deepseek-main"
    fast_model: str = "kimi-fast"
    reasoning_model: str = "deepseek-reasoner"
    demo_mode: bool = False
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    sources: dict[str, SourceConfig] = Field(default_factory=dict)
    retrieval: RetrievalModelConfig = Field(default_factory=RetrievalModelConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @classmethod
    def defaults(cls) -> ScholarSettings:
        return cls(
            providers={
                "deepseek": ProviderConfig(
                    base_url="https://api.deepseek.com",
                    api_key_env="DEEPSEEK_API_KEY",
                ),
                "kimi": ProviderConfig(
                    base_url="https://api.moonshot.cn/v1",
                    api_key_env="MOONSHOT_API_KEY",
                ),
            },
            sources={
                "arxiv": SourceConfig(max_results=100, timeout_seconds=30.0, weight=1.2),
                "openalex": SourceConfig(max_results=100, timeout_seconds=30.0, weight=0.9),
                "crossref": SourceConfig(max_results=50, timeout_seconds=30.0, weight=0.6),
                "semantic_scholar": SourceConfig(
                    max_results=100,
                    timeout_seconds=30.0,
                    weight=1.0,
                    api_key_env="SEMANTIC_SCHOLAR_API_KEY",
                ),
            },
            models={
                "deepseek-main": ModelConfig(provider="deepseek", model="deepseek-chat"),
                "deepseek-reasoner": ModelConfig(
                    provider="deepseek",
                    model="deepseek-reasoner",
                    preserve_provider_state=True,
                ),
                "kimi-fast": ModelConfig(provider="kimi", model="moonshot-v1-8k"),
            },
        )

    @classmethod
    def load(cls, project_path: Path) -> ScholarSettings:
        base = cls.defaults().model_dump(mode="json")
        for path in [
            Path.home() / ".scholar" / "config.toml",
            project_path / ".scholar" / "config.toml",
        ]:
            if path.exists():
                loaded = tomllib.loads(path.read_text(encoding="utf-8"))
                base = _deep_merge(base, loaded)
        if os.environ.get("SCHOLAR_DEMO_MODE") == "1":
            base["demo_mode"] = True
        return cls.model_validate(base)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def default_config_text() -> str:
    return """default_model = "deepseek-main"
fast_model = "kimi-fast"
reasoning_model = "deepseek-reasoner"
demo_mode = false

[budget]
main_search_loops = 2
verification_search_loops = 1
max_raw_candidates = 300
max_rerank_candidates = 100
max_core_papers = 15
max_deep_read = 10
max_core_claims = 30
max_hypotheses = 5
max_search_calls = 64
max_source_calls_per_source = 32
max_pdf_downloads = 20
max_pdf_parses = 20
max_dense_retrievals = 8
max_reranker_documents = 100
max_mcp_tool_calls = 64
max_model_requests = 32
max_retries = 32
max_input_tokens = 250000
max_output_tokens = 50000

[retrieval]
bge_m3_model = "BAAI/bge-m3"
bge_reranker_model = "BAAI/bge-reranker-v2-m3"
device = "auto"
batch_size = 8
max_length = 8192
allow_cpu_fallback = true

[sources.arxiv]
enabled = true
max_results = 100
timeout_seconds = 30
max_retries = 1
weight = 1.2

[sources.openalex]
enabled = true
max_results = 100
timeout_seconds = 30
max_retries = 1
weight = 0.9

[sources.crossref]
enabled = true
max_results = 50
timeout_seconds = 30
max_retries = 1
weight = 0.6

[sources.semantic_scholar]
enabled = true
max_results = 100
timeout_seconds = 30
max_retries = 1
weight = 1.0
api_key_env = "SEMANTIC_SCHOLAR_API_KEY"

[providers.deepseek]
type = "openai-compatible"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"

[providers.kimi]
type = "openai-compatible"
base_url = "https://api.moonshot.cn/v1"
api_key_env = "MOONSHOT_API_KEY"

[models.deepseek-main]
provider = "deepseek"
model = "deepseek-chat"
tool_calling = true
structured_output = true

[models.deepseek-reasoner]
provider = "deepseek"
model = "deepseek-reasoner"
tool_calling = true
structured_output = true
preserve_provider_state = true

[models.kimi-fast]
provider = "kimi"
model = "moonshot-v1-8k"
tool_calling = true
structured_output = true
"""

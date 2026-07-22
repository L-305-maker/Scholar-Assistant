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


class BudgetConfig(BaseModel):
    main_search_loops: int = 2
    verification_search_loops: int = 1
    max_raw_candidates: int = 300
    max_rerank_candidates: int = 100
    max_core_papers: int = 15
    max_deep_read: int = 10
    max_core_claims: int = 30
    max_hypotheses: int = 5


class ScholarSettings(BaseModel):
    default_model: str = "deepseek-main"
    fast_model: str = "kimi-fast"
    reasoning_model: str = "deepseek-reasoner"
    demo_mode: bool = False
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
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

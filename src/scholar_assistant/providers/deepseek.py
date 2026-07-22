from __future__ import annotations

from scholar_assistant.core.config import ProviderConfig
from scholar_assistant.providers.openai_compatible import OpenAICompatibleProvider


def build_deepseek_provider(config: ProviderConfig) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider("deepseek", config)

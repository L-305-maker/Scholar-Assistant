from __future__ import annotations

from scholar_assistant.core.config import ModelConfig, ScholarSettings
from scholar_assistant.providers.base import ModelProvider, ModelRequest, ModelResponse
from scholar_assistant.providers.openai_compatible import OpenAICompatibleProvider


class ModelRouter:
    def __init__(self, settings: ScholarSettings) -> None:
        self.settings = settings
        self._providers: dict[str, ModelProvider] = {}

    def provider_names(self) -> list[str]:
        return sorted(self.settings.providers)

    def model_names(self) -> list[str]:
        return sorted(self.settings.models)

    def get_model_config(self, model_alias: str) -> ModelConfig:
        return self.settings.models[model_alias]

    def get_provider(self, provider_name: str) -> ModelProvider:
        if provider_name not in self._providers:
            config = self.settings.providers[provider_name]
            if config.type != "openai-compatible":
                msg = f"Unsupported provider type: {config.type}"
                raise ValueError(msg)
            self._providers[provider_name] = OpenAICompatibleProvider(provider_name, config)
        return self._providers[provider_name]

    async def complete(self, model_alias: str, request: ModelRequest) -> ModelResponse:
        model_config = self.get_model_config(model_alias)
        provider = self.get_provider(model_config.provider)
        provider_request = request.model_copy(update={"model": model_config.model})
        return await provider.complete(provider_request)

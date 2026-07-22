from __future__ import annotations

import httpx
import pytest
import respx

from scholar_assistant.core.config import ProviderConfig, ScholarSettings
from scholar_assistant.providers.base import (
    ModelMessage,
    ModelRequest,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorType,
)
from scholar_assistant.providers.openai_compatible import OpenAICompatibleProvider


def test_deepseek_kimi_default_config_parse() -> None:
    settings = ScholarSettings.defaults()
    assert settings.providers["deepseek"].base_url == "https://api.deepseek.com"
    assert settings.providers["kimi"].api_key_env == "MOONSHOT_API_KEY"
    assert settings.models["deepseek-reasoner"].preserve_provider_state is True


@pytest.mark.asyncio
async def test_provider_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(
        "deepseek",
        ProviderConfig(base_url="https://api.example.test/v1", api_key_env="DEEPSEEK_API_KEY"),
    )
    with pytest.raises(ProviderConfigurationError):
        await provider.complete(
            ModelRequest(model="demo", messages=[ModelMessage(role="user", content="hi")])
        )


@pytest.mark.asyncio
@respx.mock
async def test_provider_plain_response_and_reasoning_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-token")
    respx.post("https://api.example.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "OK",
                            "reasoning_content": "hidden reasoning state",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )
    )
    provider = OpenAICompatibleProvider(
        "deepseek",
        ProviderConfig(base_url="https://api.example.test/v1", api_key_env="DEEPSEEK_API_KEY"),
    )
    response = await provider.complete(
        ModelRequest(model="deepseek-reasoner", messages=[ModelMessage(role="user", content="hi")])
    )
    assert response.content == "OK"
    assert response.provider_state["reasoning_content"] == "hidden reasoning state"
    assert response.usage.total_tokens == 4


@pytest.mark.asyncio
@respx.mock
async def test_provider_tool_calling_and_structured_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-secret")
    route = respx.post("https://api.kimi.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {},
            },
        )
    )
    provider = OpenAICompatibleProvider(
        "kimi",
        ProviderConfig(base_url="https://api.kimi.test/v1", api_key_env="MOONSHOT_API_KEY"),
    )
    response = await provider.complete(
        ModelRequest(
            model="kimi",
            messages=[ModelMessage(role="user", content="use tool")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
    )
    sent = httpx.Request("GET", "https://unused")
    assert route.calls
    sent = route.calls.last.request
    assert response.tool_calls[0]["function"]["name"] == "search"
    assert b"response_format" in sent.content
    assert b"tools" in sent.content


@pytest.mark.asyncio
@respx.mock
async def test_provider_error_classification_and_secret_redaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-token")
    respx.post("https://api.err.test/chat/completions").mock(
        return_value=httpx.Response(401, text="bad secret-token")
    )
    provider = OpenAICompatibleProvider(
        "deepseek",
        ProviderConfig(base_url="https://api.err.test", api_key_env="DEEPSEEK_API_KEY"),
    )
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete(
            ModelRequest(model="demo", messages=[ModelMessage(role="user", content="hi")])
        )
    assert exc_info.value.error_type == ProviderErrorType.AUTHENTICATION
    assert "secret-token" not in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_provider_403_429_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-token")
    for status_code, expected in [
        (403, ProviderErrorType.PERMISSION),
        (429, ProviderErrorType.RATE_LIMIT),
    ]:
        respx.post(f"https://api.status-{status_code}.test/chat/completions").mock(
            return_value=httpx.Response(status_code, text="error")
        )
        provider = OpenAICompatibleProvider(
            "deepseek",
            ProviderConfig(
                base_url=f"https://api.status-{status_code}.test",
                api_key_env="DEEPSEEK_API_KEY",
                max_retries=0,
            ),
        )
        with pytest.raises(ProviderError) as exc_info:
            await provider.complete(
                ModelRequest(model="demo", messages=[ModelMessage(role="user", content="hi")])
            )
        assert exc_info.value.error_type == expected

    respx.post("https://api.timeout.test/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    provider = OpenAICompatibleProvider(
        "deepseek",
        ProviderConfig(
            base_url="https://api.timeout.test",
            api_key_env="DEEPSEEK_API_KEY",
            max_retries=0,
        ),
    )
    with pytest.raises(ProviderError) as exc_info:
        await provider.complete(
            ModelRequest(model="demo", messages=[ModelMessage(role="user", content="hi")])
        )
    assert exc_info.value.error_type == ProviderErrorType.TIMEOUT


@pytest.mark.asyncio
@respx.mock
async def test_provider_retries_5xx_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-token")
    respx.post("https://api.retry.test/chat/completions").mock(
        side_effect=[
            httpx.Response(500, text="temporary"),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}],
                    "usage": {},
                },
            ),
        ]
    )
    provider = OpenAICompatibleProvider(
        "deepseek",
        ProviderConfig(
            base_url="https://api.retry.test",
            api_key_env="DEEPSEEK_API_KEY",
            max_retries=1,
        ),
    )
    response = await provider.complete(
        ModelRequest(model="demo", messages=[ModelMessage(role="user", content="hi")])
    )
    assert response.content == "OK"

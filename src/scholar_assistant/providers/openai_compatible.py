from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from scholar_assistant.core.config import ProviderConfig
from scholar_assistant.providers.base import (
    ModelRequest,
    ModelResponse,
    ProviderConfigurationError,
    ProviderError,
    ProviderErrorType,
    TokenUsage,
    redact_secret,
)


class OpenAICompatibleProvider:
    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.config = config

    async def complete(self, request: ModelRequest) -> ModelResponse:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            msg = (
                f"Missing API key environment variable {self.config.api_key_env} "
                f"for provider {self.name}"
            )
            raise ProviderConfigurationError(msg)

        payload = self._build_payload(request)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        last_error: ProviderError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)
                if response.status_code >= 400:
                    raise self._error_from_response(response, api_key)
                data = response.json()
                return self._parse_response(data)
            except httpx.TimeoutException as exc:
                last_error = ProviderError(
                    f"Provider {self.name} request timed out",
                    ProviderErrorType.TIMEOUT,
                )
                if attempt >= self.config.max_retries:
                    raise last_error from exc
            except httpx.HTTPError as exc:
                clean = redact_secret(str(exc), api_key)
                last_error = ProviderError(
                    f"Provider {self.name} network error: {clean}",
                    ProviderErrorType.NETWORK,
                )
                if attempt >= self.config.max_retries:
                    raise last_error from exc
            except ProviderError as exc:
                last_error = exc
                if exc.error_type not in {ProviderErrorType.RATE_LIMIT, ProviderErrorType.SERVER}:
                    raise
                if attempt >= self.config.max_retries:
                    raise
            await asyncio.sleep(min(2**attempt, 8) * 0.25)

        if last_error:
            raise last_error
        raise ProviderError("Provider request failed", ProviderErrorType.BAD_RESPONSE)

    def _build_payload(self, request: ModelRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if request.tools:
            payload["tools"] = request.tools
        if request.output_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "scholar_structured_output",
                    "schema": request.output_schema,
                    "strict": True,
                },
            }
        return payload

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                "Provider response did not contain choices[0].message",
                ProviderErrorType.BAD_RESPONSE,
            ) from exc

        usage = data.get("usage") or {}
        provider_state: dict[str, Any] = {}
        if "reasoning_content" in message:
            provider_state["reasoning_content"] = message["reasoning_content"]
        if "reasoning" in message:
            provider_state["reasoning"] = message["reasoning"]

        return ModelResponse(
            content=message.get("content") or "",
            tool_calls=message.get("tool_calls") or [],
            provider_state=provider_state,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
            raw_response=data,
        )

    def _error_from_response(self, response: httpx.Response, api_key: str) -> ProviderError:
        status = response.status_code
        body = redact_secret(response.text[:500], api_key)
        if status == 401:
            kind = ProviderErrorType.AUTHENTICATION
        elif status == 403:
            kind = ProviderErrorType.PERMISSION
        elif status == 429:
            kind = ProviderErrorType.RATE_LIMIT
        elif status >= 500:
            kind = ProviderErrorType.SERVER
        else:
            kind = ProviderErrorType.BAD_RESPONSE
        return ProviderError(f"Provider {self.name} returned HTTP {status}: {body}", kind, status)

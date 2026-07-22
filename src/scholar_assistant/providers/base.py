from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ProviderErrorType(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    TIMEOUT = "timeout"
    NETWORK = "network"
    BAD_RESPONSE = "bad_response"


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        error_type: ProviderErrorType,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code


class ProviderConfigurationError(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__(message, ProviderErrorType.CONFIGURATION)


class ModelMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ModelRequest(BaseModel):
    model: str
    messages: list[ModelMessage]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    output_schema: dict[str, Any] | None = None
    temperature: float = 0.2
    max_output_tokens: int | None = None
    stream: bool = False
    task_type: str | None = None
    reasoning_mode: str | None = None


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ModelResponse(BaseModel):
    content: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    provider_state: dict[str, Any] = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse: ...


def redact_secret(value: str, secret: str | None) -> str:
    if not secret:
        return value
    return value.replace(secret, "[REDACTED]")

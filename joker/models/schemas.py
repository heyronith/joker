"""Schemas for model requests, results, health, and configuration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

T = TypeVar("T", bound=BaseModel)

ProviderHealthStatus = Literal["healthy", "degraded", "unavailable"]


class ProviderHealth(BaseModel):
    """Health snapshot for a model provider."""

    model_config = ConfigDict(frozen=True)

    status: ProviderHealthStatus
    provider_name: str
    available_models: tuple[str, ...] = ()
    detail: str | None = None
    checked_at: datetime

    @field_validator("checked_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("checked_at must be timezone-aware")
        return value


class ModelRequest(BaseModel):
    """Provider-neutral structured completion request."""

    model_config = ConfigDict(frozen=True)

    request_id: UUID
    idempotency_key: str

    role: str
    prompt_id: str
    prompt_version: str

    model_profile: str
    context_payload: dict[str, Any]

    timeout_seconds: float
    max_output_tokens: int
    temperature: float | None = None

    snapshot_id: UUID
    cycle_id: str

    @field_validator("timeout_seconds")
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be > 0")
        return value

    @field_validator("max_output_tokens")
    @classmethod
    def _positive_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_output_tokens must be > 0")
        return value


class ModelResult(BaseModel, Generic[T]):
    """Validated structured output from a model provider."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request_id: UUID
    provider_name: str
    model_name: str
    output: T

    prompt_version: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int
    attempt_count: int = 1
    escalated_from: str | None = None

    completed_at: datetime

    @field_validator("completed_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("completed_at must be timezone-aware")
        return value


class ModelProfileConfig(BaseModel):
    """Configuration for a named model profile."""

    provider: str
    model: str
    enabled: bool = True
    max_output_tokens: int = 1200
    temperature: float | None = None

    @field_validator("max_output_tokens")
    @classmethod
    def _positive_tokens(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_output_tokens must be > 0")
        return value


class OllamaProviderConfig(BaseModel):
    """Ollama adapter configuration."""

    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    request_timeout_seconds: float = 45.0
    max_concurrent_requests: int = 1


class OpenAIProviderConfig(BaseModel):
    """OpenAI adapter configuration."""

    enabled: bool = True
    api_key_env: str = "OPENAI_API_KEY"
    model_env: str = "OPENAI_MODEL"
    request_timeout_seconds: float = 60.0
    store: bool = False


def default_model_profiles() -> dict[str, ModelProfileConfig]:
    """Default Task 2 model profiles (configurable, not hard dependencies)."""
    return {
        "fast_structured": ModelProfileConfig(
            provider="ollama",
            model="qwen3.5:9b",
            max_output_tokens=800,
        ),
        "general_reasoning": ModelProfileConfig(
            provider="ollama",
            model="gemma4:12b",
            max_output_tokens=1200,
        ),
        "independent_critic": ModelProfileConfig(
            provider="ollama",
            model="ministral-3:14b",
            max_output_tokens=1200,
        ),
        "deep_local": ModelProfileConfig(
            provider="ollama",
            model="gemma4:26b",
            enabled=False,
            max_output_tokens=1500,
        ),
        "remote_escalation": ModelProfileConfig(
            provider="openai",
            model="${OPENAI_MODEL}",
            max_output_tokens=1500,
        ),
    }


class ModelsConfig(BaseModel):
    """YAML-like configuration for model providers and profiles."""

    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    profiles: dict[str, ModelProfileConfig] = Field(default_factory=default_model_profiles)

    max_schema_repair_attempts: int = 1
    max_provider_escalations: int = 1
    max_parallel_model_calls: int = 5


def utc_now() -> datetime:
    """Return the current UTC time with timezone info."""
    return datetime.now(timezone.utc)

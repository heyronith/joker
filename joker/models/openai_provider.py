"""OpenAI-backed structured model provider."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from joker.config.validation import redact_secrets
from joker.models.exceptions import (
    ModelConfigurationError,
    ModelProviderUnavailable,
    ModelRefusal,
    ModelResponseEmpty,
    ModelTimeout,
    StructuredOutputFailure,
)
from joker.models.schemas import ModelRequest, ModelResult, OpenAIProviderConfig, ProviderHealth, utc_now

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIModelProvider:
    """Async OpenAI adapter with strict structured outputs."""

    def __init__(
        self,
        config: OpenAIProviderConfig | None = None,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config or OpenAIProviderConfig()
        self._api_key = api_key
        self._client = client
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        value = os.environ.get(self._config.api_key_env, "").strip()
        if not value:
            raise ModelConfigurationError(
                f"OpenAI API key not configured (env {self._config.api_key_env})"
            )
        return value

    def _resolve_model_name(self, configured_model: str) -> str:
        if configured_model == "${OPENAI_MODEL}":
            value = os.environ.get(self._config.model_env, "").strip()
            if not value:
                raise ModelConfigurationError(
                    f"OpenAI model not configured (env {self._config.model_env})"
                )
            return value
        return configured_model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self._resolve_api_key())
        self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the owned AsyncOpenAI client when present; allow later recreate."""
        client = self._client
        if client is None or not getattr(self, "_owns_client", False):
            return
        close = getattr(client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result
        self._client = None
        # Keep ownership so the next _get_client() allocates a fresh client.
        self._owns_client = True

    async def healthcheck(self) -> ProviderHealth:
        """Check OpenAI availability without logging credentials."""
        checked_at = utc_now()
        if not self._config.enabled:
            return ProviderHealth(
                status="unavailable",
                provider_name=self.provider_name,
                available_models=(),
                detail="openai provider disabled in configuration",
                checked_at=checked_at,
            )
        try:
            self._resolve_api_key()
            return ProviderHealth(
                status="healthy",
                provider_name=self.provider_name,
                available_models=(),
                detail="credentials configured",
                checked_at=checked_at,
            )
        except ModelConfigurationError as exc:
            return ProviderHealth(
                status="unavailable",
                provider_name=self.provider_name,
                available_models=(),
                detail=str(exc),
                checked_at=checked_at,
            )

    async def complete_structured(
        self,
        *,
        request: ModelRequest,
        output_type: type[T],
    ) -> ModelResult[T]:
        """Complete a structured request using OpenAI chat parsing."""
        if not self._config.enabled:
            raise ModelProviderUnavailable("openai provider is disabled")

        health = await self.healthcheck()
        if health.status == "unavailable":
            raise ModelProviderUnavailable(health.detail or "openai unavailable")

        configured_model = request.context_payload.get(
            "resolved_model",
            request.model_profile,
        )
        model_name = self._resolve_model_name(str(configured_model))

        system_prompt = request.context_payload.get("system_prompt", "")
        user_prompt = request.context_payload.get("user_prompt")
        if user_prompt is None:
            user_prompt = json.dumps(request.context_payload, default=str)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(user_prompt)})

        client = self._get_client()
        started = time.perf_counter()
        attempt_count = int(request.context_payload.get("attempt_count", 1))
        try:
            response = await client.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=output_type,
                timeout=request.timeout_seconds,
                store=self._config.store,
            )
        except Exception as exc:
            err_name = type(exc).__name__.lower()
            safe = redact_secrets(str(exc))
            if "timeout" in err_name or "timed out" in safe.lower():
                raise ModelTimeout(
                    f"openai request timed out after {request.timeout_seconds}s"
                ) from exc
            logger.warning("openai completion failed", extra={"error_type": err_name})
            raise ModelProviderUnavailable(f"openai request failed: {safe}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        message = response.choices[0].message
        if message.refusal:
            raise ModelRefusal(redact_secrets(f"model refused request: {message.refusal}"))

        parsed = message.parsed
        if parsed is None:
            if not message.content:
                raise ModelResponseEmpty("openai returned empty parsed output")
            try:
                parsed = output_type.model_validate_json(message.content)
            except ValidationError as exc:
                raise StructuredOutputFailure(
                    f"openai output failed validation: {exc}"
                ) from exc

        usage = response.usage
        return ModelResult(
            request_id=request.request_id,
            provider_name=self.provider_name,
            model_name=model_name,
            output=parsed,
            prompt_version=request.prompt_version,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            latency_ms=latency_ms,
            attempt_count=attempt_count,
            escalated_from=request.context_payload.get("escalated_from"),
            completed_at=utc_now(),
        )

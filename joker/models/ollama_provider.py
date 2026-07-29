"""Ollama-backed structured model provider."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from joker.models.exceptions import (
    ModelProviderUnavailable,
    ModelRefusal,
    ModelResponseEmpty,
    ModelTimeout,
    StructuredOutputFailure,
)
from joker.models.schemas import ModelRequest, ModelResult, OllamaProviderConfig, ProviderHealth, utc_now

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaModelProvider:
    """Async Ollama adapter using schema-constrained JSON chat completions."""

    def __init__(
        self,
        config: OllamaProviderConfig | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or OllamaProviderConfig()
        self._client = client
        self._owns_client = client is None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url.rstrip("/"),
                timeout=self._config.request_timeout_seconds,
            )
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client when owned by this provider."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def healthcheck(self) -> ProviderHealth:
        """List installed models via ``/api/tags`` without pulling models."""
        checked_at = utc_now()
        if not self._config.enabled:
            return ProviderHealth(
                status="unavailable",
                provider_name=self.provider_name,
                available_models=(),
                detail="ollama provider disabled in configuration",
                checked_at=checked_at,
            )
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            payload = response.json()
            models = tuple(item.get("name", "") for item in payload.get("models", []) if item.get("name"))
            return ProviderHealth(
                status="healthy" if models else "degraded",
                provider_name=self.provider_name,
                available_models=models,
                detail=None if models else "ollama reachable but no models reported",
                checked_at=checked_at,
            )
        except httpx.HTTPError as exc:
            return ProviderHealth(
                status="unavailable",
                provider_name=self.provider_name,
                available_models=(),
                detail=f"ollama healthcheck failed: {exc}",
                checked_at=checked_at,
            )

    async def complete_structured(
        self,
        *,
        request: ModelRequest,
        output_type: type[T],
    ) -> ModelResult[T]:
        """Complete a structured request against a local Ollama model."""
        if not self._config.enabled:
            raise ModelProviderUnavailable("ollama provider is disabled")

        health = await self.healthcheck()
        if health.status == "unavailable":
            raise ModelProviderUnavailable(health.detail or "ollama unavailable")

        profile_model = request.context_payload.get("resolved_model", request.model_profile)
        if profile_model not in health.available_models and health.available_models:
            raise ModelProviderUnavailable(
                f"ollama model {profile_model!r} is not installed; available={health.available_models}"
            )

        system_prompt = request.context_payload.get("system_prompt", "")
        user_prompt = request.context_payload.get("user_prompt")
        if user_prompt is None:
            user_prompt = json.dumps(request.context_payload, default=str)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": str(user_prompt)})

        body: dict[str, Any] = {
            "model": profile_model,
            "messages": messages,
            "stream": False,
            # Thinking models (e.g. qwen3.5) otherwise fill message.thinking and
            # leave message.content empty, which fails structured validation.
            "think": False,
            "options": {
                "num_predict": request.max_output_tokens,
            },
        }
        if request.temperature is not None:
            body["options"]["temperature"] = request.temperature
        body["format"] = output_type.model_json_schema()

        started = time.perf_counter()
        client = self._get_client()
        timeout = httpx.Timeout(request.timeout_seconds)
        try:
            response = await client.post("/api/chat", json=body, timeout=timeout)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelTimeout(f"ollama request timed out after {request.timeout_seconds}s") from exc
        except httpx.HTTPError as exc:
            raise ModelProviderUnavailable(f"ollama request failed: {exc}") from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        payload = response.json()
        message = payload.get("message") or {}
        content = message.get("content")
        if not content:
            raise ModelResponseEmpty("ollama returned empty message content")

        if message.get("refusal"):
            raise ModelRefusal(str(message["refusal"]))

        try:
            if isinstance(content, str):
                parsed = output_type.model_validate_json(content)
            elif isinstance(content, dict):
                parsed = output_type.model_validate(content)
            else:
                raise StructuredOutputFailure(f"unexpected ollama content type: {type(content).__name__}")
        except ValidationError as exc:
            raise StructuredOutputFailure(f"ollama output failed validation: {exc}") from exc

        prompt_tokens = payload.get("prompt_eval_count")
        output_tokens = payload.get("eval_count")
        attempt_count = int(request.context_payload.get("attempt_count", 1))
        return ModelResult(
            request_id=request.request_id,
            provider_name=self.provider_name,
            model_name=str(profile_model),
            output=parsed,
            prompt_version=request.prompt_version,
            input_tokens=int(prompt_tokens) if prompt_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            latency_ms=latency_ms,
            attempt_count=attempt_count,
            escalated_from=request.context_payload.get("escalated_from"),
            completed_at=utc_now(),
        )

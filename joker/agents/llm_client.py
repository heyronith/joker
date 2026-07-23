"""LLM client interface for structured agent outputs."""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel, ValidationError

from joker.config.validation import redact_secrets

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClientError(Exception):
    """Base error for LLM client failures. Messages are safe for display."""


class LLMTimeoutError(LLMClientError):
    pass


class LLMValidationError(LLMClientError):
    pass


class LLMRefusalError(LLMClientError):
    pass


class LLMAPIError(LLMClientError):
    pass


@dataclass(frozen=True)
class LLMCompletionResult(Generic[T]):
    result: T
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None


@dataclass
class LLMRequestLogEntry:
    request_id: str
    model: str
    schema_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_chars: int = 0
    success: bool = False
    error: str | None = None
    token_usage: dict[str, int | None] = field(default_factory=dict)


class LLMClient(ABC):
    """Structured completion interface for agent council."""

    @abstractmethod
    def complete_structured(
        self,
        prompt: str,
        schema: Type[T],
        metadata: dict[str, Any] | None = None,
        *,
        system_prompt: str | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletionResult[T]:
        ...

    @property
    @abstractmethod
    def request_log(self) -> list[LLMRequestLogEntry]:
        ...


def _log_request_metadata(
    log: list[LLMRequestLogEntry],
    *,
    request_id: str,
    model: str,
    schema: Type[BaseModel],
    metadata: dict[str, Any] | None,
    prompt: str,
    success: bool,
    error: str | None = None,
    token_usage: dict[str, int | None] | None = None,
) -> None:
    safe_metadata = {
        k: redact_secrets(str(v)) if isinstance(v, str) else v
        for k, v in (metadata or {}).items()
    }
    log.append(
        LLMRequestLogEntry(
            request_id=request_id,
            model=model,
            schema_name=schema.__name__,
            metadata=safe_metadata,
            prompt_chars=len(prompt),
            success=success,
            error=redact_secrets(error) if error else None,
            token_usage=token_usage or {},
        )
    )


class MockLLMClient(LLMClient):
    """Deterministic LLM client for offline tests."""

    def __init__(
        self,
        responses: dict[str, Any] | None = None,
        *,
        model: str = "mock-model",
        fail_on_schema: set[str] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self._responses = responses or {}
        self._model = model
        self._fail_on_schema = fail_on_schema or set()
        self._delay_seconds = delay_seconds
        self._log: list[LLMRequestLogEntry] = []

    @property
    def request_log(self) -> list[LLMRequestLogEntry]:
        return list(self._log)

    def set_response(
        self,
        schema: Type[T],
        value: T | dict[str, Any] | str,
        *,
        agent: str | None = None,
    ) -> None:
        key = f"{schema.__name__}:{agent}" if agent else schema.__name__
        self._responses[key] = value

    def _lookup_response(self, schema: Type[T], metadata: dict[str, Any] | None) -> Any:
        if metadata and metadata.get("agent"):
            key = f"{schema.__name__}:{metadata['agent']}"
            if key in self._responses:
                return self._responses[key]
        return self._responses.get(schema.__name__)

    def complete_structured(
        self,
        prompt: str,
        schema: Type[T],
        metadata: dict[str, Any] | None = None,
        *,
        system_prompt: str | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletionResult[T]:
        request_id = str(uuid.uuid4())
        if self._delay_seconds and timeout_seconds and self._delay_seconds > timeout_seconds:
            _log_request_metadata(
                self._log,
                request_id=request_id,
                model=self._model,
                schema=schema,
                metadata=metadata,
                prompt=prompt,
                success=False,
                error="timeout",
            )
            raise LLMTimeoutError("Mock LLM timed out")

        if schema.__name__ in self._fail_on_schema:
            _log_request_metadata(
                self._log,
                request_id=request_id,
                model=self._model,
                schema=schema,
                metadata=metadata,
                prompt=prompt,
                success=False,
                error="invalid structured response",
            )
            raise LLMValidationError("Mock LLM returning invalid structured response")

        raw = self._lookup_response(schema, metadata)
        try:
            if raw is None:
                raise LLMValidationError(
                    f"No mock response configured for schema {schema.__name__}"
                )
            if isinstance(raw, schema):
                parsed = raw
            elif isinstance(raw, str):
                parsed = schema.model_validate_json(raw)
            elif isinstance(raw, dict):
                parsed = schema.model_validate(raw)
            else:
                raise LLMValidationError(f"Unsupported mock response type for {schema.__name__}")
        except ValidationError as exc:
            _log_request_metadata(
                self._log,
                request_id=request_id,
                model=self._model,
                schema=schema,
                metadata=metadata,
                prompt=prompt,
                success=False,
                error=str(exc),
            )
            raise LLMValidationError(f"Invalid structured response: {exc}") from exc

        _log_request_metadata(
            self._log,
            request_id=request_id,
            model=self._model,
            schema=schema,
            metadata=metadata,
            prompt=prompt,
            success=True,
            token_usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )
        return LLMCompletionResult(
            result=parsed,
            model=self._model,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            request_id=request_id,
        )


class OpenAILLMClient(LLMClient):
    """OpenAI-backed structured completion client."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_retries: int = 2,
        default_timeout_seconds: float = 120.0,
        client: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_retries = max_retries
        self._default_timeout = default_timeout_seconds
        self._client = client
        self._log: list[LLMRequestLogEntry] = []

    @property
    def request_log(self) -> list[LLMRequestLogEntry]:
        return list(self._log)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI

        return OpenAI(api_key=self._api_key)

    def complete_structured(
        self,
        prompt: str,
        schema: Type[T],
        metadata: dict[str, Any] | None = None,
        *,
        system_prompt: str | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMCompletionResult[T]:
        from joker.agents.security import reject_forbidden_agent_payload
        from joker.compliance.openai_audit import audit_and_sanitize_openai_context

        request_id = str(uuid.uuid4())
        timeout = timeout_seconds or self._default_timeout
        client = self._get_client()
        safe_metadata, audit = audit_and_sanitize_openai_context(
            metadata or {},
            prompt_type=schema.__name__,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = client.beta.chat.completions.parse(
                    model=self._model,
                    messages=messages,
                    response_format=schema,
                    timeout=timeout,
                )
                message = response.choices[0].message
                if message.refusal:
                    err = f"Model refused request: {message.refusal}"
                    _log_request_metadata(
                        self._log,
                        request_id=request_id,
                        model=self._model,
                        schema=schema,
                        metadata={**safe_metadata, **audit.to_dict()},
                        prompt=prompt,
                        success=False,
                        error=err,
                    )
                    raise LLMRefusalError(redact_secrets(err))

                parsed = message.parsed
                if parsed is None:
                    raw = message.content or ""
                    reject_forbidden_agent_payload(raw)
                    raise LLMValidationError("Model returned empty parsed output")

                usage = response.usage
                token_usage = {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                }
                _log_request_metadata(
                    self._log,
                    request_id=request_id,
                    model=self._model,
                    schema=schema,
                    metadata={**safe_metadata, **audit.to_dict()},
                    prompt=prompt,
                    success=True,
                    token_usage=token_usage,
                )
                return LLMCompletionResult(
                    result=parsed,
                    model=self._model,
                    prompt_tokens=token_usage["prompt_tokens"],
                    completion_tokens=token_usage["completion_tokens"],
                    total_tokens=token_usage["total_tokens"],
                    request_id=request_id,
                )
            except LLMRefusalError:
                raise
            except LLMValidationError:
                raise
            except Exception as exc:
                last_error = exc
                err_name = type(exc).__name__.lower()
                if "timeout" in err_name or "timed out" in str(exc).lower():
                    _log_request_metadata(
                        self._log,
                        request_id=request_id,
                        model=self._model,
                        schema=schema,
                        metadata={**safe_metadata, **audit.to_dict()},
                        prompt=prompt,
                        success=False,
                        error="timeout",
                    )
                    raise LLMTimeoutError(
                        redact_secrets(f"OpenAI request timed out after {timeout}s")
                    ) from exc
                if attempt < self._max_retries:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                safe = redact_secrets(str(exc))
                _log_request_metadata(
                    self._log,
                    request_id=request_id,
                    model=self._model,
                    schema=schema,
                    metadata={**safe_metadata, **audit.to_dict()},
                    prompt=prompt,
                    success=False,
                    error=safe,
                )
                raise LLMAPIError(f"OpenAI API error: {safe}") from exc

        raise LLMAPIError(redact_secrets(str(last_error)))

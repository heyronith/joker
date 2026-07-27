"""Deterministic fake model provider for tests and offline replay."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel, ValidationError

from joker.models.exceptions import (
    ModelProviderUnavailable,
    ModelRefusal,
    ModelResponseEmpty,
    ModelTimeout,
    StructuredOutputFailure,
)
from joker.models.schemas import ModelRequest, ModelResult, ProviderHealth, utc_now
from joker.models.provider import ModelProvider

T = TypeVar("T", bound=BaseModel)
RoleResponseFactory = Callable[[ModelRequest], BaseModel | dict[str, Any] | str]


@dataclass
class FakeCallRecord:
    """Recorded model call for assertions and replay."""

    request: ModelRequest
    output_type_name: str
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    error_code: str | None = None


@dataclass
class FakeModelProvider:
    """Offline model provider with canned responses and failure injection."""

    provider_name: str = "fake"
    model_name: str = "fake-model"
    available: bool = True
    latency_seconds: float = 0.0
    simulate_timeout: bool = False
    simulate_malformed: bool = False
    simulate_failure: bool = False
    simulate_refusal: bool = False
    failure_message: str = "simulated provider failure"
    _by_role: dict[str, Any] = field(default_factory=dict)
    _role_factories: dict[str, RoleResponseFactory] = field(default_factory=dict)
    _by_request_id: dict[UUID, Any] = field(default_factory=dict)
    _idempotency_cache: dict[str, ModelResult[Any]] = field(default_factory=dict)
    _calls: list[FakeCallRecord] = field(default_factory=list)

    def set_canned_for_role(self, role: str, value: BaseModel | dict[str, Any] | str) -> None:
        """Register a canned response for an agent role.

        Clears any role factory for the same role so static and factory
        registrations do not silently compete.
        """
        self._role_factories.pop(role, None)
        self._by_role[role] = value

    def set_role_factory(self, role: str, factory: RoleResponseFactory) -> None:
        """Register a per-invocation factory for an agent role.

        The factory receives the current :class:`ModelRequest` and must return a
        fresh response object. Exact request idempotency is still enforced by the
        provider cache before the factory runs.
        """
        self._by_role.pop(role, None)
        self._role_factories[role] = factory

    def set_canned_for_request_id(
        self,
        request_id: UUID,
        value: BaseModel | dict[str, Any] | str,
    ) -> None:
        """Register a canned response for a specific request ID."""
        self._by_request_id[request_id] = value

    def clear_calls(self) -> None:
        """Clear recorded calls and idempotency cache."""
        self._calls.clear()
        self._idempotency_cache.clear()

    @property
    def calls(self) -> list[FakeCallRecord]:
        """Return a copy of recorded calls."""
        return list(self._calls)

    async def healthcheck(self) -> ProviderHealth:
        """Return fake provider health."""
        return ProviderHealth(
            status="healthy" if self.available else "unavailable",
            provider_name=self.provider_name,
            available_models=(self.model_name,),
            detail=None if self.available else "fake provider disabled",
            checked_at=utc_now(),
        )

    async def complete_structured(
        self,
        *,
        request: ModelRequest,
        output_type: type[T],
    ) -> ModelResult[T]:
        """Return a canned or cached structured response."""
        started_at = utc_now()
        record = FakeCallRecord(
            request=request,
            output_type_name=output_type.__name__,
            started_at=started_at,
        )
        self._calls.append(record)

        if not self.available or self.simulate_failure:
            record.error_code = "provider_unavailable"
            raise ModelProviderUnavailable(self.failure_message)

        cached = self._idempotency_cache.get(request.idempotency_key)
        if cached is not None:
            record.completed_at = utc_now()
            record.success = True
            return ModelResult[T](
                request_id=request.request_id,
                provider_name=cached.provider_name,
                model_name=cached.model_name,
                output=output_type.model_validate(cached.output.model_dump()),
                prompt_version=request.prompt_version,
                input_tokens=cached.input_tokens,
                output_tokens=cached.output_tokens,
                latency_ms=0,
                attempt_count=cached.attempt_count,
                escalated_from=cached.escalated_from,
                completed_at=record.completed_at,
            )

        if self.simulate_timeout:
            record.error_code = "timeout"
            if self.latency_seconds > 0:
                await asyncio.sleep(self.latency_seconds)
            raise ModelTimeout("fake provider simulated timeout")

        if self.latency_seconds > 0:
            await asyncio.sleep(self.latency_seconds)
            elapsed = (utc_now() - started_at).total_seconds()
            if elapsed > request.timeout_seconds:
                record.error_code = "timeout"
                raise ModelTimeout("fake provider exceeded request timeout")

        if self.simulate_refusal:
            record.error_code = "refusal"
            raise ModelRefusal("fake provider simulated refusal")

        raw = self._lookup_raw(request)
        if raw is None:
            record.error_code = "empty_response"
            raise ModelResponseEmpty(
                f"No canned response for role={request.role!r} request_id={request.request_id}"
            )

        if self.simulate_malformed:
            raw = {"__invalid__": True}

        try:
            parsed = self._parse_output(raw, output_type)
        except ValidationError as exc:
            record.error_code = "structured_output_failure"
            raise StructuredOutputFailure(f"fake provider malformed output: {exc}") from exc

        completed_at = utc_now()
        latency_ms = max(0, int((completed_at - started_at).total_seconds() * 1000))
        attempt_count = int(request.context_payload.get("attempt_count", 1))
        result: ModelResult[T] = ModelResult(
            request_id=request.request_id,
            provider_name=self.provider_name,
            model_name=self.model_name,
            output=parsed,
            prompt_version=request.prompt_version,
            input_tokens=10,
            output_tokens=20,
            latency_ms=latency_ms,
            attempt_count=attempt_count,
            escalated_from=request.context_payload.get("escalated_from"),
            completed_at=completed_at,
        )
        self._idempotency_cache[request.idempotency_key] = result  # type: ignore[assignment]
        record.completed_at = completed_at
        record.success = True
        return result

    def _lookup_raw(self, request: ModelRequest) -> Any:
        if request.request_id in self._by_request_id:
            return self._by_request_id[request.request_id]
        factory = self._role_factories.get(request.role)
        if factory is not None:
            return factory(request)
        if request.role in self._by_role:
            return self._by_role[request.role]
        return request.context_payload.get("canned_output")

    @staticmethod
    def _parse_output(raw: Any, output_type: type[T]) -> T:
        if isinstance(raw, output_type):
            return raw
        if isinstance(raw, str):
            return output_type.model_validate_json(raw)
        if isinstance(raw, dict):
            return output_type.model_validate(raw)
        raise StructuredOutputFailure(
            f"Unsupported canned output type {type(raw).__name__} for {output_type.__name__}"
        )

    def replay_from_calls(self) -> dict[str, Any]:
        """Return a deterministic replay manifest from recorded calls."""
        return {
            "provider": self.provider_name,
            "call_count": len(self._calls),
            "calls": [
                {
                    "request_id": str(call.request.request_id),
                    "idempotency_key": call.request.idempotency_key,
                    "role": call.request.role,
                    "output_type": call.output_type_name,
                    "success": call.success,
                    "error_code": call.error_code,
                }
                for call in self._calls
            ],
        }

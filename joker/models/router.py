"""Local-first model routing with schema repair and escalation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from joker.models.exceptions import (
    ModelBudgetExceeded,
    ModelError,
    ModelProviderUnavailable,
    StructuredOutputFailure,
)
from joker.models.registry import ModelRegistry
from joker.models.schemas import ModelRequest, ModelResult, utc_now
from joker.models.telemetry import (
    build_model_call_completed,
    build_model_call_failed,
    build_model_call_started,
    build_routing_decision,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FAST_STRUCTURED_ROLES = frozenset(
    {
        "anomaly",
        "temporal_context",
    }
)

CRITIC_ROLES = frozenset(
    {
        "strategy_advocate",
        "falsifier",
        "historical_critic",
        "execution_critic",
        "alternative_explanation",
    }
)

DEEP_LOCAL_ROLES = frozenset(
    {
        "meta_decision",
        "entry_tactician",
        "order_manager",
        "position_thesis",
        "position_decision",
    }
)

INVENTOR_ROLES = frozenset(
    {
        "bullish_inventor",
        "bearish_inventor",
        "neutral_advocate",
    }
)


class ModelRouter:
    """Route cognitive model calls with local-first policy and safe escalation."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        max_schema_repair_attempts: int | None = None,
        max_provider_escalations: int | None = None,
        max_parallel_model_calls: int | None = None,
        session_id: str | None = None,
        model_call_repo: Any | None = None,
    ) -> None:
        self._registry = registry
        config = registry.config
        self._max_schema_repair_attempts = (
            max_schema_repair_attempts
            if max_schema_repair_attempts is not None
            else config.max_schema_repair_attempts
        )
        self._max_provider_escalations = (
            max_provider_escalations
            if max_provider_escalations is not None
            else config.max_provider_escalations
        )
        limit = (
            max_parallel_model_calls
            if max_parallel_model_calls is not None
            else config.max_parallel_model_calls
        )
        self._semaphore = asyncio.Semaphore(limit)
        self._session_id = session_id
        self._routing_logs: list[dict[str, Any]] = []
        self._model_call_repo = model_call_repo

    def set_model_call_repo(self, repo: Any | None) -> None:
        """Attach or replace the durable model-call repository."""
        self._model_call_repo = repo

    @property
    def routing_logs(self) -> list[dict[str, Any]]:
        """Return recorded routing decisions for tests and audit."""
        return list(self._routing_logs)

    def select_profile(
        self,
        request: ModelRequest,
        *,
        force_profile: str | None = None,
        escalate: bool = False,
    ) -> tuple[str, str, bool]:
        """Select a profile and return ``(profile_name, reason_code, independence_degraded)``."""
        if force_profile:
            return force_profile, "forced_profile", False

        if escalate or request.context_payload.get("escalate"):
            return "remote_escalation", "explicit_escalation", False

        if request.model_profile in self._registry.profiles:
            return request.model_profile, "request_profile_override", False

        role = request.role
        if role in CRITIC_ROLES:
            independence_degraded = False
            inventor_profile = request.context_payload.get("inventor_profile", "general_reasoning")
            if inventor_profile == "independent_critic":
                independence_degraded = True
            return "independent_critic", "critic_independence", independence_degraded

        if role in DEEP_LOCAL_ROLES:
            deep_profile = self._registry.get_profile("deep_local")
            if deep_profile.enabled:
                return "deep_local", "deep_local_arbitration", False
            return "general_reasoning", "deep_local_disabled", False

        if role in FAST_STRUCTURED_ROLES:
            return "fast_structured", "fast_extraction", False

        if role in INVENTOR_ROLES or role.startswith("pattern_") or role.endswith("_analyst"):
            return "general_reasoning", "general_reasoning", False

        return "general_reasoning", "default_general_reasoning", False

    async def route_and_complete(
        self,
        request: ModelRequest,
        output_type: type[T],
        *,
        force_profile: str | None = None,
        escalate: bool = False,
    ) -> ModelResult[T]:
        """Route a request, apply one schema repair retry, and escalate on failure."""
        profile_name, reason_code, independence_degraded = self.select_profile(
            request,
            force_profile=force_profile,
            escalate=escalate,
        )
        self._record_routing(
            request,
            profile_name,
            reason_code,
            independence_degraded=independence_degraded,
        )

        try:
            async with self._semaphore:
                return await self._complete_with_retries(
                    request,
                    output_type,
                    profile_name=profile_name,
                    reason_code=reason_code,
                    escalation_count=0,
                )
        except ModelBudgetExceeded:
            raise
        except asyncio.TimeoutError as exc:
            raise ModelBudgetExceeded("model router semaphore acquisition timed out") from exc

    async def _complete_with_retries(
        self,
        request: ModelRequest,
        output_type: type[T],
        *,
        profile_name: str,
        reason_code: str,
        escalation_count: int,
        repair_attempts: int = 0,
        escalated_from: str | None = None,
    ) -> ModelResult[T]:
        provider, profile, model_name = self._registry.provider_for_profile(profile_name)
        attempt_count = repair_attempts + escalation_count + 1
        routed_request = self._build_routed_request(
            request,
            profile=profile,
            model_name=model_name,
            attempt_count=attempt_count,
            escalated_from=escalated_from,
            schema_repair=repair_attempts > 0,
        )

        started_at = utc_now()
        started_event = build_model_call_started(
            request=routed_request,
            provider_name=provider.provider_name,
            model_name=model_name,
            profile_name=profile_name,
            routing_reason=reason_code,
            started_at=started_at,
            session_id=self._session_id,
        )
        logger.info("model call started", extra=started_event)

        # Durable idempotency: reuse a completed validated call before invoking provider.
        reused = await self._try_reuse_completed(request, output_type)
        if reused is not None:
            return reused

        await self._record_call_started(
            routed_request,
            provider_name=provider.provider_name,
            model_name=model_name,
            started_at=started_at,
            escalated_from=escalated_from,
            attempt_count=attempt_count,
        )

        try:
            result = await provider.complete_structured(
                request=routed_request,
                output_type=output_type,
            )
        except StructuredOutputFailure as exc:
            if repair_attempts < self._max_schema_repair_attempts:
                logger.info(
                    "schema repair retry",
                    extra={
                        "request_id": str(request.request_id),
                        "profile": profile_name,
                        "attempt": repair_attempts + 1,
                    },
                )
                return await self._complete_with_retries(
                    request,
                    output_type,
                    profile_name=profile_name,
                    reason_code="schema_repair_retry",
                    escalation_count=escalation_count,
                    repair_attempts=repair_attempts + 1,
                    escalated_from=escalated_from,
                )
            await self._record_call_failed(
                routed_request, error_code=type(exc).__name__
            )
            return await self._escalate_or_raise(
                request,
                output_type,
                profile_name=profile_name,
                reason_code=reason_code,
                escalation_count=escalation_count,
                error=exc,
            )
        except (ModelProviderUnavailable, ModelError) as exc:
            await self._record_call_failed(
                routed_request, error_code=type(exc).__name__
            )
            return await self._escalate_or_raise(
                request,
                output_type,
                profile_name=profile_name,
                reason_code=reason_code,
                escalation_count=escalation_count,
                error=exc,
            )

        completed = ModelResult[T](
            request_id=result.request_id,
            provider_name=result.provider_name,
            model_name=result.model_name,
            output=result.output,
            prompt_version=result.prompt_version,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=result.latency_ms,
            attempt_count=attempt_count,
            escalated_from=escalated_from,
            completed_at=result.completed_at,
        )
        await self._record_call_completed(routed_request, completed)
        logger.info(
            "model call completed",
            extra=build_model_call_completed(
                request=routed_request,
                result=completed,
                profile_name=profile_name,
                routing_reason=reason_code,
                session_id=self._session_id,
            ),
        )
        return completed

    async def _escalate_or_raise(
        self,
        request: ModelRequest,
        output_type: type[T],
        *,
        profile_name: str,
        reason_code: str,
        escalation_count: int,
        error: Exception,
    ) -> ModelResult[T]:
        if profile_name == "remote_escalation" or escalation_count >= self._max_provider_escalations:
            finished_at = utc_now()
            logger.warning(
                "model call failed without further escalation",
                extra=build_model_call_failed(
                    request=request,
                    provider_name=profile_name,
                    model_name=profile_name,
                    profile_name=profile_name,
                    routing_reason=reason_code,
                    error_code=type(error).__name__,
                    error_message=str(error),
                    attempt_count=escalation_count + 1,
                    started_at=finished_at,
                    finished_at=finished_at,
                    session_id=self._session_id,
                    escalated_from=profile_name,
                ),
            )
            raise error

        logger.info(
            "escalating model call",
            extra={
                "request_id": str(request.request_id),
                "from_profile": profile_name,
                "to_profile": "remote_escalation",
                "reason": type(error).__name__,
            },
        )
        self._record_routing(request, "remote_escalation", "escalation_after_failure")
        return await self._complete_with_retries(
            request,
            output_type,
            profile_name="remote_escalation",
            reason_code="escalation_after_failure",
            escalation_count=escalation_count + 1,
            repair_attempts=0,
            escalated_from=profile_name,
        )

    def _build_routed_request(
        self,
        request: ModelRequest,
        *,
        profile: Any,
        model_name: str,
        attempt_count: int,
        escalated_from: str | None,
        schema_repair: bool,
    ) -> ModelRequest:
        payload = dict(request.context_payload)
        payload["resolved_model"] = model_name
        payload["attempt_count"] = attempt_count
        if escalated_from:
            payload["escalated_from"] = escalated_from
        if schema_repair:
            payload["schema_repair"] = True
            payload["schema_repair_hint"] = (
                "Previous response failed schema validation. Return only valid JSON matching the schema."
            )
        max_tokens = min(request.max_output_tokens, profile.max_output_tokens)
        temperature = request.temperature if request.temperature is not None else profile.temperature
        return request.model_copy(
            update={
                "model_profile": profile.model,
                "max_output_tokens": max_tokens,
                "temperature": temperature,
                "context_payload": payload,
            }
        )

    def _record_routing(
        self,
        request: ModelRequest,
        profile_name: str,
        reason_code: str,
        *,
        independence_degraded: bool = False,
    ) -> None:
        entry = build_routing_decision(
            request_id=request.request_id,
            agent_role=request.role,
            selected_profile=profile_name,
            reason_code=reason_code,
            independence_degraded=independence_degraded,
        )
        self._routing_logs.append(entry)
        logger.info("model routing decision", extra=entry)

    async def _try_reuse_completed(
        self,
        request: ModelRequest,
        output_type: type[T],
    ) -> ModelResult[T] | None:
        if self._model_call_repo is None:
            return None
        try:
            existing = await self._model_call_repo.get_by_idempotency(request.idempotency_key)
        except Exception:
            return None
        if existing is None or existing.status.value != "completed":
            return None
        if not existing.validated_output_json:
            return None
        try:
            output = output_type.model_validate_json(existing.validated_output_json)
        except Exception:
            return None
        return ModelResult[T](
            request_id=existing.request_id,
            provider_name=existing.provider or "reused",
            model_name=existing.model or "reused",
            output=output,
            prompt_version=existing.prompt_version,
            input_tokens=existing.input_tokens,
            output_tokens=existing.output_tokens,
            latency_ms=existing.latency_ms or 0,
            attempt_count=existing.attempt_count,
            escalated_from=existing.escalation_source,
            completed_at=existing.finished_at or utc_now(),
        )

    async def _record_call_started(
        self,
        request: ModelRequest,
        *,
        provider_name: str,
        model_name: str,
        started_at,
        escalated_from: str | None,
        attempt_count: int,
    ) -> None:
        if self._model_call_repo is None:
            return
        from joker.cognition.artifacts import ModelCallRecord
        from joker.cognition.schemas import AgentRole, ModelCallStatus

        try:
            role = AgentRole(request.role)
        except ValueError:
            return
        try:
            await self._model_call_repo.append(
                ModelCallRecord(
                    request_id=request.request_id,
                    idempotency_key=request.idempotency_key,
                    session_id=self._session_id or "unknown",
                    cycle_id=request.cycle_id,
                    snapshot_id=request.snapshot_id,
                    agent_role=role,
                    prompt_id=request.prompt_id,
                    prompt_version=request.prompt_version,
                    provider=provider_name,
                    model=model_name,
                    status=ModelCallStatus.IN_PROGRESS,
                    attempt_count=attempt_count,
                    escalation_source=escalated_from,
                    started_at=started_at,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_call_start_persist_failed", extra={"error": str(exc)})

    async def _record_call_completed(
        self,
        request: ModelRequest,
        result: ModelResult[Any],
    ) -> None:
        if self._model_call_repo is None:
            return
        try:
            await self._model_call_repo.mark_complete(
                result.request_id,
                provider=result.provider_name,
                model=result.model_name,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                validated_output_json=result.output.model_dump_json(),
                finished_at=result.completed_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_call_complete_persist_failed", extra={"error": str(exc)})

    async def _record_call_failed(self, request: ModelRequest, *, error_code: str) -> None:
        if self._model_call_repo is None:
            return
        try:
            await self._model_call_repo.mark_failed(
                request.request_id,
                error_code=error_code,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("model_call_fail_persist_failed", extra={"error": str(exc)})

"""Helpers for structured model-call telemetry."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from joker.models.schemas import ModelRequest, ModelResult


def build_model_call_started(
    *,
    request: ModelRequest,
    provider_name: str,
    model_name: str,
    profile_name: str,
    routing_reason: str,
    started_at: datetime,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build a telemetry dict for a model call start event."""
    return {
        "event_type": "model_call.started",
        "request_id": str(request.request_id),
        "idempotency_key": request.idempotency_key,
        "session_id": session_id,
        "cycle_id": request.cycle_id,
        "snapshot_id": str(request.snapshot_id),
        "agent_role": request.role,
        "prompt_id": request.prompt_id,
        "prompt_version": request.prompt_version,
        "provider": provider_name,
        "model": model_name,
        "profile": profile_name,
        "routing_reason": routing_reason,
        "started_at": started_at.isoformat(),
    }


def build_model_call_completed(
    *,
    request: ModelRequest,
    result: ModelResult[Any],
    profile_name: str,
    routing_reason: str,
    session_id: str | None = None,
    artefact_id: str | None = None,
) -> dict[str, Any]:
    """Build a telemetry dict for a successful model call."""
    return {
        "event_type": "model_call.completed",
        "request_id": str(request.request_id),
        "idempotency_key": request.idempotency_key,
        "session_id": session_id,
        "cycle_id": request.cycle_id,
        "snapshot_id": str(request.snapshot_id),
        "agent_role": request.role,
        "prompt_id": request.prompt_id,
        "prompt_version": request.prompt_version,
        "provider": result.provider_name,
        "model": result.model_name,
        "profile": profile_name,
        "routing_reason": routing_reason,
        "attempt_count": result.attempt_count,
        "escalated_from": result.escalated_from,
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "completed_at": result.completed_at.isoformat(),
        "artefact_id": artefact_id,
        "status": "completed",
    }


def build_model_call_failed(
    *,
    request: ModelRequest,
    provider_name: str,
    model_name: str,
    profile_name: str,
    routing_reason: str,
    error_code: str,
    error_message: str,
    attempt_count: int,
    started_at: datetime,
    finished_at: datetime,
    session_id: str | None = None,
    escalated_from: str | None = None,
) -> dict[str, Any]:
    """Build a telemetry dict for a failed model call."""
    latency_ms = int((finished_at - started_at).total_seconds() * 1000)
    return {
        "event_type": "model_call.failed",
        "request_id": str(request.request_id),
        "idempotency_key": request.idempotency_key,
        "session_id": session_id,
        "cycle_id": request.cycle_id,
        "snapshot_id": str(request.snapshot_id),
        "agent_role": request.role,
        "prompt_id": request.prompt_id,
        "prompt_version": request.prompt_version,
        "provider": provider_name,
        "model": model_name,
        "profile": profile_name,
        "routing_reason": routing_reason,
        "attempt_count": attempt_count,
        "escalated_from": escalated_from,
        "latency_ms": latency_ms,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "error_code": error_code,
        "error_message": error_message,
        "status": "failed",
    }


def build_routing_decision(
    *,
    request_id: UUID,
    agent_role: str,
    selected_profile: str,
    reason_code: str,
    detail: str | None = None,
    independence_degraded: bool = False,
) -> dict[str, Any]:
    """Build a non-trading routing decision log entry."""
    return {
        "event_type": "model.routing_decision",
        "request_id": str(request_id),
        "agent_role": agent_role,
        "selected_profile": selected_profile,
        "reason_code": reason_code,
        "detail": detail,
        "independence_degraded": independence_degraded,
    }

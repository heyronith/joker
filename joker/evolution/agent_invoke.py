"""Shared ModelRouter invocation helpers for Task 3 agent graphs."""

from __future__ import annotations

from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel

from joker.evolution.hashing import content_hash, stable_json_dumps
from joker.models.router import ModelRouter
from joker.models.schemas import ModelRequest

T = TypeVar("T", bound=BaseModel)


async def invoke_evolution_agent(
    router: ModelRouter,
    *,
    role: str,
    prompt_id: str,
    prompt_version: str,
    payload: dict[str, Any],
    output_type: type[T],
    snapshot_id: UUID,
    cycle_id: str,
    session_id: str,
    attempt_level: int = 0,
) -> tuple[T, UUID]:
    """Route a Task 3 structured call through ModelRouter with provenance."""
    context_hash = content_hash(stable_json_dumps(payload))
    idempotency_key = content_hash(
        session_id,
        cycle_id,
        str(snapshot_id),
        role,
        prompt_version,
        context_hash,
        str(attempt_level),
    )
    request = ModelRequest(
        request_id=uuid4(),
        idempotency_key=idempotency_key,
        role=role,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        model_profile="",
        context_payload=payload,
        timeout_seconds=45.0,
        max_output_tokens=1200,
        snapshot_id=snapshot_id,
        cycle_id=cycle_id,
    )
    result = await router.route_and_complete(request, output_type)
    return result.output, result.request_id

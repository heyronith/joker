"""Agent runner with optional bounded data-request round-trip."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentDataRequest, AgentEvidence
from joker.cognition.tools import CognitiveReadTools
from joker.models.router import ModelRouter

T = TypeVar("T", bound=BaseModel)


def _extract_data_request(artefact: BaseModel) -> AgentDataRequest | None:
    if isinstance(artefact, AgentEvidence):
        if artefact.requires_more_data and artefact.data_request is not None:
            return artefact.data_request
    data_request = getattr(artefact, "data_request", None)
    if isinstance(data_request, AgentDataRequest):
        return data_request
    return None


async def run_agent_with_optional_data_request(
    agent: CognitiveAgent[T],
    context: ContextPackage,
    router: ModelRouter,
    tools: CognitiveReadTools,
    *,
    max_data_rounds: int = 1,
) -> T:
    """Run an agent; optionally fulfil one bounded data request then re-run.

    Only ``max_data_rounds`` (default 1) supplementary data fetches are allowed per
    invocation to prevent unbounded tool loops.
    """
    extra_payload: dict[str, Any] | None = None
    artefact: T | None = None

    for round_index in range(max_data_rounds + 1):
        artefact = await agent.run(
            context,
            router,
            attempt_level=round_index,
            extra_payload=extra_payload,
        )
        data_request = _extract_data_request(artefact)
        if data_request is None or round_index >= max_data_rounds:
            break

        fulfilled = await tools.fulfill_data_request(
            data_request,
            snapshot_id=context.snapshot_id,
        )
        prior = extra_payload.get("fulfilled_data_requests", []) if extra_payload else []
        extra_payload = {
            "fulfilled_data_requests": [
                *prior,
                {
                    "round": round_index,
                    "request": data_request.model_dump(mode="json"),
                    "response": fulfilled,
                },
            ],
            "data_request_satisfied": True,
        }

    assert artefact is not None
    return artefact

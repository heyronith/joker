"""Position thesis and decision agents."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, PositionThesisVersion
from joker.models.router import ModelRouter


class PositionThesisAgent(CognitiveAgent[PositionThesisVersion]):
    role = AgentRole.POSITION_THESIS
    output_type = PositionThesisVersion

    async def reassess(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        position_id: str,
        contract_id: str,
        original_strategy_id: UUID,
        position_projection: dict[str, Any] | None = None,
        prior_version: PositionThesisVersion | None = None,
        evidence_ids: Sequence[UUID] = (),
    ) -> PositionThesisVersion:
        extra: dict[str, Any] = {
            "position_id": position_id,
            "contract_id": contract_id,
            "original_strategy_id": str(original_strategy_id),
            "evidence_ids": [str(eid) for eid in evidence_ids],
        }
        if position_projection is not None:
            extra["position_projection"] = position_projection
        if prior_version is not None:
            extra["prior_thesis_version"] = prior_version.model_dump(mode="json")

        thesis = await self.run(context, router, extra_payload=extra)
        updates: dict[str, Any] = {
            "position_id": position_id,
            "contract_id": contract_id,
            "session_id": context.session_id,
            "snapshot_id": context.snapshot_id,
            "original_strategy_id": original_strategy_id,
        }
        if prior_version is not None:
            updates["prior_version_id"] = prior_version.thesis_version_id
        return thesis.model_copy(update=updates)


class PositionDecisionAgent(CognitiveAgent[PositionThesisVersion]):
    role = AgentRole.POSITION_DECISION
    output_type = PositionThesisVersion

    async def decide(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        latest_thesis: PositionThesisVersion,
        position_projection: dict[str, Any] | None = None,
        evidence_ids: Sequence[UUID] = (),
    ) -> PositionThesisVersion:
        extra: dict[str, Any] = {
            "latest_thesis_version": latest_thesis.model_dump(mode="json"),
            "evidence_ids": [str(eid) for eid in evidence_ids],
        }
        if position_projection is not None:
            extra["position_projection"] = position_projection

        decision = await self.run(context, router, extra_payload=extra)
        return decision.model_copy(
            update={
                "position_id": latest_thesis.position_id,
                "contract_id": latest_thesis.contract_id,
                "session_id": context.session_id,
                "snapshot_id": context.snapshot_id,
                "original_strategy_id": latest_thesis.original_strategy_id,
                "prior_version_id": latest_thesis.thesis_version_id,
            }
        )


async def run_position_cycle(
    *,
    state,
    router: ModelRouter,
    context: ContextPackage,
    position_id: str,
    contract_id: str,
    original_strategy_id: UUID,
) -> PositionThesisVersion:
    """Graph-facing position cycle wrapper."""
    evidence_ids = tuple(e.evidence_id for e in (state.get("evidence") or []))
    agent = PositionThesisAgent()
    strategy_uuid = (
        original_strategy_id
        if isinstance(original_strategy_id, UUID)
        else UUID(str(original_strategy_id))
    )
    return await agent.reassess(
        context,
        router,
        position_id=position_id,
        contract_id=contract_id,
        original_strategy_id=strategy_uuid,
        evidence_ids=evidence_ids,
    )

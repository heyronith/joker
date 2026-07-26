"""World-model synthesiser agent — model-built MarketWorldModel."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from joker.agents.cognitive.base import CognitiveAgent, require_fields
from joker.cognition.context import ContextPackage
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.schemas import AgentEvidence, AgentRole, MarketWorldModel
from joker.models.router import ModelRouter


class WorldModelSynthesiserAgent(CognitiveAgent[MarketWorldModel]):
    """Synthesise a typed MarketWorldModel from perception evidence via the model layer.

    Deterministic code may validate the result but must not construct the market
    interpretation (direction, confidence, summaries).
    """

    role = AgentRole.WORLD_MODEL_SYNTHESISER
    output_type = MarketWorldModel

    async def synthesise(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        evidence: Sequence[AgentEvidence],
    ) -> MarketWorldModel:
        if len(evidence) < 1:
            raise CognitiveValidationError(
                "world-model synthesis requires perception evidence artefacts"
            )
        evidence_ids = tuple(e.evidence_id for e in evidence)
        world_model = await self.run(
            context,
            router,
            extra_payload={
                "perception_evidence": [e.model_dump(mode="json") for e in evidence],
                "evidence_ids": [str(eid) for eid in evidence_ids],
                "data_quality": (
                    context.data_quality.model_dump(mode="json")
                    if context.data_quality is not None
                    else None
                ),
                "snapshot_metadata": {
                    "snapshot_id": str(context.snapshot_id),
                    "session_id": context.session_id,
                    "cycle_id": context.cycle_id,
                    "assembled_at": context.assembled_at.isoformat(),
                },
            },
        )
        validate_world_model(world_model, evidence_ids=evidence_ids)
        return world_model.model_copy(
            update={
                "session_id": context.session_id,
                "snapshot_id": context.snapshot_id,
                "cycle_id": context.cycle_id,
                "evidence_ids": evidence_ids,
                "synthesizer_model_call_id": world_model.model_call_id,
            }
        )


def validate_world_model(
    world_model: MarketWorldModel,
    *,
    evidence_ids: Sequence[UUID],
) -> None:
    """Deterministic validation only — never invents market interpretation."""
    require_fields(world_model, "prompt_version")
    if not world_model.regime_hypotheses:
        raise CognitiveValidationError("world model must include at least one regime hypothesis")
    if not world_model.evidence_ids and not evidence_ids:
        raise CognitiveValidationError("world model must reference contributing evidence IDs")
    known = {str(eid) for eid in evidence_ids}
    for eid in world_model.evidence_ids:
        if str(eid) not in known and known:
            # Allow synthesizer to cite subset; reject unknown IDs.
            raise CognitiveValidationError(
                f"world model references unknown evidence_id={eid}"
            )
    # Preserve disagreement: if multiple directions appear in regime hypotheses,
    # unresolved_questions or evidence_conflicts should acknowledge them.
    directions = {h.direction for h in world_model.regime_hypotheses}
    if len(directions) > 1 and not (
        world_model.evidence_conflicts or world_model.unresolved_questions
    ):
        raise CognitiveValidationError(
            "world model with disagreeing regime hypotheses must record conflicts "
            "or unresolved questions"
        )


async def run_world_model_synthesis(
    *,
    router: ModelRouter,
    context: ContextPackage,
    evidence: Sequence[AgentEvidence],
) -> MarketWorldModel:
    """Graph-facing world-model synthesis wrapper."""
    return await WorldModelSynthesiserAgent().synthesise(
        context,
        router,
        evidence=evidence,
    )

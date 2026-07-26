"""Meta-decision agent — model-routed, not majority vote."""

from __future__ import annotations

from typing import Sequence
from uuid import UUID

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.schemas import AgentRole, DebateReview, MetaDecision, StrategyHypothesis
from joker.models.router import ModelRouter


class MetaDecisionAgent(CognitiveAgent[MetaDecision]):
    """Synthesise a routing decision from evidence, strategies, and debate reviews."""

    role = AgentRole.META_DECISION
    output_type = MetaDecision

    async def decide(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        strategies: Sequence[StrategyHypothesis],
        reviews: Sequence[DebateReview],
        evidence_ids: Sequence[UUID] = (),
    ) -> MetaDecision:
        """Call the model with debate and strategy context; validate strategy refs."""
        strategy_summaries = [s.model_dump(mode="json") for s in strategies]
        review_summaries = [r.model_dump(mode="json") for r in reviews]
        decision = await self.run(
            context,
            router,
            extra_payload={
                "candidate_strategies": strategy_summaries,
                "debate_reviews": review_summaries,
                "evidence_ids": [str(eid) for eid in evidence_ids],
            },
        )
        validate_meta_decision(decision, strategies)
        return decision


def validate_meta_decision(
    decision: MetaDecision,
    strategies: Sequence[StrategyHypothesis],
) -> None:
    """Ensure selected_strategy_id references a known strategy when set."""
    if decision.selected_strategy_id is None:
        return

    known = {strategy.strategy_id for strategy in strategies}
    if decision.selected_strategy_id not in known:
        raise CognitiveValidationError(
            f"selected_strategy_id={decision.selected_strategy_id} not in candidate strategies"
        )

    for alt_id in decision.alternate_strategy_ids:
        if alt_id not in known:
            raise CognitiveValidationError(
                f"alternate_strategy_id={alt_id} not in candidate strategies"
            )


async def run_meta_decision(
    *,
    state,
    router: ModelRouter,
    context: ContextPackage,
    strategies,
):
    """Graph-facing meta-decision wrapper."""
    reviews = state.get("reviews") or []
    evidence_ids = tuple(e.evidence_id for e in state.get("evidence") or [])
    agent = MetaDecisionAgent()
    return await agent.decide(
        context,
        router,
        strategies=strategies,
        reviews=reviews,
        evidence_ids=evidence_ids,
    )

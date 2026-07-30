"""Meta-decision agent — model-routed, not majority vote."""

from __future__ import annotations

from typing import Any, Sequence
from uuid import UUID

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.schemas import (
    AgentRole,
    DebateReview,
    MetaDecision,
    MetaDecisionAction,
    StrategyHypothesis,
)
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
        objective_scores: Sequence[dict[str, Any]] | None = None,
    ) -> MetaDecision:
        """Call the model with debate, strategy, and objective score context."""
        strategy_summaries = [s.model_dump(mode="json") for s in strategies]
        review_summaries = [r.model_dump(mode="json") for r in reviews]
        score_payload = list(objective_scores or [])
        decision = await self.run(
            context,
            router,
            extra_payload={
                "candidate_strategies": strategy_summaries,
                "debate_reviews": review_summaries,
                "evidence_ids": [str(eid) for eid in evidence_ids],
                "objective_strategy_scores": score_payload,
            },
        )
        validate_meta_decision(decision, strategies, objective_scores=score_payload)
        return decision


def validate_meta_decision(
    decision: MetaDecision,
    strategies: Sequence[StrategyHypothesis],
    *,
    objective_scores: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Ensure selected_strategy_id references a known *and valid* objective score."""
    if decision.selected_strategy_id is None:
        if decision.action in {MetaDecisionAction.EXECUTE, MetaDecisionAction.PROBE}:
            raise CognitiveValidationError(
                "EXECUTE/PROBE requires selected_strategy_id"
            )
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

    if not objective_scores:
        # Objective scoring was not run this cycle (legacy / non-objective path).
        return

    scores = list(objective_scores)
    by_id = {
        str(s.get("strategy_id")): s
        for s in scores
        if not s.get("is_no_trade") and s.get("strategy_id") is not None
    }
    no_trade = next((s for s in scores if s.get("is_no_trade")), None)
    selected = by_id.get(str(decision.selected_strategy_id))
    if selected is None:
        raise CognitiveValidationError(
            "selected strategy has no objective score for this snapshot"
        )
    if not selected.get("valid", False):
        raise CognitiveValidationError(
            "selected strategy objective score is invalid"
        )

    # Snapshot / objective membership
    for s in scores:
        if s.get("strategy_id") is not None and str(s.get("strategy_id")) == str(
            decision.selected_strategy_id
        ):
            if selected.get("objective_id") and s.get("objective_id") not in {
                None,
                selected.get("objective_id"),
            }:
                raise CognitiveValidationError(
                    "selected score belongs to a different objective"
                )

    if decision.action in {MetaDecisionAction.EXECUTE, MetaDecisionAction.PROBE}:
        if no_trade and no_trade.get("valid"):
            # Prefer no-trade when all trade scores invalid (caller should abandon).
            valid_trades = [s for s in scores if s.get("valid") and not s.get("is_no_trade")]
            if not valid_trades:
                raise CognitiveValidationError(
                    "no valid trade objective scores; must abandon/no-trade"
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
    objective_scores = state.get("_strategy_scores") or []
    agent = MetaDecisionAgent()
    return await agent.decide(
        context,
        router,
        strategies=strategies,
        reviews=reviews,
        evidence_ids=evidence_ids,
        objective_scores=objective_scores,
    )

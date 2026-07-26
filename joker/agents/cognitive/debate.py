"""Adversarial debate panel producing DebateReview artefacts."""

from __future__ import annotations

import asyncio
from typing import Any, Sequence
from uuid import UUID

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, DebateReview, StrategyHypothesis
from joker.models.router import ModelRouter

DEBATE_ROLES: tuple[AgentRole, ...] = (
    AgentRole.STRATEGY_ADVOCATE,
    AgentRole.FALSIFIER,
    AgentRole.HISTORICAL_CRITIC,
    AgentRole.EXECUTION_CRITIC,
    AgentRole.ALTERNATIVE_EXPLANATION,
)


class StrategyAdvocateAgent(CognitiveAgent[DebateReview]):
    role = AgentRole.STRATEGY_ADVOCATE
    output_type = DebateReview

    def enrich_output(
        self,
        output: DebateReview,
        *,
        context: ContextPackage,
        model_call_id: UUID,
    ) -> DebateReview:
        return output.model_copy(
            update={
                "snapshot_id": context.snapshot_id,
                "cycle_id": context.cycle_id,
                "prompt_version": self.prompt_version,
                "model_call_id": model_call_id,
                "reviewer_role": self.role,
            }
        )


class FalsifierAgent(CognitiveAgent[DebateReview]):
    role = AgentRole.FALSIFIER
    output_type = DebateReview

    def enrich_output(
        self,
        output: DebateReview,
        *,
        context: ContextPackage,
        model_call_id: UUID,
    ) -> DebateReview:
        return output.model_copy(
            update={
                "snapshot_id": context.snapshot_id,
                "cycle_id": context.cycle_id,
                "prompt_version": self.prompt_version,
                "model_call_id": model_call_id,
                "reviewer_role": self.role,
            }
        )


class HistoricalCriticAgent(CognitiveAgent[DebateReview]):
    role = AgentRole.HISTORICAL_CRITIC
    output_type = DebateReview

    def enrich_output(
        self,
        output: DebateReview,
        *,
        context: ContextPackage,
        model_call_id: UUID,
    ) -> DebateReview:
        return output.model_copy(
            update={
                "snapshot_id": context.snapshot_id,
                "cycle_id": context.cycle_id,
                "prompt_version": self.prompt_version,
                "model_call_id": model_call_id,
                "reviewer_role": self.role,
            }
        )


class ExecutionCriticAgent(CognitiveAgent[DebateReview]):
    role = AgentRole.EXECUTION_CRITIC
    output_type = DebateReview

    def enrich_output(
        self,
        output: DebateReview,
        *,
        context: ContextPackage,
        model_call_id: UUID,
    ) -> DebateReview:
        return output.model_copy(
            update={
                "snapshot_id": context.snapshot_id,
                "cycle_id": context.cycle_id,
                "prompt_version": self.prompt_version,
                "model_call_id": model_call_id,
                "reviewer_role": self.role,
            }
        )


class AlternativeExplanationAgent(CognitiveAgent[DebateReview]):
    role = AgentRole.ALTERNATIVE_EXPLANATION
    output_type = DebateReview

    def enrich_output(
        self,
        output: DebateReview,
        *,
        context: ContextPackage,
        model_call_id: UUID,
    ) -> DebateReview:
        return output.model_copy(
            update={
                "snapshot_id": context.snapshot_id,
                "cycle_id": context.cycle_id,
                "prompt_version": self.prompt_version,
                "model_call_id": model_call_id,
                "reviewer_role": self.role,
            }
        )


_DEBATE_AGENTS: dict[AgentRole, type[CognitiveAgent[DebateReview]]] = {
    AgentRole.STRATEGY_ADVOCATE: StrategyAdvocateAgent,
    AgentRole.FALSIFIER: FalsifierAgent,
    AgentRole.HISTORICAL_CRITIC: HistoricalCriticAgent,
    AgentRole.EXECUTION_CRITIC: ExecutionCriticAgent,
    AgentRole.ALTERNATIVE_EXPLANATION: AlternativeExplanationAgent,
}


def debate_agent_for(role: AgentRole) -> CognitiveAgent[DebateReview]:
    try:
        return _DEBATE_AGENTS[role]()
    except KeyError as exc:
        raise ValueError(f"not a debate role: {role!r}") from exc


def debate_context_for_strategy(
    context: ContextPackage,
    strategy: StrategyHypothesis,
) -> ContextPackage:
    """Attach the strategy under review to the debate context package."""
    summary: dict[str, Any] = {
        "artifact_id": strategy.strategy_id,
        "artifact_type": "strategy_hypothesis",
        "strategy": strategy.model_dump(mode="json"),
    }
    return context.model_copy(
        update={
            "session_artifact_summaries": context.session_artifact_summaries + (summary,),
        }
    )


async def run_debate_panel(
    strategy: StrategyHypothesis,
    context: ContextPackage,
    router: ModelRouter,
    *,
    roles: Sequence[AgentRole] | None = None,
) -> tuple[DebateReview, ...]:
    """Run all debate critics for a strategy hypothesis in parallel."""
    debate_context = debate_context_for_strategy(context, strategy)
    selected = tuple(roles) if roles is not None else DEBATE_ROLES
    agents = [debate_agent_for(role) for role in selected]
    extra = {
        "strategy_id": str(strategy.strategy_id),
        "strategy_name": strategy.name,
    }

    async def _run_one(agent: CognitiveAgent[DebateReview]) -> DebateReview:
        review = await agent.run(debate_context, router, extra_payload=extra)
        if review.strategy_id != strategy.strategy_id:
            review = review.model_copy(update={"strategy_id": strategy.strategy_id})
        return review

    return tuple(await asyncio.gather(*[_run_one(agent) for agent in agents]))

"""Perception swarm agents producing AgentEvidence."""

from __future__ import annotations

import asyncio
from typing import Sequence

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentEvidence, AgentRole
from joker.models.router import ModelRouter

PERCEPTION_ROLES: tuple[AgentRole, ...] = (
    AgentRole.MARKET_STRUCTURE,
    AgentRole.VOLATILITY,
    AgentRole.OPTIONS_MICROSTRUCTURE,
    AgentRole.TEMPORAL_CONTEXT,
    AgentRole.ANOMALY,
)


class MarketStructureAgent(CognitiveAgent[AgentEvidence]):
    role = AgentRole.MARKET_STRUCTURE
    output_type = AgentEvidence


class VolatilityAgent(CognitiveAgent[AgentEvidence]):
    role = AgentRole.VOLATILITY
    output_type = AgentEvidence


class OptionsMicrostructureAgent(CognitiveAgent[AgentEvidence]):
    role = AgentRole.OPTIONS_MICROSTRUCTURE
    output_type = AgentEvidence


class TemporalContextAgent(CognitiveAgent[AgentEvidence]):
    role = AgentRole.TEMPORAL_CONTEXT
    output_type = AgentEvidence


class AnomalyAgent(CognitiveAgent[AgentEvidence]):
    role = AgentRole.ANOMALY
    output_type = AgentEvidence


_PERCEPTION_AGENTS: dict[AgentRole, type[CognitiveAgent[AgentEvidence]]] = {
    AgentRole.MARKET_STRUCTURE: MarketStructureAgent,
    AgentRole.VOLATILITY: VolatilityAgent,
    AgentRole.OPTIONS_MICROSTRUCTURE: OptionsMicrostructureAgent,
    AgentRole.TEMPORAL_CONTEXT: TemporalContextAgent,
    AgentRole.ANOMALY: AnomalyAgent,
}


def perception_agent_for(role: AgentRole) -> CognitiveAgent[AgentEvidence]:
    """Return a perception agent instance for the given role."""
    try:
        return _PERCEPTION_AGENTS[role]()
    except KeyError as exc:
        raise ValueError(f"not a perception role: {role!r}") from exc


async def run_perception_swarm(
    router: ModelRouter,
    context: ContextPackage,
    *,
    roles: Sequence[AgentRole] | None = None,
) -> tuple[AgentEvidence, ...]:
    """Run all perception agents in parallel and return evidence artefacts."""
    selected = tuple(roles) if roles is not None else PERCEPTION_ROLES
    agents = [perception_agent_for(role) for role in selected]

    async def _run_one(agent: CognitiveAgent[AgentEvidence]) -> AgentEvidence:
        return await agent.run(context, router)

    return tuple(await asyncio.gather(*[_run_one(agent) for agent in agents]))

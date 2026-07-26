"""Strategy inventor agents producing StrategyHypothesis."""

from __future__ import annotations

import asyncio
from typing import Sequence

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, StrategyHypothesis
from joker.models.router import ModelRouter

# Example novel strategy names agents may invent (not in legacy playbooks).
NOVEL_STRATEGY_NAME_EXAMPLES: tuple[str, ...] = (
    "failed_breakout_reclaim_call",
    "gamma_pin_fade_put",
    "opening_drive_exhaustion_call",
)

STRATEGY_INVENTOR_ROLES: tuple[AgentRole, ...] = (
    AgentRole.BULLISH_INVENTOR,
    AgentRole.BEARISH_INVENTOR,
    AgentRole.NEUTRAL_ADVOCATE,
)


class BullishInventorAgent(CognitiveAgent[StrategyHypothesis]):
    role = AgentRole.BULLISH_INVENTOR
    output_type = StrategyHypothesis


class BearishInventorAgent(CognitiveAgent[StrategyHypothesis]):
    role = AgentRole.BEARISH_INVENTOR
    output_type = StrategyHypothesis


class NeutralAdvocateAgent(CognitiveAgent[StrategyHypothesis]):
    role = AgentRole.NEUTRAL_ADVOCATE
    output_type = StrategyHypothesis


_STRATEGY_AGENTS: dict[AgentRole, type[CognitiveAgent[StrategyHypothesis]]] = {
    AgentRole.BULLISH_INVENTOR: BullishInventorAgent,
    AgentRole.BEARISH_INVENTOR: BearishInventorAgent,
    AgentRole.NEUTRAL_ADVOCATE: NeutralAdvocateAgent,
}


def strategy_agent_for(role: AgentRole) -> CognitiveAgent[StrategyHypothesis]:
    try:
        return _STRATEGY_AGENTS[role]()
    except KeyError as exc:
        raise ValueError(f"not a strategy inventor role: {role!r}") from exc


def is_novel_strategy_name(name: str, legacy_playbook_names: set[str]) -> bool:
    """Return True when a strategy name is absent from the legacy playbook."""
    return name.strip().lower() not in {n.strip().lower() for n in legacy_playbook_names}


async def run_strategy_inventors(
    router: ModelRouter,
    context: ContextPackage,
    *,
    roles: Sequence[AgentRole] | None = None,
    legacy_playbook_names: set[str] | None = None,
) -> tuple[StrategyHypothesis, ...]:
    """Run strategy inventor agents in parallel."""
    selected = tuple(roles) if roles is not None else STRATEGY_INVENTOR_ROLES
    agents = [strategy_agent_for(role) for role in selected]

    extra: dict[str, object] = {
        "novel_strategy_name_examples": list(NOVEL_STRATEGY_NAME_EXAMPLES),
    }
    if legacy_playbook_names is not None:
        extra["legacy_playbook_names"] = sorted(legacy_playbook_names)

    async def _run_one(agent: CognitiveAgent[StrategyHypothesis]) -> StrategyHypothesis:
        return await agent.run(context, router, extra_payload=extra)

    return tuple(await asyncio.gather(*[_run_one(agent) for agent in agents]))

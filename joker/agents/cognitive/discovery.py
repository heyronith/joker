"""Discovery agents producing PatternHypothesis with bounded coordination."""

from __future__ import annotations

import asyncio
from typing import Sequence

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, PatternHypothesis
from joker.models.router import ModelRouter

DISCOVERY_ROLES: tuple[AgentRole, ...] = (
    AgentRole.PATTERN_MINER,
    AgentRole.SEQUENCE_ANALYST,
    AgentRole.ANALOGY_RETRIEVER,
)


class PatternMinerAgent(CognitiveAgent[PatternHypothesis]):
    role = AgentRole.PATTERN_MINER
    output_type = PatternHypothesis


class SequenceAnalystAgent(CognitiveAgent[PatternHypothesis]):
    role = AgentRole.SEQUENCE_ANALYST
    output_type = PatternHypothesis


class AnalogyRetrieverAgent(CognitiveAgent[PatternHypothesis]):
    role = AgentRole.ANALOGY_RETRIEVER
    output_type = PatternHypothesis


_DISCOVERY_AGENTS: dict[AgentRole, type[CognitiveAgent[PatternHypothesis]]] = {
    AgentRole.PATTERN_MINER: PatternMinerAgent,
    AgentRole.SEQUENCE_ANALYST: SequenceAnalystAgent,
    AgentRole.ANALOGY_RETRIEVER: AnalogyRetrieverAgent,
}


def discovery_agent_for(role: AgentRole) -> CognitiveAgent[PatternHypothesis]:
    try:
        return _DISCOVERY_AGENTS[role]()
    except KeyError as exc:
        raise ValueError(f"not a discovery role: {role!r}") from exc


def select_pattern_hypotheses(
    hypotheses: Sequence[PatternHypothesis],
    *,
    max_hypotheses: int = 5,
) -> tuple[PatternHypothesis, ...]:
    """Select a bounded, de-duplicated set ranked by novelty and confidence."""
    if max_hypotheses <= 0:
        return ()

    seen_names: set[str] = set()
    ranked: list[PatternHypothesis] = []

    for hypothesis in sorted(
        hypotheses,
        key=lambda h: (h.novelty_score * h.confidence, h.confidence, h.novelty_score),
        reverse=True,
    ):
        key = hypothesis.name.strip().lower()
        if key in seen_names:
            continue
        seen_names.add(key)
        ranked.append(hypothesis)
        if len(ranked) >= max_hypotheses:
            break

    return tuple(ranked)


async def run_discovery_swarm(
    router: ModelRouter,
    context: ContextPackage,
    *,
    max_hypotheses: int = 5,
    roles: Sequence[AgentRole] | None = None,
) -> tuple[PatternHypothesis, ...]:
    """Run discovery agents in parallel and return a coordinator-selected subset."""
    selected = tuple(roles) if roles is not None else DISCOVERY_ROLES
    agents = [discovery_agent_for(role) for role in selected]

    async def _run_one(agent: CognitiveAgent[PatternHypothesis]) -> PatternHypothesis:
        return await agent.run(context, router)

    raw = await asyncio.gather(*[_run_one(agent) for agent in agents])
    return select_pattern_hypotheses(raw, max_hypotheses=max_hypotheses)

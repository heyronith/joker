"""Council disagreement and confidence analysis."""

from __future__ import annotations

from joker.schemas.domain import AgentCouncilDecision, AgentOpinion, MarketRegime, SCHEMA_VERSION, VersionedModel

LOW_CONFIDENCE_THRESHOLD = 0.5
CRITIC_BLOCK_THRESHOLD = 0.4


class CouncilAnalysis(VersionedModel):
    agent_confidence: dict[str, float] = {}
    low_confidence_agents: list[str] = []
    conflicting_regimes: bool = False
    regime_values: list[str] = []
    critic_warnings: list[str] = []
    synthesizer_rationale: str = ""
    council_blocked: bool = False
    critic_confidence: float | None = None


def analyze_council(decision: AgentCouncilDecision) -> CouncilAnalysis:
    """Extract disagreement metadata from council decision."""
    agent_confidence: dict[str, float] = {}
    low_confidence: list[str] = []
    regimes: set[str] = set()
    critic_warnings: list[str] = []
    critic_confidence: float | None = None
    council_blocked = False

    for opinion in decision.opinions:
        agent_confidence[opinion.agent_name] = opinion.confidence
        if opinion.confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence.append(opinion.agent_name)
        if opinion.regime is not None:
            regimes.add(opinion.regime.value)
        if opinion.agent_name == "CriticAgent":
            critic_confidence = opinion.confidence
            if "low confidence" in opinion.summary.lower() or "weak" in opinion.summary.lower():
                critic_warnings.append(opinion.summary)
            if opinion.confidence < CRITIC_BLOCK_THRESHOLD:
                council_blocked = True
                critic_warnings.append(
                    f"Critic confidence {opinion.confidence:.2f} below block threshold"
                )

    conflicting = len(regimes) > 1

    return CouncilAnalysis(
        agent_confidence=agent_confidence,
        low_confidence_agents=low_confidence,
        conflicting_regimes=conflicting,
        regime_values=sorted(regimes),
        critic_warnings=critic_warnings,
        synthesizer_rationale=decision.synthesis_summary,
        council_blocked=council_blocked,
        critic_confidence=critic_confidence,
    )

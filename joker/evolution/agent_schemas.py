"""Typed evaluator / improvement / evolution-decision agent outputs (no CoT)."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.schemas import assert_no_chain_of_thought


class _NoCot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def model_post_init(self, __context: object) -> None:
        assert_no_chain_of_thought(self.model_dump(mode="json"))


class EvaluatorAgentScores(_NoCot):
    """Structured scores from an evaluator agent role."""

    thesis_quality: Decimal | None = None
    evidence_grounding_score: Decimal | None = None
    calibration_score: Decimal | None = None
    debate_quality: Decimal | None = None
    execution_quality: Decimal | None = None
    position_management_score: Decimal | None = None
    efficiency_score: Decimal | None = None
    decision_consistency_score: Decimal | None = None
    avoidable_error_codes: tuple[str, ...] = ()
    finding_codes: tuple[str, ...] = ()


class ImprovementAgentProposal(_NoCot):
    """Agent-owned weakness detection and permitted cognitive patch."""

    weakness: str
    hypothesis: str
    patch_type: Literal[
        "prompt",
        "context_policy",
        "routing_policy",
        "debate_policy",
        "memory_policy",
        "escalation_policy",
    ]
    role: str = "meta_decision"
    replacement_template: str | None = None
    preferred_profile: str | None = None
    change_rationale: str = "improve declared weakness"
    metrics_to_improve: tuple[str, ...] = ("calibration_score",)
    metrics_must_not_regress: tuple[str, ...] = ("tail_loss", "safety_violations")
    critic_accepted: bool = True
    critic_rejection_codes: tuple[str, ...] = ()


class EvolutionDecisionAgentOutput(_NoCot):
    """Strategic promotion judgement — still bound by deterministic gates."""

    action: Literal["promote", "reject", "extend_shadow", "request_more_evidence"]
    rationale_codes: tuple[str, ...] = ()
    summary: str = Field(min_length=1)

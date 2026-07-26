"""Evaluation-facing schemas and evidence packages (no chain-of-thought)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.schemas import (
    DecisionTraceSummary,
    EpisodeEvaluation,
    TradingEpisode,
    assert_no_chain_of_thought,
)

__all__ = [
    "DecisionTraceSummary",
    "DeterministicOutcomeMetrics",
    "EpisodeEvaluation",
    "EvidencePackage",
    "TradingEpisode",
]


class DeterministicOutcomeMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    realised_pnl: Decimal | None = None
    return_on_premium: Decimal | None = None
    max_favourable_excursion: Decimal | None = None
    max_adverse_excursion: Decimal | None = None
    profit_capture_ratio: Decimal | None = None
    exit_efficiency: Decimal | None = None
    entry_slippage: Decimal | None = None
    exit_slippage: Decimal | None = None
    fill_ratio: Decimal | None = None
    holding_seconds: int | None = None
    quote_age_at_submission_seconds: Decimal | None = None
    spread_at_submission: Decimal | None = None
    model_call_latency_ms_total: int = 0
    model_call_count: int = 0
    approximate_token_usage: int = 0
    provider_failures: int = 0
    escalation_frequency: int = 0
    evidence_request_count: int = 0
    debate_iterations: int = 0
    rejected_order_count: int = 0
    duplicate_action_count: int = 0
    calibration_error: Decimal | None = None
    brier_score: Decimal | None = None
    directional_consistency: Decimal | None = None
    thesis_invalidation_response_delay_seconds: int | None = None
    data_quality_exposure: int = 0
    safety_violations: int = 0
    integrity_violations: int = 0


class EvidencePackage(BaseModel):
    """Immutable evidence for evaluator agents — never includes hidden CoT."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode: TradingEpisode
    trace_summary: DecisionTraceSummary | None = None
    metrics: DeterministicOutcomeMetrics
    artifact_refs: tuple[UUID, ...] = ()
    snapshot_ids: tuple[UUID, ...] = ()
    extra_facts: dict[str, Any] = Field(default_factory=dict)
    assembled_at: datetime

    def model_post_init(self, __context: Any) -> None:
        assert_no_chain_of_thought(self.extra_facts)

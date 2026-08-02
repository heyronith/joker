"""Typed schemas for Task-3 historical outcome retrieval and leakage reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _money(value: Decimal | float | int | str) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"))


def _dec4(value: Decimal | float | int | str) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"))


class HistoricalOutcomeQuery(BaseModel):
    """As-of historical analogue query — must not encode future information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    strategy_id: UUID
    snapshot_id: UUID

    configuration_version_id: UUID | None = None
    pattern_ids: tuple[UUID, ...] = ()
    strategy_family: str | None = None
    direction: str | None = None

    underlying_symbol: str = "SPY"
    option_type: str | None = None
    expiration_class: str | None = None

    regime_labels: tuple[str, ...] = ()
    session_phase: str = "unknown"
    volatility_bucket: str | None = None
    liquidity_bucket: str | None = None
    premium_bucket: str | None = None
    horizon_bucket: str | None = None

    maximum_samples: int = 200
    minimum_similarity: Decimal = Decimal("0.65")
    as_of_timestamp: datetime
    current_episode_id: UUID | None = None
    allow_synthetic_replay: bool = False
    # Datasets that trained / contaminated the active configuration under evaluation.
    blocked_training_dataset_ids: tuple[UUID, ...] = ()
    challenger_dataset_ids: tuple[UUID, ...] = ()
    # When a configuration_version_id is supplied, provenance must be resolved
    # (even if the dataset ID tuples are empty). Missing provenance fails closed.
    configuration_dataset_provenance_resolved: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("minimum_similarity", mode="before")
    @classmethod
    def _sim(cls, value: object) -> Decimal:
        return _dec4(value)  # type: ignore[arg-type]

    @field_validator("as_of_timestamp", "created_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class ComparableOutcome(BaseModel):
    """One independent factual observation eligible for EV aggregation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: UUID
    evaluation_id: UUID
    dataset_id: UUID | None = None

    strategy_id: UUID | None = None
    strategy_version_id: UUID | None = None
    configuration_version_id: UUID

    entry_snapshot_id: UUID
    terminal_event_id: UUID
    entry_timestamp: datetime
    terminal_timestamp: datetime

    regime_labels: tuple[str, ...] = ()
    session_phase: str = "unknown"
    option_type: str | None = None
    entry_premium_usd: Decimal | None = None
    holding_seconds: int | None = None

    realized_pnl_usd: Decimal
    outcome_label: str
    confidence_at_decision: Decimal | None = None

    similarity_score: Decimal
    similarity_components: dict[str, Decimal] = Field(default_factory=dict)

    evidence_ids: tuple[UUID, ...] = ()
    complete: bool = True
    independence_key: str = ""
    truth_degraded: bool = False
    historical_ev_eligible: bool = True

    @field_validator(
        "realized_pnl_usd",
        "entry_premium_usd",
        "confidence_at_decision",
        mode="before",
    )
    @classmethod
    def _dec(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _money(value)  # type: ignore[arg-type]

    @field_validator("similarity_score", mode="before")
    @classmethod
    def _sim(cls, value: object) -> Decimal:
        return _dec4(value)  # type: ignore[arg-type]

    @field_validator("entry_timestamp", "terminal_timestamp")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class HistoricalOutcomeSummary(BaseModel):
    """Aggregated factual statistics for a historical query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary_id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    strategy_id: UUID
    snapshot_id: UUID

    sample_count: int
    profitable_count: int
    losing_count: int
    flat_count: int

    average_pnl_usd: Decimal | None = None
    median_pnl_usd: Decimal | None = None
    pnl_standard_deviation_usd: Decimal | None = None

    hit_rate: Decimal | None = None
    average_win_usd: Decimal | None = None
    average_loss_usd: Decimal | None = None
    payoff_ratio: Decimal | None = None

    lower_confidence_bound_ev_usd: Decimal | None = None
    effective_sample_size: Decimal | None = None

    minimum_similarity: Decimal
    average_similarity: Decimal | None = None

    comparable_episode_ids: tuple[UUID, ...] = ()
    evaluation_ids: tuple[UUID, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()

    exclusion_counts: dict[str, int] = Field(default_factory=dict)
    valid_for_ev: bool = False
    invalidation_reasons: tuple[str, ...] = ()
    similarity_policy_version: str = "1.0.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "average_pnl_usd",
        "median_pnl_usd",
        "pnl_standard_deviation_usd",
        "hit_rate",
        "average_win_usd",
        "average_loss_usd",
        "payoff_ratio",
        "lower_confidence_bound_ev_usd",
        "effective_sample_size",
        "minimum_similarity",
        "average_similarity",
        mode="before",
    )
    @classmethod
    def _dec(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))

    @field_validator("created_at")
    @classmethod
    def _tz(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class HistoricalLeakageReport(BaseModel):
    """Explicit as-of / independence leakage exclusions for a query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: UUID
    excluded_future_episodes: tuple[UUID, ...] = ()
    excluded_current_episode: tuple[UUID, ...] = ()
    excluded_dataset_overlap: tuple[UUID, ...] = ()
    excluded_duplicate_truth: tuple[UUID, ...] = ()
    excluded_incomplete: tuple[UUID, ...] = ()
    excluded_truth_degraded: tuple[UUID, ...] = ()
    safe: bool = True
    notes: tuple[str, ...] = ()


class RepricedStrategyEstimate(BaseModel):
    """Execution-time EV recomputed against the current option quote."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    original_estimate_id: UUID
    request_snapshot_id: UUID
    quote_timestamp: datetime

    original_premium_usd: Decimal
    current_premium_usd: Decimal
    premium_change_usd: Decimal
    premium_change_pct: Decimal | None = None

    original_expected_value_usd: Decimal
    repriced_expected_value_usd: Decimal | None = None
    repricing_method: str

    original_maximum_loss_usd: Decimal
    repriced_maximum_loss_usd: Decimal

    assumptions_still_valid: bool
    invalidation_reasons: tuple[str, ...] = ()
    valid: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "original_premium_usd",
        "current_premium_usd",
        "premium_change_usd",
        "premium_change_pct",
        "original_expected_value_usd",
        "repriced_expected_value_usd",
        "original_maximum_loss_usd",
        "repriced_maximum_loss_usd",
        mode="before",
    )
    @classmethod
    def _dec(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return Decimal(str(value))


class CalibrationOutcomeResolution(BaseModel):
    """Whether a declared confidence may form a calibration sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: UUID
    outcome: int | None = None
    included: bool
    exclusion_reason: str | None = None
    evidence_ids: tuple[UUID, ...] = ()


class ColdStartEvidenceProgress(BaseModel):
    """Operator-visible progress toward historical-EV eligibility."""

    model_config = ConfigDict(extra="forbid")

    current_sample_count: int
    minimum_required: int
    similarity_threshold: Decimal
    exclusion_counts: dict[str, int] = Field(default_factory=dict)
    estimated_progress_pct: Decimal = Decimal("0.00")
    status: str = "insufficient_historical_evidence"


class SimilarityPolicy(BaseModel):
    """Versioned, configuration-driven similarity weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = "1.0.0"
    strategy_family_weight: Decimal = Decimal("0.20")
    pattern_overlap_weight: Decimal = Decimal("0.20")
    regime_similarity_weight: Decimal = Decimal("0.20")
    session_phase_weight: Decimal = Decimal("0.10")
    volatility_similarity_weight: Decimal = Decimal("0.10")
    liquidity_similarity_weight: Decimal = Decimal("0.10")
    premium_similarity_weight: Decimal = Decimal("0.05")
    horizon_similarity_weight: Decimal = Decimal("0.05")

    def weight_map(self) -> dict[str, Decimal]:
        return {
            "strategy_family_match": self.strategy_family_weight,
            "pattern_overlap": self.pattern_overlap_weight,
            "regime_similarity": self.regime_similarity_weight,
            "session_phase_match": self.session_phase_weight,
            "volatility_similarity": self.volatility_similarity_weight,
            "liquidity_similarity": self.liquidity_similarity_weight,
            "premium_similarity": self.premium_similarity_weight,
            "horizon_similarity": self.horizon_similarity_weight,
        }

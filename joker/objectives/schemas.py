"""Durable session-objective domain models (Task 1 owned)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _money(value: Decimal | float | int | str) -> Decimal:
    d = Decimal(str(value))
    return d.quantize(Decimal("0.01"))


class SessionObjectiveDefinition(BaseModel):
    """Immutable session objective after confirmation (new version on change)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_id: UUID = Field(default_factory=uuid4)
    session_id: str
    authorised_capital_usd: Decimal
    target_profit_pct: Decimal
    target_profit_usd: Decimal
    target_ending_equity_usd: Decimal
    deadline_exchange_time: datetime
    max_concurrent_positions: int
    pause_entries_when_goal_met: bool = True
    accepted_total_loss_risk: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    definition_version: int = 1
    armed: bool = False
    first_broker_submission_at: datetime | None = None

    @field_validator(
        "authorised_capital_usd",
        "target_profit_pct",
        "target_profit_usd",
        "target_ending_equity_usd",
        mode="before",
    )
    @classmethod
    def _decimal_money(cls, value: object) -> Decimal:
        return _money(value)  # type: ignore[arg-type]

    @field_validator("deadline_exchange_time", "created_at", "first_broker_submission_at")
    @classmethod
    def _tz_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_derived(self) -> SessionObjectiveDefinition:
        if self.authorised_capital_usd <= 0:
            raise ValueError("authorised_capital_usd must be > 0")
        if self.target_profit_pct < 0:
            raise ValueError("target_profit_pct must be >= 0")
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")
        if not self.accepted_total_loss_risk:
            raise ValueError("accepted_total_loss_risk must be true to arm an objective")
        expected_profit = (
            self.authorised_capital_usd * self.target_profit_pct / Decimal("100")
        ).quantize(Decimal("0.01"))
        if abs(self.target_profit_usd - expected_profit) > Decimal("0.01"):
            raise ValueError("target_profit_usd must equal authorised * pct / 100")
        expected_end = (self.authorised_capital_usd + self.target_profit_usd).quantize(
            Decimal("0.01")
        )
        if abs(self.target_ending_equity_usd - expected_end) > Decimal("0.01"):
            raise ValueError(
                "target_ending_equity_usd must equal authorised + target_profit"
            )
        return self


ObjectiveStatus = Literal[
    "pending_confirmation",
    "active",
    "target_reached",
    "capital_exhausted",
    "deadline_reached",
    "temporarily_infeasible",
    "paused",
    "stopped_by_user",
    "truth_degraded",
    "insufficient_historical_evidence",
]

FeasibilityClassification = Literal[
    "unknown",
    "high",
    "medium",
    "low",
    "infeasible",
]

ObjectiveStance = Literal[
    "observe",
    "accumulate",
    "press",
    "defend",
    "deadline",
    "infeasible",
]


class SessionObjectiveState(BaseModel):
    """Durable runtime objective state reconstructed from ledger/broker truth."""

    model_config = ConfigDict(extra="forbid")

    objective_id: UUID
    session_id: str
    status: ObjectiveStatus

    authorised_capital_usd: Decimal
    target_profit_usd: Decimal
    target_ending_equity_usd: Decimal

    # Explicit exposure split (both encumber authorised capital)
    working_order_reservation_usd: Decimal = Decimal("0.00")
    filled_position_exposure_usd: Decimal = Decimal("0.00")
    # Alias: total encumbered = working + filled
    reserved_capital_usd: Decimal = Decimal("0.00")
    available_capital_usd: Decimal
    realised_pnl_usd: Decimal = Decimal("0.00")
    unrealised_pnl_usd: Decimal = Decimal("0.00")

    progress_to_goal_pct: Decimal = Decimal("0.00")
    required_profit_remaining_usd: Decimal
    time_remaining_seconds: int

    estimated_success_probability: Decimal | None = None
    feasibility_classification: FeasibilityClassification = "unknown"
    current_stance: ObjectiveStance = "observe"

    last_recomputed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    version: int = 1
    entries_paused: bool = False
    open_position_count: int = 0
    max_concurrent_positions: int = 1
    deadline_exchange_time: datetime | None = None
    truth_degraded: bool = False

    @field_validator(
        "authorised_capital_usd",
        "target_profit_usd",
        "target_ending_equity_usd",
        "working_order_reservation_usd",
        "filled_position_exposure_usd",
        "reserved_capital_usd",
        "available_capital_usd",
        "realised_pnl_usd",
        "unrealised_pnl_usd",
        "progress_to_goal_pct",
        "required_profit_remaining_usd",
        mode="before",
    )
    @classmethod
    def _dec(cls, value: object) -> Decimal:
        return _money(value)  # type: ignore[arg-type]

    @field_validator("last_recomputed_at", "deadline_exchange_time")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @property
    def total_encumbered_usd(self) -> Decimal:
        return (
            self.working_order_reservation_usd + self.filled_position_exposure_usd
        ).quantize(Decimal("0.01"))


ExposureStatus = Literal[
    "working_order_reservation",
    "filled_position_exposure",
    "partial",
    "released",
    "closed",
]

# Backward-compatible alias used by older call sites / tests.
ReservationStatus = ExposureStatus


class CapitalExposure(BaseModel):
    """Working-order reservation and/or filled-position cost-basis exposure.

    Idempotent by ``client_order_id``. Available capital invariant:

        available = authorised - filled_position_exposure - working_order_reservation
    """

    model_config = ConfigDict(extra="forbid")

    exposure_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    session_id: str
    client_order_id: str
    broker_order_id: str | None = None
    contract_id: str | None = None
    position_lifecycle_id: str | None = None

    estimated_premium_per_contract_usd: Decimal
    requested_quantity: int = 1
    working_quantity: int = 0
    filled_quantity: int = 0
    average_fill_price: Decimal | None = None  # option premium per share

    working_order_reservation_usd: Decimal = Decimal("0.00")
    filled_exposure_usd: Decimal = Decimal("0.00")

    status: ExposureStatus = "working_order_reservation"
    objective_state_version: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "estimated_premium_per_contract_usd",
        "working_order_reservation_usd",
        "filled_exposure_usd",
        "average_fill_price",
        mode="before",
    )
    @classmethod
    def _dec(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _money(value)  # type: ignore[arg-type]

    @property
    def total_encumbered_usd(self) -> Decimal:
        return (
            self.working_order_reservation_usd + self.filled_exposure_usd
        ).quantize(Decimal("0.01"))

    @property
    def reserved_usd(self) -> Decimal:
        """Backward-compatible total encumbrance for this row."""
        return self.total_encumbered_usd

    @property
    def estimated_premium_usd(self) -> Decimal:
        return (
            self.estimated_premium_per_contract_usd
            * Decimal("100")
            * Decimal(max(1, self.requested_quantity))
        ).quantize(Decimal("0.01"))

    def derive_status(self) -> ExposureStatus:
        if self.status in {"released", "closed"}:
            return self.status
        if self.filled_exposure_usd > 0 and self.working_order_reservation_usd > 0:
            return "partial"
        if self.filled_exposure_usd > 0:
            return "filled_position_exposure"
        if self.working_order_reservation_usd > 0:
            return "working_order_reservation"
        return "released"

    @property
    def reservation_id(self) -> UUID:
        """Backward-compatible alias for exposure_id."""
        return self.exposure_id


# Alias retained for imports / gradual migration.
CapitalReservation = CapitalExposure


class ObjectiveContext(BaseModel):
    """Sanitised objective section for goal-aware agents only."""

    model_config = ConfigDict(extra="forbid")

    authorised_capital_usd: Decimal
    available_capital_usd: Decimal
    reserved_capital_usd: Decimal
    working_order_reservation_usd: Decimal = Decimal("0.00")
    filled_position_exposure_usd: Decimal = Decimal("0.00")
    realised_pnl_usd: Decimal
    unrealised_pnl_usd: Decimal = Decimal("0.00")
    target_profit_usd: Decimal
    required_profit_remaining_usd: Decimal
    progress_to_goal_pct: Decimal
    time_remaining_seconds: int
    feasibility_classification: str
    estimated_success_probability: Decimal | None
    stance: str
    maximum_permitted_loss_usd: Decimal
    maximum_concurrent_positions: int
    objective_id: str | None = None
    objective_version: int | None = None
    status: str | None = None
    policy: str | None = None
    no_trade_target_hit_estimate: Decimal | None = None

    def model_dump_for_hash(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class GoalFeasibilityAssessment(BaseModel):
    """Deterministic-first feasibility assessment."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    snapshot_id: UUID
    classification: Literal["high", "medium", "low", "infeasible"]
    estimated_success_probability: Decimal | None = None

    required_return_remaining_pct: Decimal
    required_profit_remaining_usd: Decimal
    time_remaining_seconds: int

    estimated_opportunities_remaining: int | None = None
    minimum_required_expected_value_usd: Decimal | None = None
    minimum_required_win_probability: Decimal | None = None
    minimum_required_payoff_ratio: Decimal | None = None

    binding_constraints: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    calculation_inputs: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StrategyObjectiveEstimate(BaseModel):
    """Typed EV/capital estimate for a strategy — never model prose alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    estimate_id: UUID = Field(default_factory=uuid4)
    strategy_id: UUID
    objective_id: UUID
    snapshot_id: UUID

    expected_value_usd: Decimal | None = None
    estimated_win_probability: Decimal | None = None
    estimated_payoff_ratio: Decimal | None = None
    estimated_resolution_seconds: int | None = None

    capital_required_usd: Decimal
    maximum_loss_usd: Decimal

    calculation_method: str
    assumptions: tuple[str, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    quote_inputs: dict[str, Any] = Field(default_factory=dict)
    valid: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Historical-EV provenance (reproducible from persisted inputs)
    historical_query_id: UUID | None = None
    historical_summary_id: UUID | None = None
    comparable_episode_ids: tuple[UUID, ...] = ()
    evaluation_ids: tuple[UUID, ...] = ()
    similarity_policy_version: str | None = None
    sample_count: int = 0
    effective_sample_size: Decimal | None = None
    average_similarity: Decimal | None = None
    lower_confidence_bound_ev_usd: Decimal | None = None
    estimate_version: str = "1.0.0"
    valid_until: datetime | None = None

    @field_validator(
        "expected_value_usd",
        "estimated_win_probability",
        "estimated_payoff_ratio",
        "capital_required_usd",
        "maximum_loss_usd",
        "effective_sample_size",
        "average_similarity",
        "lower_confidence_bound_ev_usd",
        mode="before",
    )
    @classmethod
    def _dec(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _money(value)  # type: ignore[arg-type]

    @field_validator("created_at", "valid_until")
    @classmethod
    def _aware_est(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class ObjectiveStrategyScore(BaseModel):
    """Typed score for a strategy candidate or no-trade."""

    model_config = ConfigDict(extra="forbid")

    score_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    strategy_id: UUID | None
    snapshot_id: UUID
    estimate_id: UUID | None = None

    expected_value_usd: Decimal | None = None
    expected_return_on_authorised_capital_pct: Decimal | None = None
    estimated_win_probability: Decimal | None = None
    estimated_total_loss_probability: Decimal | None = None
    estimated_payoff_ratio: Decimal | None = None
    estimated_resolution_seconds: int | None = None

    target_probability_before: Decimal | None = None
    target_probability_after: Decimal | None = None
    target_probability_delta: Decimal | None = None

    maximum_loss_usd: Decimal
    capital_required_usd: Decimal
    opportunity_cost_usd: Decimal | None = None

    calculation_inputs: dict[str, Any] = Field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    evidence_ids: tuple[UUID, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    valid: bool = True
    invalidation_codes: tuple[str, ...] = ()
    is_no_trade: bool = False


class ObjectiveSizingDecision(BaseModel):
    """Deterministic final quantity — agent quantity is advisory only."""

    model_config = ConfigDict(extra="forbid")

    sizing_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    strategy_id: UUID | None = None
    estimate_id: UUID | None = None
    requested_quantity: int | None = None
    approved_quantity: int

    premium_per_contract_usd: Decimal
    approved_notional_usd: Decimal
    available_before_usd: Decimal
    available_after_reservation_usd: Decimal

    aggression_cap: Decimal
    estimated_expected_value_usd: Decimal | None = None
    estimated_win_probability: Decimal | None = None
    expected_r: Decimal | None = None

    approved: bool
    reason_codes: tuple[str, ...] = ()
    calculation_inputs: dict[str, Any] = Field(default_factory=dict)
    is_probe: bool = False


def build_definition(
    *,
    session_id: str,
    authorised_capital_usd: Decimal | float,
    target_profit_pct: Decimal | float,
    deadline_exchange_time: datetime,
    max_concurrent_positions: int,
    accepted_total_loss_risk: bool,
    pause_entries_when_goal_met: bool = True,
) -> SessionObjectiveDefinition:
    """Derive profit/ending equity and construct a validated definition."""
    auth = _money(authorised_capital_usd)
    pct = _money(target_profit_pct)
    profit = (auth * pct / Decimal("100")).quantize(Decimal("0.01"))
    ending = (auth + profit).quantize(Decimal("0.01"))
    return SessionObjectiveDefinition(
        session_id=session_id,
        authorised_capital_usd=auth,
        target_profit_pct=pct,
        target_profit_usd=profit,
        target_ending_equity_usd=ending,
        deadline_exchange_time=deadline_exchange_time,
        max_concurrent_positions=max_concurrent_positions,
        pause_entries_when_goal_met=pause_entries_when_goal_met,
        accepted_total_loss_risk=accepted_total_loss_risk,
    )


def state_to_context(state: SessionObjectiveState) -> ObjectiveContext:
    encumbered = state.total_encumbered_usd
    return ObjectiveContext(
        authorised_capital_usd=state.authorised_capital_usd,
        available_capital_usd=state.available_capital_usd,
        reserved_capital_usd=encumbered,
        working_order_reservation_usd=state.working_order_reservation_usd,
        filled_position_exposure_usd=state.filled_position_exposure_usd,
        realised_pnl_usd=state.realised_pnl_usd,
        unrealised_pnl_usd=getattr(state, "unrealised_pnl_usd", Decimal("0.00"))
        or Decimal("0.00"),
        target_profit_usd=state.target_profit_usd,
        required_profit_remaining_usd=state.required_profit_remaining_usd,
        progress_to_goal_pct=state.progress_to_goal_pct,
        time_remaining_seconds=state.time_remaining_seconds,
        feasibility_classification=state.feasibility_classification,
        estimated_success_probability=state.estimated_success_probability,
        stance=state.current_stance,
        maximum_permitted_loss_usd=state.authorised_capital_usd,
        maximum_concurrent_positions=state.max_concurrent_positions,
        objective_id=str(state.objective_id),
        objective_version=state.version,
        status=state.status,
        policy=getattr(state, "policy", None),
    )


def premium_notional_usd(
    premium_per_contract: Decimal | float, quantity: int
) -> Decimal:
    return (
        Decimal(str(premium_per_contract)) * Decimal("100") * Decimal(int(quantity))
    ).quantize(Decimal("0.01"))


# Roles that must NOT receive ObjectiveContext (perception-neutral).
OBJECTIVE_NEUTRAL_ROLES: frozenset[str] = frozenset(
    {
        "market_structure",
        "volatility",
        "options_microstructure",
        "temporal_context",
        "anomaly",
        "pattern_miner",
        "sequence_analyst",
        "analogy_retriever",
        "world_model_synthesiser",
    }
)

# Roles that may receive sanitised ObjectiveContext.
OBJECTIVE_AWARE_ROLES: frozenset[str] = frozenset(
    {
        "bullish_inventor",
        "bearish_inventor",
        "neutral_advocate",
        "strategy_advocate",
        "meta_decision",
        "entry_tactician",
        "position_thesis",
        "position_decision",
        "order_manager",
    }
)

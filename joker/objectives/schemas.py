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

    @field_validator(
        "authorised_capital_usd",
        "target_profit_usd",
        "target_ending_equity_usd",
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


ReservationStatus = Literal["open", "released", "converted", "partial"]


class CapitalReservation(BaseModel):
    """Idempotent capital reservation keyed by client_order_id."""

    model_config = ConfigDict(extra="forbid")

    reservation_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    session_id: str
    client_order_id: str
    broker_order_id: str | None = None
    estimated_premium_usd: Decimal
    reserved_usd: Decimal
    status: ReservationStatus = "open"
    objective_state_version: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("estimated_premium_usd", "reserved_usd", mode="before")
    @classmethod
    def _dec(cls, value: object) -> Decimal:
        return _money(value)  # type: ignore[arg-type]


class ObjectiveContext(BaseModel):
    """Sanitised objective section for goal-aware agents only."""

    model_config = ConfigDict(extra="forbid")

    authorised_capital_usd: Decimal
    available_capital_usd: Decimal
    reserved_capital_usd: Decimal
    realised_pnl_usd: Decimal
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ObjectiveStrategyScore(BaseModel):
    """Typed score for a strategy candidate or no-trade."""

    model_config = ConfigDict(extra="forbid")

    score_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    strategy_id: UUID | None
    snapshot_id: UUID

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
    valid: bool = True
    invalidation_codes: tuple[str, ...] = ()
    is_no_trade: bool = False


class ObjectiveSizingDecision(BaseModel):
    """Deterministic final quantity — agent quantity is advisory only."""

    model_config = ConfigDict(extra="forbid")

    sizing_id: UUID = Field(default_factory=uuid4)
    objective_id: UUID
    strategy_id: UUID | None = None
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
    return ObjectiveContext(
        authorised_capital_usd=state.authorised_capital_usd,
        available_capital_usd=state.available_capital_usd,
        reserved_capital_usd=state.reserved_capital_usd,
        realised_pnl_usd=state.realised_pnl_usd,
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
    )


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

"""Target-attainment objective policy — maximize P(goal by deadline).

Authoritative paper policy when ``objective.policy == target_attainment``.
Compares each candidate × affordable quantity against no-trade using estimated
target-hit probability. Never fabricates calibrated probabilities. Never
implements martingale. Never bypasses execution correctness constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.scoring import StrategyScoreInput
from joker.objectives.session_eligibility import (
    ObjectiveSessionEligibility,
    ObjectiveSessionState,
)
from joker.time.clock import SessionPhase

ProbabilityEstimateType = Literal[
    "calibrated",
    "empirical_low_sample",
    "ordinal",
    "unknown",
]

AttainmentFeasibility = Literal[
    "attainable",
    "low_probability",
    "physically_impossible",
]

# Typed data-quality findings that imply stale / unusable market truth.
_STALE_DATA_QUALITY_CODES: frozenset[str] = frozenset(
    {
        "stale",
        "stale_quote",
        "stale_quotes",
        "stale_underlying",
        "stale_option_surface",
        "quote_stale",
        "underlying_stale",
        "surface_stale",
        "data_stale",
    }
)


class TargetAttainmentAction(StrEnum):
    ENTER = "enter"
    WAIT = "wait"
    BLOCK = "block"


@dataclass(frozen=True)
class TargetProbabilityEstimate:
    """Target-hit probability estimate with explicit calibration class."""

    p_goal: Decimal | None
    estimate_type: ProbabilityEstimateType
    sample_count: int = 0
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None
    uncertainty_reasons: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "p_goal": str(self.p_goal) if self.p_goal is not None else None,
            "estimate_type": self.estimate_type,
            "sample_count": self.sample_count,
            "lower_bound": str(self.lower_bound) if self.lower_bound is not None else None,
            "upper_bound": str(self.upper_bound) if self.upper_bound is not None else None,
            "uncertainty_reasons": list(self.uncertainty_reasons),
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class TargetAttainmentContext:
    """Operator objective + capital truth for one decision cycle."""

    objective_id: UUID
    snapshot_id: UUID
    authorised_capital_usd: Decimal
    available_capital_usd: Decimal
    reserved_capital_usd: Decimal
    realised_pnl_usd: Decimal
    unrealised_pnl_usd: Decimal
    target_profit_usd: Decimal
    remaining_goal_gap_usd: Decimal
    time_remaining_seconds: int
    objective_duration_seconds: int
    elapsed_seconds: int
    open_position_count: int
    working_order_count: int
    max_concurrent_positions: int
    maximum_authorised_contracts: int
    allow_full_remaining_capital: bool = True
    maximum_capital_fraction: float = 1.0
    minimum_calibrated_samples: int = 20
    # Authoritative exchange phase controls physical eligibility.
    exchange_session_phase: str | None = "regular"
    # Similarity bucket is for historical comparison only (open/midday/close).
    session_similarity_bucket: str | None = None
    session_phase: str = "regular"  # back-compat alias → exchange phase
    objective_version: int = 1
    market_usable_for_execution: bool = True
    option_surface_usable: bool = True
    underlying_symbol: str = "SPY"
    spy_last: Decimal | None = None
    data_quality_codes: tuple[str, ...] = ()
    future_opportunity_estimate: Decimal | None = None  # ordinal 0-1 if known

    @property
    def fraction_remaining(self) -> Decimal:
        duration = max(int(self.objective_duration_seconds), 1)
        return (
            Decimal(max(0, int(self.time_remaining_seconds))) / Decimal(duration)
        ).quantize(Decimal("0.0001"))

    @classmethod
    def from_state(
        cls,
        state: SessionObjectiveState,
        *,
        snapshot_id: UUID,
        objective_duration_seconds: int | None = None,
        maximum_authorised_contracts: int = 20,
        allow_full_remaining_capital: bool = True,
        maximum_capital_fraction: float = 1.0,
        minimum_calibrated_samples: int = 20,
        working_order_count: int = 0,
        exchange_session_phase: str | None = None,
        session_similarity_bucket: str | None = None,
        session_phase: str | None = None,
        market_usable_for_execution: bool = True,
        option_surface_usable: bool = True,
        spy_last: Decimal | None = None,
        data_quality_codes: tuple[str, ...] = (),
        future_opportunity_estimate: Decimal | None = None,
    ) -> TargetAttainmentContext:
        # Prefer durable original duration from objective state — never remaining time.
        duration = (
            objective_duration_seconds
            if objective_duration_seconds is not None
            else state.objective_duration_seconds
        )
        if duration is None or int(duration) <= 0:
            duration = max(int(state.time_remaining_seconds), 1)
        elapsed = int(getattr(state, "elapsed_seconds", 0) or 0)
        if elapsed <= 0:
            elapsed = max(0, int(duration) - int(state.time_remaining_seconds))
        phase = exchange_session_phase or session_phase or "regular"
        return cls(
            objective_id=state.objective_id,
            snapshot_id=snapshot_id,
            authorised_capital_usd=Decimal(str(state.authorised_capital_usd)),
            available_capital_usd=Decimal(str(state.available_capital_usd)),
            reserved_capital_usd=Decimal(str(state.reserved_capital_usd)),
            realised_pnl_usd=Decimal(str(state.realised_pnl_usd)),
            unrealised_pnl_usd=Decimal(str(state.unrealised_pnl_usd)),
            target_profit_usd=Decimal(str(state.target_profit_usd)),
            remaining_goal_gap_usd=Decimal(str(state.required_profit_remaining_usd)),
            time_remaining_seconds=int(state.time_remaining_seconds),
            objective_duration_seconds=int(duration),
            elapsed_seconds=elapsed,
            open_position_count=int(state.open_position_count),
            working_order_count=int(working_order_count),
            max_concurrent_positions=int(state.max_concurrent_positions),
            maximum_authorised_contracts=int(maximum_authorised_contracts),
            allow_full_remaining_capital=allow_full_remaining_capital,
            maximum_capital_fraction=float(maximum_capital_fraction),
            minimum_calibrated_samples=int(minimum_calibrated_samples),
            exchange_session_phase=phase,
            session_similarity_bucket=session_similarity_bucket,
            session_phase=str(phase),
            objective_version=int(state.version),
            market_usable_for_execution=market_usable_for_execution,
            option_surface_usable=option_surface_usable,
            spy_last=spy_last,
            data_quality_codes=data_quality_codes,
            future_opportunity_estimate=future_opportunity_estimate,
        )


@dataclass(frozen=True)
class TargetAttainmentContractCandidate:
    """One strategy × executable contract opportunity before quantity expansion."""

    strategy_id: UUID
    contract_id: str
    option_type: str
    strike: Decimal
    premium_per_contract_usd: Decimal  # execution-reference (ask for long buys)
    bid: Decimal
    ask: Decimal
    mid: Decimal
    relative_spread: Decimal
    liquidity_score: float = 1.0
    estimated_win_probability: Decimal | None = None
    expected_value_usd: Decimal | None = None
    estimated_payoff_ratio: Decimal | None = None
    estimated_useful_upside_usd: Decimal | None = None
    estimated_resolution_seconds: int | None = None
    maximum_loss_usd_per_contract: Decimal = Decimal("0")
    historical_sample_count: int = 0
    historical_hit_rate: Decimal | None = None
    evidence_ids: tuple[UUID, ...] = ()
    direction: str | None = None
    quote_timestamp: str | None = None
    assumptions: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    calculation_method: str = "unknown"

    def as_candidate(self) -> TargetAttainmentCandidate:
        return TargetAttainmentCandidate(
            strategy_id=self.strategy_id,
            premium_per_contract_usd=self.premium_per_contract_usd,
            estimated_win_probability=self.estimated_win_probability,
            expected_value_usd=self.expected_value_usd,
            estimated_payoff_ratio=self.estimated_payoff_ratio,
            estimated_useful_upside_usd=self.estimated_useful_upside_usd,
            estimated_resolution_seconds=self.estimated_resolution_seconds,
            maximum_loss_usd_per_contract=(
                self.maximum_loss_usd_per_contract
                if self.maximum_loss_usd_per_contract > 0
                else self.premium_per_contract_usd * Decimal("100")
            ),
            sample_count=self.historical_sample_count,
            historical_hit_rate=self.historical_hit_rate,
            calculation_method=self.calculation_method,
            evidence_ids=self.evidence_ids,
            assumptions=self.assumptions,
            uncertainty_reasons=self.uncertainty_reasons,
            direction=self.direction,
            contract_id=self.contract_id,
        )


@dataclass(frozen=True)
class TargetAttainmentCandidate:
    """One strategy/contract opportunity before quantity expansion."""

    strategy_id: UUID | None
    premium_per_contract_usd: Decimal
    estimated_win_probability: Decimal | None
    expected_value_usd: Decimal | None
    estimated_payoff_ratio: Decimal | None  # win_pnl / loss_pnl
    estimated_useful_upside_usd: Decimal | None  # profit if win (per contract * 100 scale already)
    estimated_resolution_seconds: int | None
    maximum_loss_usd_per_contract: Decimal
    sample_count: int = 0
    historical_hit_rate: Decimal | None = None
    lower_confidence_bound_ev_usd: Decimal | None = None
    calculation_method: str = "unknown"
    evidence_ids: tuple[UUID, ...] = ()
    assumptions: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    direction: str | None = None
    contract_id: str | None = None


@dataclass
class QuantityAttainmentEvaluation:
    """Evaluation of one (strategy, contract, quantity) tuple."""

    evaluation_id: UUID = field(default_factory=uuid4)
    strategy_id: UUID | None = None
    contract_id: str | None = None
    quantity: int = 0
    evaluation_premium_usd: Decimal = Decimal("0")
    capital_required_usd: Decimal = Decimal("0")
    maximum_loss_usd: Decimal = Decimal("0")
    useful_upside_usd: Decimal = Decimal("0")
    p_goal: TargetProbabilityEstimate = field(
        default_factory=lambda: TargetProbabilityEstimate(
            p_goal=None, estimate_type="unknown"
        )
    )
    expected_pnl_usd: Decimal | None = None
    median_pnl_usd: Decimal | None = None
    lower_tail_usd: Decimal | None = None
    upper_tail_usd: Decimal | None = None
    estimated_resolution_seconds: int | None = None
    can_close_goal_gap: bool = False
    material_progress: bool = False
    physically_impossible: bool = False
    reason_codes: list[str] = field(default_factory=list)
    selected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": str(self.evaluation_id),
            "strategy_id": str(self.strategy_id) if self.strategy_id else None,
            "contract_id": self.contract_id,
            "quantity": self.quantity,
            "evaluation_premium_usd": str(self.evaluation_premium_usd),
            "capital_required_usd": str(self.capital_required_usd),
            "maximum_loss_usd": str(self.maximum_loss_usd),
            "useful_upside_usd": str(self.useful_upside_usd),
            "p_goal": self.p_goal.as_dict(),
            "expected_pnl_usd": str(self.expected_pnl_usd)
            if self.expected_pnl_usd is not None
            else None,
            "can_close_goal_gap": self.can_close_goal_gap,
            "material_progress": self.material_progress,
            "physically_impossible": self.physically_impossible,
            "reason_codes": list(self.reason_codes),
            "selected": self.selected,
            "estimated_resolution_seconds": self.estimated_resolution_seconds,
        }


@dataclass
class NoTradeAttainmentEvaluation:
    evaluation_id: UUID = field(default_factory=uuid4)
    p_goal: TargetProbabilityEstimate = field(
        default_factory=lambda: TargetProbabilityEstimate(
            p_goal=None, estimate_type="unknown"
        )
    )
    reason_codes: list[str] = field(default_factory=list)
    selected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": str(self.evaluation_id),
            "p_goal": self.p_goal.as_dict(),
            "reason_codes": list(self.reason_codes),
            "selected": self.selected,
        }


@dataclass
class TargetAttainmentDecision:
    """Authoritative target-attainment decision for one cycle."""

    decision_id: UUID = field(default_factory=uuid4)
    action: TargetAttainmentAction = TargetAttainmentAction.WAIT
    feasibility: AttainmentFeasibility = "low_probability"
    selected_strategy_id: UUID | None = None
    selected_contract_id: str | None = None
    selected_quantity: int = 0
    selected_capital_usd: Decimal = Decimal("0")
    selected_evaluation_premium_usd: Decimal | None = None
    selected_p_goal: TargetProbabilityEstimate | None = None
    no_trade_p_goal: TargetProbabilityEstimate | None = None
    probability_delta: Decimal | None = None
    snapshot_id: UUID | None = None
    objective_version: int | None = None
    authoritative: bool = True
    reason_codes: list[str] = field(default_factory=list)
    quantity_evaluations: list[QuantityAttainmentEvaluation] = field(
        default_factory=list
    )
    no_trade: NoTradeAttainmentEvaluation | None = None
    baseline_shadow: dict[str, Any] | None = None
    exchange_session_phase: str | None = None
    session_similarity_bucket: str | None = None
    objective_duration_seconds: int | None = None
    elapsed_seconds: int | None = None
    time_remaining_seconds: int | None = None
    fraction_remaining: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": str(self.decision_id),
            "action": self.action.value,
            "feasibility": self.feasibility,
            "selected_strategy_id": (
                str(self.selected_strategy_id) if self.selected_strategy_id else None
            ),
            "selected_contract_id": self.selected_contract_id,
            "selected_quantity": self.selected_quantity,
            "selected_capital_usd": str(self.selected_capital_usd),
            "selected_evaluation_premium_usd": (
                str(self.selected_evaluation_premium_usd)
                if self.selected_evaluation_premium_usd is not None
                else None
            ),
            "selected_p_goal": (
                self.selected_p_goal.as_dict() if self.selected_p_goal else None
            ),
            "no_trade_p_goal": (
                self.no_trade_p_goal.as_dict() if self.no_trade_p_goal else None
            ),
            "probability_delta": (
                str(self.probability_delta) if self.probability_delta is not None else None
            ),
            "snapshot_id": str(self.snapshot_id) if self.snapshot_id else None,
            "objective_version": self.objective_version,
            "authoritative": self.authoritative,
            "reason_codes": list(self.reason_codes),
            "quantity_evaluations": [q.as_dict() for q in self.quantity_evaluations],
            "no_trade": self.no_trade.as_dict() if self.no_trade else None,
            "baseline_shadow": self.baseline_shadow,
            "exchange_session_phase": self.exchange_session_phase,
            "session_similarity_bucket": self.session_similarity_bucket,
            "objective_duration_seconds": self.objective_duration_seconds,
            "elapsed_seconds": self.elapsed_seconds,
            "time_remaining_seconds": self.time_remaining_seconds,
            "fraction_remaining": self.fraction_remaining,
        }


def _d(value: Decimal | float | int | str | None, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    return Decimal(str(value))


def classify_physical_impossibility(
    ctx: TargetAttainmentContext,
    *,
    session_state: ObjectiveSessionState | None = None,
) -> tuple[bool, list[str]]:
    """Hard correctness / physical blocks — not low-probability judgements.

    Physical eligibility uses authoritative exchange phase only. Similarity
    buckets such as open/midday/close never alone block entry.
    """
    codes: list[str] = []
    if ctx.time_remaining_seconds <= 0:
        codes.append("deadline_passed")

    if session_state is not None:
        if session_state.eligibility is ObjectiveSessionEligibility.UNKNOWN:
            codes.append("exchange_session_truth_unavailable")
        elif not session_state.entries_permitted:
            codes.append("market_not_regular")
            codes.extend(session_state.reason_codes)
    else:
        phase = (ctx.exchange_session_phase or ctx.session_phase or "").strip().lower()
        if phase in {"", "unknown", "none"}:
            codes.append("exchange_session_truth_unavailable")
        elif phase != SessionPhase.REGULAR.value:
            # Similarity buckets are NOT exchange phases — ignore them here.
            if phase not in {"open", "midday", "close", "regular"}:
                codes.append("market_not_regular")
            # If caller passed a similarity bucket as session_phase by mistake,
            # treat open/midday/close as regular (compat) only when no session_state.
            # Prefer resolving via ObjectiveSessionState in production wiring.

    if not ctx.market_usable_for_execution:
        codes.append("market_truth_unusable")
    if not ctx.option_surface_usable:
        codes.append("option_surface_unusable")
    if ctx.underlying_symbol.upper() != "SPY":
        codes.append("wrong_underlying")
    if ctx.available_capital_usd <= 0 and ctx.open_position_count == 0:
        codes.append("no_available_capital")
    if ctx.open_position_count >= ctx.max_concurrent_positions:
        codes.append("max_concurrent_positions")
    normalized_dq = {str(c).strip().lower() for c in ctx.data_quality_codes}
    if normalized_dq & _STALE_DATA_QUALITY_CODES:
        codes.append("stale_data_quality")
    return bool(codes), codes


def max_affordable_quantity(
    *,
    premium_per_contract_usd: Decimal,
    available_capital_usd: Decimal,
    maximum_authorised_contracts: int,
    maximum_capital_fraction: float = 1.0,
) -> int:
    """Largest quantity whose premium risk fits remaining authorised capital."""
    premium = _d(premium_per_contract_usd)
    if premium <= 0:
        return 0
    # Option premium is quoted per share; contract notional = premium * 100.
    cost_per = (premium * Decimal("100")).quantize(Decimal("0.01"))
    budget = (
        _d(available_capital_usd) * Decimal(str(maximum_capital_fraction))
    ).quantize(Decimal("0.01"))
    if cost_per <= 0 or budget <= 0:
        return 0
    by_capital = int(budget // cost_per)
    return max(0, min(by_capital, int(maximum_authorised_contracts)))


def estimate_useful_upside(
    candidate: TargetAttainmentCandidate,
    quantity: int,
) -> Decimal:
    """Estimate useful upside USD if the trade wins."""
    if candidate.estimated_useful_upside_usd is not None:
        return (_d(candidate.estimated_useful_upside_usd) * Decimal(quantity)).quantize(
            Decimal("0.01")
        )
    # Fallback: payoff_ratio * max_loss, or 1R assumption from premium.
    max_loss = _d(candidate.maximum_loss_usd_per_contract) * Decimal(quantity)
    if candidate.estimated_payoff_ratio is not None and max_loss > 0:
        return (max_loss * _d(candidate.estimated_payoff_ratio)).quantize(
            Decimal("0.01")
        )
    # Unknown upside — treat as unable to prove goal closure.
    return Decimal("0.00")


def estimate_target_hit_probability(
    *,
    ctx: TargetAttainmentContext,
    win_p: Decimal | None,
    useful_upside_usd: Decimal,
    capital_required_usd: Decimal,
    sample_count: int,
    historical_hit_rate: Decimal | None,
    resolution_seconds: int | None,
    is_no_trade: bool = False,
) -> TargetProbabilityEstimate:
    """Estimate P(goal | action) without fabricating calibrated confidence."""
    gap = ctx.remaining_goal_gap_usd
    if gap <= 0:
        return TargetProbabilityEstimate(
            p_goal=Decimal("1.0000"),
            estimate_type="ordinal",
            sample_count=sample_count,
            assumptions=("goal_already_met",),
        )

    if is_no_trade:
        # Waiting value decays with urgency when gap remains and time shrinks.
        duration = max(ctx.objective_duration_seconds, 1)
        frac_left = Decimal(ctx.time_remaining_seconds) / Decimal(duration)
        future = ctx.future_opportunity_estimate
        if future is None:
            # Ordinal: remaining time fraction dampened when gap is large vs capital.
            gap_ratio = min(
                Decimal("1"),
                gap / max(ctx.authorised_capital_usd, Decimal("0.01")),
            )
            p = (frac_left * (Decimal("1") - gap_ratio * Decimal("0.5"))).quantize(
                Decimal("0.0001")
            )
            return TargetProbabilityEstimate(
                p_goal=max(Decimal("0"), min(Decimal("1"), p)),
                estimate_type="ordinal",
                sample_count=0,
                uncertainty_reasons=("no_trade_opportunity_cost_ordinal",),
                assumptions=("wait_value_decays_with_urgency_and_gap",),
            )
        p = (frac_left * _d(future)).quantize(Decimal("0.0001"))
        return TargetProbabilityEstimate(
            p_goal=max(Decimal("0"), min(Decimal("1"), p)),
            estimate_type="ordinal",
            sample_count=0,
            assumptions=("future_opportunity_estimate_provided",),
        )

    if resolution_seconds is not None and resolution_seconds > ctx.time_remaining_seconds:
        return TargetProbabilityEstimate(
            p_goal=Decimal("0.0000"),
            estimate_type="ordinal",
            sample_count=sample_count,
            uncertainty_reasons=("resolution_after_deadline",),
        )

    can_close = useful_upside_usd >= gap
    material = useful_upside_usd >= (gap * Decimal("0.25"))
    if useful_upside_usd <= 0:
        return TargetProbabilityEstimate(
            p_goal=Decimal("0.0000"),
            estimate_type="ordinal",
            sample_count=sample_count,
            uncertainty_reasons=("no_useful_upside",),
        )

    # Base success likelihood from historical/win evidence.
    base: Decimal | None = None
    estimate_type: ProbabilityEstimateType = "unknown"
    uncertainty: list[str] = []
    assumptions: list[str] = []

    if (
        sample_count >= ctx.minimum_calibrated_samples
        and historical_hit_rate is not None
    ):
        base = _d(historical_hit_rate)
        estimate_type = "calibrated"
        assumptions.append("historical_hit_rate_calibrated")
    elif sample_count > 0 and historical_hit_rate is not None:
        base = _d(historical_hit_rate)
        estimate_type = "empirical_low_sample"
        uncertainty.append("below_minimum_calibrated_samples")
    elif win_p is not None:
        base = _d(win_p)
        estimate_type = "ordinal"
        uncertainty.append("agent_or_estimate_win_probability_only")
    else:
        uncertainty.append("probability_unknown")
        # Ordinal capability score — not presented as calibrated.
        if can_close:
            base = Decimal("0.35")
            estimate_type = "ordinal"
            assumptions.append("ordinal_can_close_gap_without_calibrated_p")
        elif material:
            base = Decimal("0.20")
            estimate_type = "ordinal"
            assumptions.append("ordinal_material_progress_only")
        else:
            return TargetProbabilityEstimate(
                p_goal=None,
                estimate_type="unknown",
                sample_count=sample_count,
                uncertainty_reasons=tuple(uncertainty + ["cannot_rank_without_evidence"]),
            )

    assert base is not None
    # Scale by goal-closure capability: cannot close gap → heavily discount.
    if can_close:
        p = base
        assumptions.append("upside_can_close_full_gap")
    elif material:
        progress = min(Decimal("1"), useful_upside_usd / gap)
        p = (base * progress * Decimal("0.5")).quantize(Decimal("0.0001"))
        assumptions.append("partial_progress_only_discounted")
    else:
        p = (base * Decimal("0.05")).quantize(Decimal("0.0001"))
        assumptions.append("immaterial_progress_heavily_discounted")

    # Capital efficiency: consuming capital for tiny progress is worse.
    if capital_required_usd > 0 and useful_upside_usd > 0:
        assumptions.append("capital_efficiency_considered")

    p = max(Decimal("0"), min(Decimal("1"), p))
    return TargetProbabilityEstimate(
        p_goal=p,
        estimate_type=estimate_type,
        sample_count=sample_count,
        lower_bound=p * Decimal("0.8") if estimate_type != "unknown" else None,
        upper_bound=min(Decimal("1"), p * Decimal("1.2"))
        if estimate_type != "unknown"
        else None,
        uncertainty_reasons=tuple(uncertainty),
        assumptions=tuple(assumptions),
    )


def _rank_key(ev: QuantityAttainmentEvaluation) -> tuple:
    """Higher is better for target-hit ranking."""
    p = ev.p_goal.p_goal
    if p is None or ev.physically_impossible:
        return (
            Decimal("-1"),
            Decimal("-1"),
            10**12,
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
        )
    lb = ev.p_goal.lower_bound if ev.p_goal.lower_bound is not None else p
    resolution = ev.estimated_resolution_seconds or 10**9
    remaining_capital = -ev.capital_required_usd  # less capital used → higher
    expected = ev.expected_pnl_usd or Decimal("0")
    adverse = -(ev.lower_tail_usd or Decimal("0"))
    return (p, lb, -resolution, remaining_capital, expected, -adverse)


class TargetAttainmentPolicy:
    """Authoritative maximizer of P(goal by deadline) under capital constraints."""

    def decide(
        self,
        ctx: TargetAttainmentContext,
        candidates: list[TargetAttainmentCandidate],
        *,
        baseline_shadow: dict[str, Any] | None = None,
        session_state: ObjectiveSessionState | None = None,
    ) -> TargetAttainmentDecision:
        timing_fields = {
            "snapshot_id": ctx.snapshot_id,
            "objective_version": ctx.objective_version,
            "exchange_session_phase": ctx.exchange_session_phase,
            "session_similarity_bucket": ctx.session_similarity_bucket,
            "objective_duration_seconds": ctx.objective_duration_seconds,
            "elapsed_seconds": ctx.elapsed_seconds,
            "time_remaining_seconds": ctx.time_remaining_seconds,
            "fraction_remaining": str(ctx.fraction_remaining),
            "authoritative": True,
        }
        impossible, hard_codes = classify_physical_impossibility(
            ctx, session_state=session_state
        )
        if impossible:
            no_trade = NoTradeAttainmentEvaluation(
                p_goal=TargetProbabilityEstimate(
                    p_goal=Decimal("0.0000"),
                    estimate_type="ordinal",
                    uncertainty_reasons=tuple(hard_codes),
                ),
                reason_codes=list(hard_codes),
                selected=False,
            )
            return TargetAttainmentDecision(
                action=TargetAttainmentAction.BLOCK,
                feasibility="physically_impossible",
                reason_codes=list(hard_codes),
                no_trade=no_trade,
                no_trade_p_goal=no_trade.p_goal,
                baseline_shadow=baseline_shadow,
                **timing_fields,
            )

        if ctx.remaining_goal_gap_usd <= 0:
            return TargetAttainmentDecision(
                action=TargetAttainmentAction.WAIT,
                feasibility="attainable",
                reason_codes=["goal_already_achieved"],
                no_trade=NoTradeAttainmentEvaluation(
                    p_goal=TargetProbabilityEstimate(
                        p_goal=Decimal("1.0000"),
                        estimate_type="ordinal",
                        assumptions=("goal_already_met",),
                    ),
                    selected=True,
                    reason_codes=["goal_already_achieved"],
                ),
                baseline_shadow=baseline_shadow,
                **timing_fields,
            )

        no_trade_p = estimate_target_hit_probability(
            ctx=ctx,
            win_p=None,
            useful_upside_usd=Decimal("0"),
            capital_required_usd=Decimal("0"),
            sample_count=0,
            historical_hit_rate=None,
            resolution_seconds=None,
            is_no_trade=True,
        )
        no_trade_eval = NoTradeAttainmentEvaluation(
            p_goal=no_trade_p,
            reason_codes=["compare_wait_opportunity_cost"],
        )

        quantity_evals: list[QuantityAttainmentEvaluation] = []
        for cand in candidates:
            max_q = max_affordable_quantity(
                premium_per_contract_usd=cand.premium_per_contract_usd,
                available_capital_usd=ctx.available_capital_usd,
                maximum_authorised_contracts=ctx.maximum_authorised_contracts,
                maximum_capital_fraction=(
                    1.0
                    if ctx.allow_full_remaining_capital
                    else float(ctx.maximum_capital_fraction)
                ),
            )
            if max_q <= 0:
                quantity_evals.append(
                    QuantityAttainmentEvaluation(
                        strategy_id=cand.strategy_id,
                        contract_id=cand.contract_id,
                        quantity=0,
                        evaluation_premium_usd=_d(cand.premium_per_contract_usd),
                        physically_impossible=True,
                        reason_codes=["no_affordable_quantity"],
                    )
                )
                continue
            for q in range(1, max_q + 1):
                premium = _d(cand.premium_per_contract_usd)
                cost = (premium * Decimal("100") * Decimal(q)).quantize(Decimal("0.01"))
                max_loss = (
                    _d(cand.maximum_loss_usd_per_contract) * Decimal(q)
                ).quantize(Decimal("0.01"))
                upside = estimate_useful_upside(cand, q)
                reasons: list[str] = []
                physically = False
                if cost > ctx.available_capital_usd:
                    physically = True
                    reasons.append("capital_exceeds_available")
                if (
                    cand.estimated_resolution_seconds is not None
                    and cand.estimated_resolution_seconds > ctx.time_remaining_seconds
                ):
                    physically = True
                    reasons.append("resolution_after_deadline")
                p_est = estimate_target_hit_probability(
                    ctx=ctx,
                    win_p=cand.estimated_win_probability,
                    useful_upside_usd=upside,
                    capital_required_usd=cost,
                    sample_count=cand.sample_count,
                    historical_hit_rate=cand.historical_hit_rate,
                    resolution_seconds=cand.estimated_resolution_seconds,
                )
                if physically:
                    p_est = TargetProbabilityEstimate(
                        p_goal=Decimal("0.0000"),
                        estimate_type="ordinal",
                        sample_count=cand.sample_count,
                        uncertainty_reasons=tuple(reasons),
                    )
                expected = None
                if cand.expected_value_usd is not None:
                    expected = (_d(cand.expected_value_usd) * Decimal(q)).quantize(
                        Decimal("0.01")
                    )
                quantity_evals.append(
                    QuantityAttainmentEvaluation(
                        strategy_id=cand.strategy_id,
                        contract_id=cand.contract_id,
                        quantity=q,
                        evaluation_premium_usd=premium,
                        capital_required_usd=cost,
                        maximum_loss_usd=max_loss,
                        useful_upside_usd=upside,
                        p_goal=p_est,
                        expected_pnl_usd=expected,
                        lower_tail_usd=-max_loss,
                        upper_tail_usd=upside,
                        estimated_resolution_seconds=cand.estimated_resolution_seconds,
                        can_close_goal_gap=upside >= ctx.remaining_goal_gap_usd,
                        material_progress=upside
                        >= (ctx.remaining_goal_gap_usd * Decimal("0.25")),
                        physically_impossible=physically,
                        reason_codes=reasons,
                    )
                )

        viable = [e for e in quantity_evals if not e.physically_impossible]
        if not viable and not candidates:
            no_trade_eval.selected = True
            no_trade_eval.reason_codes.append("no_candidates")
            return TargetAttainmentDecision(
                action=TargetAttainmentAction.WAIT,
                feasibility="low_probability",
                reason_codes=["no_candidates"],
                quantity_evaluations=quantity_evals,
                no_trade=no_trade_eval,
                no_trade_p_goal=no_trade_p,
                selected_p_goal=no_trade_p,
                baseline_shadow=baseline_shadow,
                **timing_fields,
            )

        # Rank viable evaluations; compare best against no-trade.
        best: QuantityAttainmentEvaluation | None = None
        if viable:
            best = max(viable, key=_rank_key)

        best_p = best.p_goal.p_goal if best and best.p_goal.p_goal is not None else None
        wait_p = no_trade_p.p_goal
        delta = None
        if best_p is not None and wait_p is not None:
            delta = (best_p - wait_p).quantize(Decimal("0.0001"))

        select_action = TargetAttainmentAction.WAIT
        reasons = ["no_trade_preferred_or_equal"]
        feasibility: AttainmentFeasibility = "low_probability"
        if best is not None and best_p is not None:
            if wait_p is None or best_p > wait_p:
                select_action = TargetAttainmentAction.ENTER
                best.selected = True
                reasons = ["candidate_improves_target_hit_probability"]
                if best.can_close_goal_gap:
                    feasibility = "attainable"
                    reasons.append("candidate_can_close_goal_gap")
                elif best.material_progress:
                    feasibility = "low_probability"
                    reasons.append("material_progress_only")
                else:
                    feasibility = "low_probability"
                    reasons.append("immaterial_but_best_available")
            else:
                no_trade_eval.selected = True
                reasons = ["wait_has_higher_or_equal_target_hit_probability"]
                feasibility = "low_probability"
        else:
            no_trade_eval.selected = True
            if not viable:
                reasons = ["all_quantities_physically_impossible_or_unrankable"]
                feasibility = "physically_impossible" if quantity_evals else "low_probability"
            else:
                reasons = ["best_candidate_probability_unknown"]
                feasibility = "low_probability"

        return TargetAttainmentDecision(
            action=select_action,
            feasibility=feasibility,
            selected_strategy_id=(
                best.strategy_id
                if select_action == TargetAttainmentAction.ENTER and best
                else None
            ),
            selected_contract_id=(
                best.contract_id
                if select_action == TargetAttainmentAction.ENTER and best
                else None
            ),
            selected_quantity=(
                best.quantity
                if select_action == TargetAttainmentAction.ENTER and best
                else 0
            ),
            selected_capital_usd=(
                best.capital_required_usd
                if select_action == TargetAttainmentAction.ENTER and best
                else Decimal("0")
            ),
            selected_evaluation_premium_usd=(
                best.evaluation_premium_usd
                if select_action == TargetAttainmentAction.ENTER and best
                else None
            ),
            selected_p_goal=(
                best.p_goal
                if select_action == TargetAttainmentAction.ENTER and best
                else no_trade_p
            ),
            no_trade_p_goal=no_trade_p,
            probability_delta=delta,
            reason_codes=reasons,
            quantity_evaluations=quantity_evals,
            no_trade=no_trade_eval,
            baseline_shadow=baseline_shadow,
            **timing_fields,
        )


def candidate_from_score_input(
    cand: StrategyScoreInput,
    *,
    premium_per_contract_usd: Decimal | float | None = None,
) -> TargetAttainmentCandidate:
    """Bridge StrategyScoreInput → TargetAttainmentCandidate."""
    calc = cand.calculation_inputs or {}
    premium = premium_per_contract_usd
    if premium is None:
        # Infer from capital_required assuming qty=1 → capital = premium*100
        capital = _d(cand.capital_required_usd)
        premium = (capital / Decimal("100")) if capital > 0 else Decimal("0.01")
    sample_count = int(calc.get("sample_count") or 0)
    max_loss = (
        _d(cand.maximum_loss_usd)
        if _d(cand.maximum_loss_usd) > 0
        else _d(premium) * Decimal("100")
    )
    useful: Decimal | None = None
    if calc.get("useful_upside_usd") is not None:
        useful = _d(calc["useful_upside_usd"])
    elif (
        cand.estimated_payoff_ratio is not None
        and max_loss > 0
    ):
        useful = (max_loss * _d(cand.estimated_payoff_ratio)).quantize(Decimal("0.01"))
    elif (
        cand.expected_value_usd is not None
        and cand.estimated_win_probability is not None
        and _d(cand.estimated_win_probability) > 0
    ):
        # EV = p*upside + (1-p)*(-loss) → upside = (EV + (1-p)*loss) / p
        p = _d(cand.estimated_win_probability)
        ev = _d(cand.expected_value_usd)
        useful = ((ev + (Decimal("1") - p) * max_loss) / p).quantize(Decimal("0.01"))
        if useful < 0:
            useful = Decimal("0.00")

    hist_hit = None
    if calc.get("historical_hit_rate") is not None:
        hist_hit = _d(calc["historical_hit_rate"])

    return TargetAttainmentCandidate(
        strategy_id=cand.strategy_id,
        premium_per_contract_usd=_d(premium),
        estimated_win_probability=(
            _d(cand.estimated_win_probability)
            if cand.estimated_win_probability is not None
            else None
        ),
        expected_value_usd=(
            _d(cand.expected_value_usd) if cand.expected_value_usd is not None else None
        ),
        estimated_payoff_ratio=(
            _d(cand.estimated_payoff_ratio)
            if cand.estimated_payoff_ratio is not None
            else None
        ),
        estimated_useful_upside_usd=useful,
        estimated_resolution_seconds=cand.estimated_resolution_seconds,
        maximum_loss_usd_per_contract=max_loss,
        sample_count=sample_count,
        historical_hit_rate=hist_hit,
        lower_confidence_bound_ev_usd=(
            _d(calc["lower_confidence_bound_ev_usd"])
            if calc.get("lower_confidence_bound_ev_usd") is not None
            else None
        ),
        calculation_method=str(calc.get("calculation_method") or "unknown"),
        evidence_ids=cand.evidence_ids,
        assumptions=cand.assumptions,
        uncertainty_reasons=tuple(calc.get("uncertainty_reasons") or ()),
    )


def run_positive_ev_baseline_shadow(
    state: SessionObjectiveState,
    candidates: list[StrategyScoreInput],
    *,
    snapshot_id: UUID,
    require_positive_expected_value: bool = True,
    minimum_win_probability: float = 0.45,
) -> dict[str, Any]:
    """Evaluate legacy positive-EV scorer in shadow — never executes."""
    from joker.objectives.scoring import ObjectiveStrategyScorer

    scorer = ObjectiveStrategyScorer(
        require_positive_expected_value=require_positive_expected_value,
        minimum_win_probability=minimum_win_probability,
    )
    scores = scorer.score_all(state, candidates, snapshot_id=snapshot_id)
    valid = [s for s in scores if s.valid and not s.is_no_trade]
    no_trade = next((s for s in scores if s.is_no_trade), None)
    best = None
    if valid:
        best = max(
            valid,
            key=lambda s: (
                s.expected_value_usd or Decimal("-10**9"),
                s.estimated_win_probability or Decimal("0"),
            ),
        )
    action = "enter" if best is not None else "wait"
    rejection = None
    if best is None:
        invalid = [s for s in scores if not s.valid and not s.is_no_trade]
        if invalid:
            rejection = list(invalid[0].invalidation_codes or ())
        else:
            rejection = ["no_valid_positive_ev_candidate"]
    return {
        "policy": "positive_ev_baseline",
        "shadow_only": True,
        "action": action,
        "strategy_id": str(best.strategy_id) if best and best.strategy_id else None,
        "quantity": None,  # baseline does not expand quantity grid here
        "expected_value_usd": str(best.expected_value_usd)
        if best and best.expected_value_usd is not None
        else None,
        "win_probability": str(best.estimated_win_probability)
        if best and best.estimated_win_probability is not None
        else None,
        "rejection_reason": rejection,
        "no_trade_ev": str(no_trade.expected_value_usd)
        if no_trade and no_trade.expected_value_usd is not None
        else "0",
        "executes": False,
    }

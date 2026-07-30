"""EV fail-closed, estimate builder, and feasibility input evidence tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from joker.cognition.schemas import (
    AgentRole,
    EntryPlan,
    ExecutionPlan,
    ExitPlan,
    InvalidationPlan,
    MarketDirection,
    StrategyHypothesis,
    StrategyLegCandidate,
)
from joker.objectives.estimate import StrategyEstimateBuilder
from joker.objectives.feasibility import FeasibilityInputs, GoalFeasibilityEngine
from joker.objectives.feasibility_inputs import build_feasibility_inputs_from_truth
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.sizing import DeterministicObjectiveSizer

ET = ZoneInfo("America/New_York")


def _state(**kw: object) -> SessionObjectiveState:
    base = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("100"),
        "target_ending_equity_usd": Decimal("600"),
        "working_order_reservation_usd": Decimal("0"),
        "filled_position_exposure_usd": Decimal("0"),
        "reserved_capital_usd": Decimal("0"),
        "available_capital_usd": Decimal("500"),
        "realised_pnl_usd": Decimal("0"),
        "unrealised_pnl_usd": Decimal("0"),
        "progress_to_goal_pct": Decimal("0"),
        "required_profit_remaining_usd": Decimal("100"),
        "time_remaining_seconds": 7200,
        "version": 1,
        "max_concurrent_positions": 1,
    }
    base.update(kw)
    return SessionObjectiveState.model_validate(base)


def _strategy() -> StrategyHypothesis:
    return StrategyHypothesis(
        name="test",
        market_thesis="t",
        direction=MarketDirection.BULLISH,
        candidate_legs=(
            StrategyLegCandidate(
                contract_id="SPY:2026-07-30:500:call",
                side="buy",
                option_type="call",
                strike=Decimal("500"),
                quantity=1,
                rationale="primary",
            ),
        ),
        entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
        execution_plan=ExecutionPlan(
            max_quote_age_seconds=5,
            partial_fill_policy="accept",
            replacement_policy="none",
        ),
        exit_plan=ExitPlan(stop_conditions=("stop",)),
        invalidation_plan=InvalidationPlan(conditions=("inv",)),
        expected_horizon_seconds=600,
        confidence=0.5,
        novelty_score=0.1,
        agent_role=AgentRole.BULLISH_INVENTOR,
        snapshot_id=uuid4(),
        session_id="s",
        cycle_id="c",
        prompt_version="t",
        model_call_id=uuid4(),
    )


def test_sizer_rejects_missing_ev() -> None:
    d = DeterministicObjectiveSizer(require_positive_expected_value=True).size(
        _state(),
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        expected_value_usd=None,
        estimated_win_probability=0.55,
    )
    assert not d.approved
    assert "ev_unavailable" in d.reason_codes


def test_sizer_rejects_negative_ev() -> None:
    d = DeterministicObjectiveSizer(require_positive_expected_value=True).size(
        _state(),
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        expected_value_usd=Decimal("-1"),
        estimated_win_probability=0.55,
    )
    assert not d.approved
    assert "ev_non_positive" in d.reason_codes


def test_estimate_builder_leaves_ev_none_without_samples() -> None:
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        comparable_episode_count=5,
    )
    assert est.expected_value_usd is None
    assert not est.valid
    assert "insufficient_calibrated_samples_for_ev" in est.uncertainty_reasons


def test_estimate_builder_uses_calibrated_samples() -> None:
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        comparable_episode_count=25,
        historical_avg_pnl_usd=Decimal("5.00"),
        historical_hit_rate=Decimal("0.55"),
    )
    assert est.expected_value_usd == Decimal("5.00")
    assert est.valid


def test_feasibility_stale_quotes_reduce() -> None:
    a = GoalFeasibilityEngine().assess(
        _state(),
        FeasibilityInputs(snapshot_id=uuid4(), quote_age_seconds=30.0),
    )
    assert a.classification == "low"
    assert "stale_quotes" in a.binding_constraints


def test_feasibility_wide_spreads_reduce() -> None:
    a = GoalFeasibilityEngine().assess(
        _state(),
        FeasibilityInputs(snapshot_id=uuid4(), typical_spread_pct=0.5),
    )
    assert a.classification == "low"
    assert "wide_spreads" in a.binding_constraints


def test_feasibility_no_affordable_contract_infeasible() -> None:
    a = GoalFeasibilityEngine().assess(
        _state(available_capital_usd=Decimal("10")),
        FeasibilityInputs(
            snapshot_id=uuid4(),
            median_premium_usd=Decimal("50"),
            valid_contract_count=0,
        ),
    )
    assert a.classification == "infeasible"


def test_feasibility_session_closed_infeasible() -> None:
    a = GoalFeasibilityEngine().assess(
        _state(),
        FeasibilityInputs(snapshot_id=uuid4(), session_phase="closed"),
    )
    assert a.classification == "infeasible"


def test_feasibility_insufficient_samples_keep_p_none() -> None:
    a = GoalFeasibilityEngine(minimum_samples_for_numeric_probability=20).assess(
        _state(),
        FeasibilityInputs(
            snapshot_id=uuid4(),
            comparable_outcome_samples=5,
            historical_hit_rate=Decimal("0.5"),
        ),
    )
    assert a.estimated_success_probability is None


def test_feasibility_adequate_samples_permit_numeric() -> None:
    a = GoalFeasibilityEngine(minimum_samples_for_numeric_probability=20).assess(
        _state(),
        FeasibilityInputs(
            snapshot_id=uuid4(),
            comparable_outcome_samples=25,
            historical_hit_rate=Decimal("0.52"),
        ),
    )
    assert a.estimated_success_probability == Decimal("0.5200")


def test_build_feasibility_inputs_not_snapshot_only() -> None:
    class _C:
        bid = Decimal("1.00")
        ask = Decimal("1.20")
        quote_age_seconds = 2.0
        implied_volatility = 0.2

    inputs = build_feasibility_inputs_from_truth(
        snapshot_id=uuid4(),
        option_surface_slice=(_C(),),
        available_capital_usd=Decimal("500"),
        session_phase="regular",
    )
    assert inputs.median_premium_usd is not None
    assert inputs.valid_contract_count == 1
    assert inputs.session_phase == "regular"
    assert inputs.typical_spread_pct is not None

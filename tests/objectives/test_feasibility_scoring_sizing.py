"""Feasibility, scoring, sizing, and context isolation tests."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.cognition.context import ContextAssembler
from joker.cognition.schemas import AgentRole
from joker.objectives.feasibility import FeasibilityInputs, GoalFeasibilityEngine
from joker.objectives.schemas import (
    OBJECTIVE_AWARE_ROLES,
    OBJECTIVE_NEUTRAL_ROLES,
    SessionObjectiveState,
)
from joker.objectives.scoring import ObjectiveStrategyScorer, StrategyScoreInput
from joker.objectives.sizing import DeterministicObjectiveSizer
from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot


def _state(**overrides: object) -> SessionObjectiveState:
    data = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500.00"),
        "target_profit_usd": Decimal("100.00"),
        "target_ending_equity_usd": Decimal("600.00"),
        "reserved_capital_usd": Decimal("0.00"),
        "available_capital_usd": Decimal("500.00"),
        "realised_pnl_usd": Decimal("0.00"),
        "unrealised_pnl_usd": Decimal("0.00"),
        "progress_to_goal_pct": Decimal("0.00"),
        "required_profit_remaining_usd": Decimal("100.00"),
        "time_remaining_seconds": 7200,
        "max_concurrent_positions": 1,
        "version": 1,
    }
    data.update(overrides)
    return SessionObjectiveState.model_validate(data)


def test_feasibility_infeasible_when_deadline_passed() -> None:
    eng = GoalFeasibilityEngine()
    a = eng.assess(
        _state(time_remaining_seconds=0),
        FeasibilityInputs(snapshot_id=uuid4()),
    )
    assert a.classification == "infeasible"
    assert a.estimated_success_probability is None


def test_feasibility_omits_probability_without_samples() -> None:
    eng = GoalFeasibilityEngine(minimum_samples_for_numeric_probability=20)
    a = eng.assess(_state(), FeasibilityInputs(snapshot_id=uuid4(), comparable_outcome_samples=2))
    assert a.estimated_success_probability is None
    assert "insufficient_samples_for_numeric_probability" in a.uncertainty_reasons


def test_feasibility_high_target_low_time() -> None:
    eng = GoalFeasibilityEngine()
    a = eng.assess(
        _state(
            required_profit_remaining_usd=Decimal("400.00"),
            time_remaining_seconds=600,
        ),
        FeasibilityInputs(snapshot_id=uuid4()),
    )
    assert a.classification in {"low", "infeasible"}


def test_scorer_rejects_negative_ev_and_scores_no_trade() -> None:
    scorer = ObjectiveStrategyScorer()
    state = _state()
    snap = uuid4()
    scores = scorer.score_all(
        state,
        [
            StrategyScoreInput(
                strategy_id=uuid4(),
                snapshot_id=snap,
                expected_value_usd=Decimal("-5"),
                capital_required_usd=50,
                maximum_loss_usd=50,
            )
        ],
        snapshot_id=snap,
        target_probability_before=Decimal("0.40"),
    )
    assert any(s.is_no_trade and s.valid for s in scores)
    trade = [s for s in scores if not s.is_no_trade][0]
    assert trade.valid is False
    assert "non_positive_expected_value" in trade.invalidation_codes


def test_sizer_rejects_non_positive_ev_and_caps_agent_qty() -> None:
    sizer = DeterministicObjectiveSizer()
    state = _state()
    rejected = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        expected_value_usd=Decimal("-1"),
        requested_quantity=5,
    )
    assert rejected.approved is False
    assert "ev_non_positive" in rejected.reason_codes

    ok = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("0.50"),
        expected_value_usd=Decimal("10"),
        estimated_win_probability=Decimal("0.55"),
        requested_quantity=100,
    )
    assert ok.approved is True
    assert ok.approved_quantity < 100
    assert ok.approved_notional_usd <= state.available_capital_usd


def test_sizer_no_martingale_on_prior_losses() -> None:
    sizer = DeterministicObjectiveSizer(prohibit_loss_multiplier=True)
    state = _state(available_capital_usd=Decimal("200.00"), authorised_capital_usd=Decimal("200.00"))
    a = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("0.50"),
        expected_value_usd=5,
        estimated_win_probability=0.55,
        prior_loss_count=0,
    )
    b = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("0.50"),
        expected_value_usd=5,
        estimated_win_probability=0.55,
        prior_loss_count=5,
    )
    assert a.approved and b.approved
    assert b.approved_quantity <= a.approved_quantity
    assert b.calculation_inputs.get("loss_multiplier_blocked") is True


def test_sizer_rejects_when_capital_below_one_premium() -> None:
    sizer = DeterministicObjectiveSizer()
    state = _state(
        authorised_capital_usd=Decimal("5.00"),
        available_capital_usd=Decimal("5.00"),
        target_ending_equity_usd=Decimal("6.00"),
        target_profit_usd=Decimal("1.00"),
    )
    d = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),  # $100 notional
        expected_value_usd=5,
        estimated_win_probability=0.6,
    )
    assert d.approved is False


def _snap() -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id=uuid4(),
        trading_date=datetime.now(timezone.utc).date(),
        exchange_time=datetime.now(timezone.utc),
        underlying=UnderlyingSnapshot(
            symbol="SPY",
            bid=Decimal("500"),
            ask=Decimal("500.1"),
            last=Decimal("500.05"),
            exchange_time=datetime.now(timezone.utc),
        ),
        data_quality_id=uuid4(),
    )


def test_perception_roles_do_not_receive_objective_context() -> None:
    assembler = ContextAssembler()
    objective = {
        "authorised_capital_usd": "500",
        "target_profit_usd": "100",
        "stance": "press",
    }
    for role_name in OBJECTIVE_NEUTRAL_ROLES:
        role = AgentRole(role_name)
        pkg = assembler.assemble(
            agent_role=role,
            session_id="s",
            cycle_id="c",
            snapshot=_snap(),
            objective_context=objective,
        )
        assert pkg.objective_context is None


def test_aware_roles_receive_objective_and_hash_changes() -> None:
    assembler = ContextAssembler()
    objective = {
        "authorised_capital_usd": "500.00",
        "available_capital_usd": "500.00",
        "reserved_capital_usd": "0.00",
        "realised_pnl_usd": "0.00",
        "target_profit_usd": "100.00",
        "required_profit_remaining_usd": "100.00",
        "progress_to_goal_pct": "0.00",
        "time_remaining_seconds": 3600,
        "feasibility_classification": "medium",
        "estimated_success_probability": None,
        "stance": "accumulate",
        "maximum_permitted_loss_usd": "500.00",
        "maximum_concurrent_positions": 1,
    }
    snap = _snap()
    with_obj = assembler.assemble(
        agent_role=AgentRole.META_DECISION,
        session_id="s",
        cycle_id="c",
        snapshot=snap,
        objective_context=objective,
    )
    without = assembler.assemble(
        agent_role=AgentRole.META_DECISION,
        session_id="s",
        cycle_id="c",
        snapshot=snap,
        objective_context=None,
    )
    assert with_obj.objective_context is not None
    assert without.objective_context is None
    assert with_obj.context_hash != without.context_hash
    assert AgentRole.META_DECISION.value in OBJECTIVE_AWARE_ROLES

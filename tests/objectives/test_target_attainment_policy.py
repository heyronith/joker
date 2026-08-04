"""Unit tests for the target-attainment objective policy.

Covers ``joker.objectives.target_attainment``: the authoritative
maximize-P(goal-by-deadline) decision policy used when
``objective.policy == "target_attainment"``. These tests never start a
paper session and never touch broker credentials — they exercise the
pure dataclasses and the ``TargetAttainmentPolicy.decide`` state machine
directly with fabricated inputs.
"""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from joker.objectives.target_attainment import (
    TargetAttainmentAction,
    TargetAttainmentCandidate,
    TargetAttainmentContext,
    TargetAttainmentPolicy,
    classify_physical_impossibility,
    max_affordable_quantity,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _ctx(**overrides: object) -> TargetAttainmentContext:
    """Build a TargetAttainmentContext with sane, physically-valid defaults."""
    data: dict[str, object] = {
        "objective_id": uuid4(),
        "snapshot_id": uuid4(),
        "authorised_capital_usd": Decimal("200.00"),
        "available_capital_usd": Decimal("200.00"),
        "reserved_capital_usd": Decimal("0.00"),
        "realised_pnl_usd": Decimal("0.00"),
        "unrealised_pnl_usd": Decimal("0.00"),
        "target_profit_usd": Decimal("300.00"),
        "remaining_goal_gap_usd": Decimal("300.00"),
        "time_remaining_seconds": 1800,
        "objective_duration_seconds": 3600,
        "elapsed_seconds": 1800,
        "open_position_count": 0,
        "working_order_count": 0,
        "max_concurrent_positions": 1,
        "maximum_authorised_contracts": 20,
        "allow_full_remaining_capital": True,
        "maximum_capital_fraction": 1.0,
        "minimum_calibrated_samples": 20,
        "exchange_session_phase": "regular",
        "session_similarity_bucket": "midday",
        "session_phase": "regular",
        "market_usable_for_execution": True,
        "option_surface_usable": True,
        "underlying_symbol": "SPY",
    }
    data.update(overrides)
    return TargetAttainmentContext(**data)  # type: ignore[arg-type]


def _candidate(**overrides: object) -> TargetAttainmentCandidate:
    """Build a TargetAttainmentCandidate with sane defaults."""
    data: dict[str, object] = {
        "strategy_id": uuid4(),
        "premium_per_contract_usd": Decimal("1.00"),
        "estimated_win_probability": Decimal("0.55"),
        "expected_value_usd": Decimal("5.00"),
        "estimated_payoff_ratio": Decimal("1.00"),
        "estimated_useful_upside_usd": Decimal("50.00"),
        "estimated_resolution_seconds": 600,
        "maximum_loss_usd_per_contract": Decimal("100.00"),
        "sample_count": 0,
        "historical_hit_rate": None,
    }
    data.update(overrides)
    return TargetAttainmentCandidate(**data)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# max_affordable_quantity / classify_physical_impossibility (pure helpers)
# ---------------------------------------------------------------------------


def test_max_affordable_quantity_limited_by_capital() -> None:
    qty = max_affordable_quantity(
        premium_per_contract_usd=Decimal("1.00"),
        available_capital_usd=Decimal("250.00"),
        maximum_authorised_contracts=20,
    )
    assert qty == 2  # $100/contract notional, $250 budget → floor(250/100)=2


def test_max_affordable_quantity_capped_by_authorised_contracts() -> None:
    qty = max_affordable_quantity(
        premium_per_contract_usd=Decimal("0.10"),
        available_capital_usd=Decimal("10000.00"),
        maximum_authorised_contracts=5,
    )
    assert qty == 5


def test_max_affordable_quantity_zero_premium_is_zero() -> None:
    assert (
        max_affordable_quantity(
            premium_per_contract_usd=Decimal("0"),
            available_capital_usd=Decimal("1000.00"),
            maximum_authorised_contracts=20,
        )
        == 0
    )


def test_max_affordable_quantity_respects_capital_fraction() -> None:
    qty = max_affordable_quantity(
        premium_per_contract_usd=Decimal("1.00"),
        available_capital_usd=Decimal("1000.00"),
        maximum_authorised_contracts=20,
        maximum_capital_fraction=0.5,
    )
    assert qty == 5  # budget = $500, cost/contract=$100 → 5


def test_classify_physical_impossibility_deadline_passed() -> None:
    impossible, codes = classify_physical_impossibility(_ctx(time_remaining_seconds=0))
    assert impossible is True
    assert "deadline_passed" in codes


def test_classify_physical_impossibility_wrong_underlying() -> None:
    impossible, codes = classify_physical_impossibility(_ctx(underlying_symbol="QQQ"))
    assert impossible is True
    assert "wrong_underlying" in codes


def test_classify_physical_impossibility_clean_context_is_possible() -> None:
    impossible, codes = classify_physical_impossibility(_ctx())
    assert impossible is False
    assert codes == []


# ---------------------------------------------------------------------------
# Core decision behavior
# ---------------------------------------------------------------------------


def test_selects_lower_win_rate_candidate_when_only_it_can_reach_goal() -> None:
    """A lower win-rate candidate that can close the goal gap must beat a
    higher win-rate candidate whose max affordable size cannot close it."""
    ctx = _ctx(
        authorised_capital_usd=Decimal("200.00"),
        available_capital_usd=Decimal("200.00"),
        remaining_goal_gap_usd=Decimal("300.00"),
        time_remaining_seconds=1800,
        objective_duration_seconds=3600,
    )
    # A: high win probability but too little upside per dollar to ever close
    # the $300 gap given the $200 capital ceiling (max 2 contracts × $50 = $100).
    cand_a = _candidate(
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),  # $100/contract
        estimated_win_probability=Decimal("0.70"),
        estimated_useful_upside_usd=Decimal("50.00"),
        maximum_loss_usd_per_contract=Decimal("100.00"),
    )
    # B: lower win probability but bigger upside per contract; 3 affordable
    # contracts (× $100 useful upside each) close the $300 gap.
    cand_b = _candidate(
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("0.50"),  # $50/contract
        estimated_win_probability=Decimal("0.40"),
        estimated_useful_upside_usd=Decimal("100.00"),
        maximum_loss_usd_per_contract=Decimal("50.00"),
    )
    decision = TargetAttainmentPolicy().decide(ctx, [cand_a, cand_b])

    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_strategy_id == cand_b.strategy_id
    assert decision.selected_quantity == 3
    assert decision.feasibility == "attainable"


def test_rejects_positive_ev_candidate_that_cannot_materially_advance_goal() -> None:
    """A cheap, high win-rate, positive-EV candidate that can only nudge the
    goal by a trivial amount must lose to waiting when waiting scores higher."""
    ctx = _ctx(
        authorised_capital_usd=Decimal("1000.00"),
        available_capital_usd=Decimal("10.00"),  # only 1 contract affordable
        remaining_goal_gap_usd=Decimal("1000.00"),
        time_remaining_seconds=3600,
        objective_duration_seconds=3600,
    )
    tiny_ev_candidate = _candidate(
        premium_per_contract_usd=Decimal("0.10"),  # $10/contract
        estimated_win_probability=Decimal("0.90"),
        expected_value_usd=Decimal("3.00"),  # positive EV
        estimated_useful_upside_usd=Decimal("5.00"),  # 0.5% of the $1000 gap
        maximum_loss_usd_per_contract=Decimal("10.00"),
    )
    decision = TargetAttainmentPolicy().decide(ctx, [tiny_ev_candidate])

    assert decision.action == TargetAttainmentAction.WAIT
    assert decision.no_trade is not None
    assert decision.no_trade.selected is True
    assert "wait_has_higher_or_equal_target_hit_probability" in decision.reason_codes


def test_compares_all_affordable_quantities() -> None:
    """Every affordable quantity 1..max_q for a candidate is evaluated, with
    capital and upside scaling linearly by quantity."""
    ctx = _ctx(available_capital_usd=Decimal("500.00"), maximum_authorised_contracts=20)
    cand = _candidate(
        premium_per_contract_usd=Decimal("1.00"),  # $100/contract, max 5
        estimated_useful_upside_usd=Decimal("40.00"),
    )
    decision = TargetAttainmentPolicy().decide(ctx, [cand])
    evals = [e for e in decision.quantity_evaluations if e.strategy_id == cand.strategy_id]
    quantities = sorted(e.quantity for e in evals)
    assert quantities == [1, 2, 3, 4, 5]
    by_qty = {e.quantity: e for e in evals}
    for q in quantities:
        assert by_qty[q].capital_required_usd == Decimal("100.00") * q
        assert by_qty[q].useful_upside_usd == Decimal("40.00") * q


def test_may_select_full_remaining_capital() -> None:
    """When probability keeps improving with quantity (material progress that
    never fully closes the gap) and waiting decays faster, the policy may
    select the maximum affordable quantity."""
    ctx = _ctx(
        authorised_capital_usd=Decimal("500.00"),
        available_capital_usd=Decimal("500.00"),
        remaining_goal_gap_usd=Decimal("1000.00"),  # unreachable at any affordable size
        time_remaining_seconds=600,  # urgency suppresses wait-value below action's
        objective_duration_seconds=3600,
        maximum_authorised_contracts=20,
    )
    cand = _candidate(
        premium_per_contract_usd=Decimal("1.00"),  # $100/contract, max 5
        estimated_win_probability=Decimal("0.60"),
        estimated_useful_upside_usd=Decimal("100.00"),  # grows w/ qty, never closes
    )
    max_q = max_affordable_quantity(
        premium_per_contract_usd=cand.premium_per_contract_usd,
        available_capital_usd=ctx.available_capital_usd,
        maximum_authorised_contracts=ctx.maximum_authorised_contracts,
    )
    decision = TargetAttainmentPolicy().decide(ctx, [cand])
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_quantity == max_q
    assert decision.selected_capital_usd == ctx.available_capital_usd


def test_does_not_always_select_full_capital() -> None:
    """Once a small quantity already closes the goal gap, buying more
    contracts adds no probability benefit — the cheaper quantity wins the
    capital-efficiency tie-break instead of consuming full capital."""
    ctx = _ctx(
        available_capital_usd=Decimal("500.00"),
        remaining_goal_gap_usd=Decimal("80.00"),
        maximum_authorised_contracts=20,
    )
    cand = _candidate(
        premium_per_contract_usd=Decimal("1.00"),  # $100/contract, max 5 affordable
        estimated_win_probability=Decimal("0.60"),
        estimated_useful_upside_usd=Decimal("50.00"),  # 2 contracts (=$100) close $80 gap
    )
    max_q = max_affordable_quantity(
        premium_per_contract_usd=cand.premium_per_contract_usd,
        available_capital_usd=ctx.available_capital_usd,
        maximum_authorised_contracts=ctx.maximum_authorised_contracts,
    )
    decision = TargetAttainmentPolicy().decide(ctx, [cand])
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.selected_quantity == 2
    assert decision.selected_quantity < max_q


def test_no_trade_wins_when_waiting_has_higher_target_probability() -> None:
    ctx = _ctx(
        authorised_capital_usd=Decimal("1000.00"),
        available_capital_usd=Decimal("1000.00"),
        remaining_goal_gap_usd=Decimal("1000.00"),
        time_remaining_seconds=3600,
        objective_duration_seconds=3600,  # frac_left=1.0 → strong wait value
    )
    weak_candidate = _candidate(
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.50"),
        estimated_useful_upside_usd=Decimal("30.00"),  # immaterial vs $1000 gap
        maximum_loss_usd_per_contract=Decimal("100.00"),
    )
    decision = TargetAttainmentPolicy().decide(ctx, [weak_candidate])
    assert decision.action == TargetAttainmentAction.WAIT
    assert decision.no_trade is not None and decision.no_trade.selected is True
    assert decision.no_trade_p_goal is not None
    assert decision.selected_p_goal == decision.no_trade_p_goal


def test_no_trade_loses_when_deadline_cost_makes_action_superior() -> None:
    """As the deadline approaches, the ordinal wait-value decays; a modest
    candidate that would have lost to waiting earlier in the session now wins."""
    ctx = _ctx(
        authorised_capital_usd=Decimal("200.00"),
        available_capital_usd=Decimal("200.00"),
        remaining_goal_gap_usd=Decimal("100.00"),
        time_remaining_seconds=60,  # very little time left
        objective_duration_seconds=3600,
    )
    candidate = _candidate(
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.50"),
        estimated_useful_upside_usd=Decimal("30.00"),  # material (>=25% of $100 gap)
        estimated_resolution_seconds=30,
    )
    decision = TargetAttainmentPolicy().decide(ctx, [candidate])
    assert decision.action == TargetAttainmentAction.ENTER
    assert decision.feasibility == "low_probability"
    assert "material_progress_only" in decision.reason_codes


def test_low_probability_does_not_block_valid_candidate() -> None:
    """feasibility == 'low_probability' is an evidentiary label, not a veto —
    the action must still be ENTER when the candidate beats no-trade."""
    ctx = _ctx(
        available_capital_usd=Decimal("200.00"),
        remaining_goal_gap_usd=Decimal("100.00"),
        time_remaining_seconds=60,
        objective_duration_seconds=3600,
    )
    candidate = _candidate(
        premium_per_contract_usd=Decimal("1.00"),
        estimated_win_probability=Decimal("0.50"),
        estimated_useful_upside_usd=Decimal("30.00"),
        estimated_resolution_seconds=30,
    )
    decision = TargetAttainmentPolicy().decide(ctx, [candidate])
    assert decision.feasibility == "low_probability"
    assert decision.action == TargetAttainmentAction.ENTER


def test_physical_impossibility_blocks_entry() -> None:
    from joker.objectives.session_eligibility import resolve_objective_session_state
    from joker.time.calendar import MarketCalendar
    from joker.time.clock import FrozenExchangeClock
    from datetime import datetime
    from zoneinfo import ZoneInfo

    clock = FrozenExchangeClock(
        datetime(2026, 8, 4, 20, 0, tzinfo=ZoneInfo("America/New_York")),
        calendar=MarketCalendar(),
    )
    session = resolve_objective_session_state(clock=clock, similarity_bucket="closed")
    ctx = _ctx(exchange_session_phase="closed", session_phase="closed")
    candidate = _candidate(estimated_useful_upside_usd=Decimal("1000.00"))
    decision = TargetAttainmentPolicy().decide(ctx, [candidate], session_state=session)
    assert decision.action == TargetAttainmentAction.BLOCK
    assert decision.feasibility == "physically_impossible"
    assert "market_not_regular" in decision.reason_codes
    assert decision.selected_strategy_id is None
    assert decision.selected_quantity == 0


def test_deadline_blocks_new_entries() -> None:
    ctx = _ctx(time_remaining_seconds=0)
    candidate = _candidate(estimated_useful_upside_usd=Decimal("1000.00"))
    decision = TargetAttainmentPolicy().decide(ctx, [candidate])
    assert decision.action == TargetAttainmentAction.BLOCK
    assert decision.feasibility == "physically_impossible"
    assert "deadline_passed" in decision.reason_codes


def test_goal_achieved_blocks_new_entries() -> None:
    ctx = _ctx(remaining_goal_gap_usd=Decimal("0.00"))
    candidate = _candidate(estimated_useful_upside_usd=Decimal("1000.00"))
    decision = TargetAttainmentPolicy().decide(ctx, [candidate])
    assert decision.action == TargetAttainmentAction.WAIT
    assert decision.reason_codes == ["goal_already_achieved"]
    assert decision.no_trade is not None
    assert decision.no_trade.selected is True
    assert decision.no_trade.p_goal.p_goal == Decimal("1.0000")


def test_goal_achieved_blocks_new_entries_when_gap_negative() -> None:
    """Over-achieving the goal (negative remaining gap) also blocks new entries."""
    ctx = _ctx(remaining_goal_gap_usd=Decimal("-50.00"))
    decision = TargetAttainmentPolicy().decide(ctx, [])
    assert decision.action == TargetAttainmentAction.WAIT
    assert "goal_already_achieved" in decision.reason_codes


def test_deadline_urgency_does_not_apply_arbitrary_dampening() -> None:
    """The policy must not shrink the affordable-quantity grid merely because
    little time remains — capital constraints (not urgency) size the grid."""
    cand = _candidate(
        premium_per_contract_usd=Decimal("1.00"),
        estimated_resolution_seconds=30,  # fits comfortably in both windows
        estimated_useful_upside_usd=Decimal("40.00"),
    )
    ctx_plenty_time = _ctx(
        available_capital_usd=Decimal("500.00"),
        time_remaining_seconds=3600,
        objective_duration_seconds=3600,
    )
    ctx_little_time = _ctx(
        available_capital_usd=Decimal("500.00"),
        time_remaining_seconds=120,
        objective_duration_seconds=3600,
    )
    decision_plenty = TargetAttainmentPolicy().decide(ctx_plenty_time, [cand])
    decision_little = TargetAttainmentPolicy().decide(ctx_little_time, [cand])

    quantities_plenty = sorted(
        e.quantity for e in decision_plenty.quantity_evaluations if not e.physically_impossible
    )
    quantities_little = sorted(
        e.quantity for e in decision_little.quantity_evaluations if not e.physically_impossible
    )
    # Same capital, same premium → identical affordable-quantity grid regardless
    # of urgency. Only the probability estimates (not the size grid) may differ.
    assert quantities_plenty == quantities_little == [1, 2, 3, 4, 5]
    for q in quantities_plenty:
        cap_plenty = next(
            e.capital_required_usd
            for e in decision_plenty.quantity_evaluations
            if e.quantity == q
        )
        cap_little = next(
            e.capital_required_usd
            for e in decision_little.quantity_evaluations
            if e.quantity == q
        )
        assert cap_plenty == cap_little


def test_loss_does_not_apply_martingale_multiplier() -> None:
    """TargetAttainmentContext carries no consecutive-loss / martingale field,
    and decide() is a pure function of (ctx, candidates) — repeated calls
    after a simulated loss cannot inflate size."""
    assert not hasattr(TargetAttainmentContext, "prior_loss_count")
    assert not hasattr(TargetAttainmentContext, "consecutive_losses")
    assert not hasattr(TargetAttainmentContext, "loss_multiplier")

    ctx = _ctx(available_capital_usd=Decimal("300.00"))
    cand = _candidate(
        premium_per_contract_usd=Decimal("1.00"),
        estimated_useful_upside_usd=Decimal("40.00"),
    )
    policy = TargetAttainmentPolicy()
    first = policy.decide(ctx, [cand])
    # Simulate "after a loss" — nothing in the API allows expressing that,
    # so calling decide again with the identical context must be deterministic.
    second = policy.decide(ctx, [cand])
    assert first.selected_quantity == second.selected_quantity
    assert first.selected_capital_usd == second.selected_capital_usd


def test_no_exact_contract_candidate_returns_wait() -> None:
    ctx = _ctx()
    decision = TargetAttainmentPolicy().decide(ctx, [])
    assert decision.action == TargetAttainmentAction.WAIT
    assert "no_valid_contract_candidates" in decision.reason_codes


def test_candidate_exceeding_capital_is_physically_impossible_not_selected() -> None:
    ctx = _ctx(available_capital_usd=Decimal("10.00"))
    too_expensive = _candidate(premium_per_contract_usd=Decimal("5.00"))  # $500/contract
    decision = TargetAttainmentPolicy().decide(ctx, [too_expensive])
    evals = [e for e in decision.quantity_evaluations if e.strategy_id == too_expensive.strategy_id]
    assert all(e.physically_impossible for e in evals)
    assert decision.selected_strategy_id != too_expensive.strategy_id


def test_candidate_resolving_after_deadline_is_physically_impossible() -> None:
    ctx = _ctx(available_capital_usd=Decimal("500.00"), time_remaining_seconds=60)
    slow_candidate = _candidate(
        premium_per_contract_usd=Decimal("1.00"),
        estimated_resolution_seconds=600,  # resolves after the 60s deadline
        estimated_useful_upside_usd=Decimal("1000.00"),
    )
    decision = TargetAttainmentPolicy().decide(ctx, [slow_candidate])
    evals = [e for e in decision.quantity_evaluations if e.strategy_id == slow_candidate.strategy_id]
    assert evals
    assert all(e.physically_impossible for e in evals)
    assert all(
        "resolution_after_deadline" in e.reason_codes for e in evals
    )
    assert decision.selected_strategy_id != slow_candidate.strategy_id

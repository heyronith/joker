"""Capital budget and allocation tests."""

from __future__ import annotations

from joker.risk.capital import CapitalBudget, CapitalPlan, CapitalError
from joker.risk.governor import RiskGovernor, RiskReasonCode
from joker.app.safety import SafetyMode
from joker.schemas.domain import RiskConfig
from tests.fixtures.domain import make_candidate, make_daily_state


def test_allocate_respects_authorized_ceiling() -> None:
    budget = CapitalBudget(
        plan=CapitalPlan(
            authorized_usd=500.0,
            target_profit_pct=20.0,
            max_contracts_per_trade=50,
            aggression_mode="fixed",
            max_kelly_fraction=1.0,
        )
    )
    # $2.00 premium => $200 per contract
    result = budget.allocate(
        premium_per_contract=2.0,
        capital_fraction=1.0,
        confidence=1.0,
    )
    assert result.quantity == 2  # 2 * 200 = 400 <= 500
    assert result.notional_usd == 400.0


def test_allocate_split_leaves_powder() -> None:
    budget = CapitalBudget(plan=CapitalPlan(authorized_usd=1000.0, max_contracts_per_trade=50))
    result = budget.allocate(
        premium_per_contract=1.0,  # $100/contract
        allocation_style="split",
        confidence=0.5,
    )
    assert 1 <= result.quantity < 10
    assert result.notional_usd <= 1000.0


def test_reserve_and_release_updates_available() -> None:
    budget = CapitalBudget(plan=CapitalPlan(authorized_usd=500.0))
    alloc = budget.allocate(premium_per_contract=1.0, capital_fraction=0.4, confidence=0.8)
    budget.reserve(alloc.notional_usd)
    assert budget.available_usd == 500.0 - alloc.notional_usd
    budget.release(alloc.notional_usd, realized_pnl_usd=50.0)
    assert budget.available_usd == 500.0
    assert budget.realized_pnl_usd == 50.0
    assert budget.progress_to_goal_pct == 50.0  # target is 20% of 500 = 100


def test_goal_met() -> None:
    budget = CapitalBudget(plan=CapitalPlan(authorized_usd=100.0, target_profit_pct=20.0))
    budget.realized_pnl_usd = 20.0
    assert budget.goal_met is True


def test_risk_governor_capital_exceeded() -> None:
    cfg = RiskConfig(
        max_daily_loss_usd=500,
        max_trades_per_day=5,
        max_open_positions=1,
        max_premium_usd=500,
        max_spread_pct=50,
        quote_max_age_seconds=9999,
        policy="agent_led",
        authorized_capital_usd=100.0,
        reserved_capital_usd=0.0,
    )
    gov = RiskGovernor(cfg, SafetyMode.PAPER)
    # 2 contracts * $1.05 * 100 = $210 > $100
    c = make_candidate(entry_limit_price=1.05, quantity=2)
    decision = gov.evaluate(c, make_daily_state())
    assert decision.approved is False
    assert RiskReasonCode.CAPITAL_EXCEEDED in decision.reason_codes


def test_invalid_plan_raises() -> None:
    try:
        CapitalPlan(authorized_usd=0)
        assert False, "expected CapitalError"
    except CapitalError:
        pass

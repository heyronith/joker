"""Shared underlying scenario grid coherence and fail-closed joining."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from joker.objectives.contract_outcomes import estimate_contract_outcome
from joker.objectives.full_chain_universe import FullChainContract, MoneynessBucket
from joker.objectives.shared_scenarios import build_shared_underlying_scenario_grid


NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
TRADING_DATE = date(2026, 8, 5)


def _strategy(*, direction: str, confidence: float = 0.7, sid=None):
    return SimpleNamespace(
        strategy_id=sid or uuid4(),
        strategy_family="directional",
        direction=SimpleNamespace(value=direction),
        confidence=confidence,
        expected_horizon_seconds=300,
    )


def _contract(
    *,
    strike: str = "500",
    ask: str = "1.00",
    iv: str = "0.30",
    delta: str = "0.50",
) -> FullChainContract:
    ask_d = Decimal(ask)
    bid_d = ask_d - Decimal("0.02")
    mid = (ask_d + bid_d) / Decimal("2")
    return FullChainContract(
        surface_id=uuid4(),
        contract_id=f"SPY:{TRADING_DATE.isoformat()}:{strike}:call",
        symbol="SPY",
        expiration=TRADING_DATE,
        option_type="call",
        strike=Decimal(strike),
        underlying_price=Decimal("500"),
        distance_from_spot=Decimal(strike) - Decimal("500"),
        distance_from_spot_pct=Decimal("0"),
        moneyness_bucket=MoneynessBucket.ATM,
        premium_bucket="premium:1.00",
        delta_bucket="delta:0.50",
        bid=bid_d,
        ask=ask_d,
        mid=mid,
        relative_spread=(ask_d - bid_d) / mid,
        quote_timestamp=NOW,
        quote_age_seconds=Decimal("0"),
        evaluated_at_exchange_time=NOW,
        liquidity_score=0.8,
        delta=Decimal(delta),
        gamma=Decimal("0.02"),
        theta=Decimal("-0.05"),
        implied_volatility=Decimal(iv),
    )


def test_shared_portfolio_scenarios_use_identical_underlying_prices() -> None:
    grid = build_shared_underlying_scenario_grid(
        strategies=[_strategy(direction="bullish"), _strategy(direction="bullish")],
        reference_underlying_price=Decimal("500"),
        evaluation_time=NOW,
        horizon_seconds=300,
    )
    prices = [scenario.underlying_price for scenario in grid.scenarios]
    low = estimate_contract_outcome(
        strategy=_strategy(direction="bullish"),
        contract=_contract(strike="501", iv="0.20"),
        remaining_goal_gap_usd=Decimal("10"),
        time_remaining_seconds=300,
        shared_scenario_grid=grid,
    )
    high = estimate_contract_outcome(
        strategy=_strategy(direction="bullish"),
        contract=_contract(strike="505", iv="0.90"),
        remaining_goal_gap_usd=Decimal("10"),
        time_remaining_seconds=300,
        shared_scenario_grid=grid,
    )
    assert [s.underlying_price for s in low.scenarios] == prices
    assert [s.underlying_price for s in high.scenarios] == prices
    assert low.shared_scenario_grid_hash == high.shared_scenario_grid_hash == grid.grid_hash


def test_different_iv_contracts_are_not_joined_by_label_only() -> None:
    """Same scenario_id labels do not imply equal option responses across contracts."""
    grid = build_shared_underlying_scenario_grid(
        strategies=[_strategy(direction="bullish")],
        reference_underlying_price=Decimal("500"),
        evaluation_time=NOW,
        horizon_seconds=300,
    )
    low_delta = estimate_contract_outcome(
        strategy=_strategy(direction="bullish"),
        contract=_contract(strike="510", iv="0.15", delta="0.20", ask="0.40"),
        remaining_goal_gap_usd=Decimal("20"),
        time_remaining_seconds=300,
        shared_scenario_grid=grid,
    )
    high_delta = estimate_contract_outcome(
        strategy=_strategy(direction="bullish"),
        contract=_contract(strike="495", iv="1.20", delta="0.80", ask="2.00"),
        remaining_goal_gap_usd=Decimal("20"),
        time_remaining_seconds=300,
        shared_scenario_grid=grid,
    )
    assert [s.scenario_id for s in low_delta.scenarios] == [
        s.scenario_id for s in high_delta.scenarios
    ]
    assert [s.underlying_price for s in low_delta.scenarios] == [
        s.underlying_price for s in high_delta.scenarios
    ]
    # Shared spots only — contract response (price/PnL) remains contract-specific.
    assert any(
        a.estimated_option_price != b.estimated_option_price
        or a.pnl_per_contract_usd != b.pnl_per_contract_usd
        for a, b in zip(low_delta.scenarios, high_delta.scenarios, strict=True)
    )


def test_shared_scenario_probabilities_are_not_averaged_from_contracts() -> None:
    grid = build_shared_underlying_scenario_grid(
        strategies=[_strategy(direction="bullish", confidence=0.9)],
        reference_underlying_price=Decimal("500"),
        evaluation_time=NOW,
        horizon_seconds=300,
    )
    grid_probs = [s.probability for s in grid.scenarios]
    outcome = estimate_contract_outcome(
        strategy=_strategy(direction="bullish", confidence=0.9),
        contract=_contract(),
        remaining_goal_gap_usd=Decimal("10"),
        time_remaining_seconds=300,
        shared_scenario_grid=grid,
    )
    assert [s.probability for s in outcome.scenarios] == grid_probs
    # Fabricating a contract-level average would not match the shared grid.
    fake_avg = (Decimal("0.2") + Decimal("0.8")) / Decimal("2")
    assert fake_avg not in grid_probs

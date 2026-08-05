from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from joker.objectives.contract_outcomes import estimate_contract_outcome
from joker.objectives.full_chain_universe import (
    FullChainContract,
    MoneynessBucket,
)


def _strategy(*, horizon: int = 600):
    return SimpleNamespace(
        strategy_id=uuid4(),
        strategy_family="directional_momentum",
        direction=SimpleNamespace(value="bullish"),
        expected_horizon_seconds=horizon,
        confidence=0.65,
        supporting_evidence_ids=(uuid4(),),
    )


def _contract(
    *,
    strike: str,
    ask: str,
    bid: str,
    delta: str | None,
    gamma: str | None = "0.02",
    theta: str | None = "-0.10",
    iv: str | None = "0.30",
) -> FullChainContract:
    strike_d = Decimal(strike)
    bid_d = Decimal(bid)
    ask_d = Decimal(ask)
    mid = (bid_d + ask_d) / Decimal("2")
    distance = strike_d - Decimal("500")
    return FullChainContract(
        surface_id=uuid4(),
        contract_id=f"SPY:2026-08-05:{strike}:call",
        symbol="SPY",
        expiration=date(2026, 8, 5),
        option_type="call",
        strike=strike_d,
        underlying_price=Decimal("500"),
        distance_from_spot=distance,
        distance_from_spot_pct=distance / Decimal("5"),
        moneyness_bucket=(
            MoneynessBucket.ATM
            if abs(distance) <= Decimal("2.5")
            else MoneynessBucket.OTM
        ),
        premium_bucket="test",
        delta_bucket="test",
        bid=bid_d,
        ask=ask_d,
        mid=mid,
        relative_spread=(ask_d - bid_d) / mid,
        quote_timestamp=datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc),
        quote_age_seconds=Decimal("0"),
        liquidity_score=0.8,
        delta=Decimal(delta) if delta is not None else None,
        gamma=Decimal(gamma) if gamma is not None else None,
        theta=Decimal(theta) if theta is not None else None,
        implied_volatility=Decimal(iv) if iv is not None else None,
    )


def test_different_strikes_get_contract_specific_probabilities() -> None:
    strategy = _strategy()
    near = estimate_contract_outcome(
        strategy=strategy,
        contract=_contract(strike="500", ask="1.00", bid="0.95", delta="0.52"),
        remaining_goal_gap_usd=Decimal("50"),
        time_remaining_seconds=1200,
    )
    far = estimate_contract_outcome(
        strategy=strategy,
        contract=_contract(strike="510", ask="0.20", bid="0.18", delta="0.10"),
        remaining_goal_gap_usd=Decimal("50"),
        time_remaining_seconds=1200,
    )
    assert near.probability_closes_goal_gap != far.probability_closes_goal_gap
    assert near.required_contract_price != far.required_contract_price
    assert near.estimated_useful_upside_usd != far.estimated_useful_upside_usd


def test_missing_greeks_uses_labeled_conservative_fallback() -> None:
    estimate = estimate_contract_outcome(
        strategy=_strategy(),
        contract=_contract(
            strike="505",
            ask="0.50",
            bid="0.45",
            delta=None,
            gamma=None,
            theta=None,
            iv=None,
        ),
        remaining_goal_gap_usd=Decimal("50"),
        time_remaining_seconds=900,
    )
    assert estimate.estimate_type == "quote_intrinsic_fallback"
    assert "missing_greeks_or_implied_volatility" in estimate.uncertainty_reasons
    assert estimate.historical_sample_count == 0
    assert estimate.usable_for_ranking is True


def test_indefensible_estimate_fails_closed() -> None:
    estimate = estimate_contract_outcome(
        strategy=_strategy(),
        contract=_contract(strike="500", ask="1.00", bid="0.95", delta="0.5"),
        remaining_goal_gap_usd=Decimal("50"),
        time_remaining_seconds=0,
    )
    assert estimate.estimate_type == "unknown"
    assert estimate.probability_closes_goal_gap is None
    assert estimate.usable_for_ranking is False


def test_spread_required_move_and_remaining_time_change_estimate() -> None:
    strategy = _strategy(horizon=1200)
    tight = estimate_contract_outcome(
        strategy=strategy,
        contract=_contract(strike="500", ask="1.00", bid="0.98", delta="0.5"),
        remaining_goal_gap_usd=Decimal("40"),
        time_remaining_seconds=1200,
    )
    wide = estimate_contract_outcome(
        strategy=strategy,
        contract=_contract(strike="500", ask="1.00", bid="0.70", delta="0.5"),
        remaining_goal_gap_usd=Decimal("100"),
        time_remaining_seconds=300,
    )
    assert tight.lower_probability_bound != wide.lower_probability_bound
    assert tight.required_contract_price < wide.required_contract_price
    assert tight.estimated_resolution_seconds > wide.estimated_resolution_seconds


def test_no_fabricated_samples_or_evidence() -> None:
    evidence_id = uuid4()
    strategy = _strategy()
    strategy.supporting_evidence_ids = (evidence_id,)
    estimate = estimate_contract_outcome(
        strategy=strategy,
        contract=_contract(strike="500", ask="1.00", bid="0.95", delta="0.5"),
        remaining_goal_gap_usd=Decimal("50"),
        time_remaining_seconds=600,
        evidence_ids=(evidence_id,),
    )
    assert estimate.historical_sample_count == 0
    assert estimate.evidence_ids == (evidence_id,)

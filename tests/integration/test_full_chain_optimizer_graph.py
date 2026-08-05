from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

from joker.graph.cognitive_state import CognitiveGraphState
from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.objectives.config import FullChainOptimizerSettings
from joker.objectives.full_chain_optimizer import optimize_full_chain
from joker.objectives.portfolio_search import PortfolioAction
from joker.objectives.target_attainment import TargetAttainmentContext

TRADING_DATE = date(2026, 8, 5)
NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)


def _strategy(*, expensive_hint: bool = True):
    return SimpleNamespace(
        strategy_id=uuid4(),
        name="bullish continuation",
        strategy_family="directional_momentum",
        direction=SimpleNamespace(value="bullish"),
        expected_horizon_seconds=300,
        confidence=0.75,
        supporting_evidence_ids=(uuid4(),),
        contract_selection_preferences=(),
        candidate_legs=(
            (
                SimpleNamespace(
                    contract_id="SPY:2026-08-05:500:call",
                    quantity=1,
                    limit_price=Decimal("2.25"),
                ),
            )
            if expensive_hint
            else ()
        ),
    )


def _contract(
    strike: str,
    ask: str,
    *,
    bid: str,
    option_type: str = "call",
) -> OptionContractSnapshot:
    ask_d = Decimal(ask)
    bid_d = Decimal(bid)
    mid = (ask_d + bid_d) / Decimal("2")
    return OptionContractSnapshot(
        contract_id=f"SPY:{TRADING_DATE.isoformat()}:{strike}:{option_type}",
        symbol="SPY",
        expiry=TRADING_DATE,
        strike=Decimal(strike),
        option_type=option_type,  # type: ignore[arg-type]
        bid=bid_d,
        ask=ask_d,
        mid=mid,
        implied_volatility=Decimal("0.80"),
        delta=Decimal("0.70") if option_type == "call" else Decimal("-0.70"),
        gamma=Decimal("0.03"),
        theta=Decimal("-0.05"),
        quote_timestamp=NOW,
        quote_age_ms=0,
        relative_spread=(ask_d - bid_d) / mid,
        liquidity_score=0.8,
    )


def _surface(contracts) -> OptionSurfaceSnapshot:
    return OptionSurfaceSnapshot(
        surface_id=uuid4(),
        exchange_time=NOW,
        trading_date=TRADING_DATE,
        underlying_symbol="SPY",
        underlying_price=Decimal("500"),
        contracts=tuple(contracts),
    )


def _ctx(*, snapshot_id=None) -> TargetAttainmentContext:
    return TargetAttainmentContext(
        objective_id=uuid4(),
        snapshot_id=snapshot_id or uuid4(),
        authorised_capital_usd=Decimal("200"),
        available_capital_usd=Decimal("200"),
        reserved_capital_usd=Decimal("0"),
        realised_pnl_usd=Decimal("0"),
        unrealised_pnl_usd=Decimal("0"),
        target_profit_usd=Decimal("20"),
        remaining_goal_gap_usd=Decimal("10"),
        time_remaining_seconds=300,
        objective_duration_seconds=3600,
        elapsed_seconds=3300,
        open_position_count=0,
        working_order_count=0,
        max_concurrent_positions=1,
        maximum_authorised_contracts=20,
        exchange_session_phase="regular",
        objective_version=4,
    )


def test_expensive_agent_leg_does_not_block_affordable_full_chain_contract() -> None:
    snapshot_id = uuid4()
    result = optimize_full_chain(
        strategies=[_strategy(expensive_hint=True)],
        surface=_surface(
            [
                _contract("500", "2.25", bid="2.20"),
                _contract("501", "0.10", bid="0.09"),
                _contract("502", "0.20", bid="0.18"),
            ]
        ),
        ctx=_ctx(snapshot_id=snapshot_id),
        settings=FullChainOptimizerSettings(
            enabled=True,
            minimum_probability_improvement_over_wait=0,
        ),
        maximum_authorised_contracts=20,
    )
    evaluated = {row.contract_id for row in result.decision.quantity_grid}
    assert "SPY:2026-08-05:501:call" in evaluated
    assert "SPY:2026-08-05:502:call" in evaluated
    assert "SPY:2026-08-05:500:call" not in evaluated
    assert result.decision.action in {PortfolioAction.ENTER, PortfolioAction.WAIT}


def test_no_valid_contract_returns_wait_without_exception() -> None:
    result = optimize_full_chain(
        strategies=[_strategy()],
        surface=_surface([_contract("500", "1.00", bid="0.10")]),
        ctx=_ctx(),
        settings=FullChainOptimizerSettings(enabled=True),
        maximum_authorised_contracts=20,
    )
    assert result.decision.action == PortfolioAction.WAIT
    assert "no_valid_contract_candidates" in result.decision.reason_codes


def test_portfolio_authority_channels_are_declared_for_checkpoints() -> None:
    annotations = CognitiveGraphState.__annotations__
    for field in (
        "_full_chain_universe",
        "_contract_outcomes",
        "_quantity_grid",
        "_portfolio_grid",
        "_target_portfolio_decision",
        "_target_authorized_positions",
        "_execution_command_ids",
    ):
        assert field in annotations

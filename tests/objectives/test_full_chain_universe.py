from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
from joker.objectives.full_chain_universe import (
    FullChainUniverseSettings,
    build_full_chain_universe,
)

TRADING_DATE = date(2026, 8, 5)
NOW = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)


def _contract(
    strike: str,
    option_type: str,
    ask: str,
    *,
    bid: str | None = None,
    age_ms: int = 0,
    expiry: date = TRADING_DATE,
    symbol: str = "SPY",
) -> OptionContractSnapshot:
    ask_d = Decimal(ask)
    bid_d = Decimal(bid) if bid is not None else max(Decimal("0.01"), ask_d * Decimal("0.90"))
    mid = (bid_d + ask_d) / Decimal("2")
    return OptionContractSnapshot(
        contract_id=f"SPY:{expiry.isoformat()}:{strike}:{option_type}",
        symbol=symbol,
        expiry=expiry,
        strike=Decimal(strike),
        option_type=option_type,  # type: ignore[arg-type]
        bid=bid_d,
        ask=ask_d,
        mid=mid,
        quote_timestamp=NOW,
        quote_age_ms=age_ms,
        relative_spread=(ask_d - bid_d) / mid,
        liquidity_score=0.6,
    )


def _surface(contracts: list[OptionContractSnapshot]) -> OptionSurfaceSnapshot:
    return OptionSurfaceSnapshot(
        surface_id=uuid4(),
        exchange_time=NOW,
        trading_date=TRADING_DATE,
        underlying_symbol="SPY",
        underlying_price=Decimal("500"),
        contracts=tuple(contracts),
    )


def _build(
    contracts: list[OptionContractSnapshot],
    *,
    budget: str = "200",
    maximum: int = 200,
):
    return build_full_chain_universe(
        snapshot_id=uuid4(),
        surface=_surface(contracts),
        trading_date=TRADING_DATE,
        available_capital_usd=Decimal(budget),
        settings=FullChainUniverseSettings(maximum_contracts_evaluated=maximum),
    )


def test_cheap_far_otm_and_near_atm_are_both_included() -> None:
    universe = _build(
        [
            _contract("500", "call", "1.00"),
            _contract("520", "call", "0.10", bid="0.09"),
            _contract("480", "put", "0.20", bid="0.18"),
        ]
    )
    ids = {row.contract_id for row in universe.contracts}
    assert "SPY:2026-08-05:500:call" in ids
    assert "SPY:2026-08-05:520:call" in ids
    assert "SPY:2026-08-05:480:put" in ids


def test_discovery_is_independent_of_agent_candidate_legs() -> None:
    contracts = [
        _contract("500", "call", "1.00"),
        _contract("510", "call", "0.50"),
    ]
    assert len(_build(contracts).contracts) == 2


def test_surface_order_does_not_change_universe_or_ranking() -> None:
    contracts = [
        _contract("490", "put", "0.20"),
        _contract("500", "call", "1.00"),
        _contract("515", "call", "0.10"),
        _contract("505", "put", "0.50"),
    ]
    forward = [row.contract_id for row in _build(contracts, maximum=3).contracts]
    reverse = [row.contract_id for row in _build(list(reversed(contracts)), maximum=3).contracts]
    assert forward == reverse


def test_invalid_surface_rows_are_excluded_fail_closed() -> None:
    stale = _contract("501", "call", "0.50", age_ms=31_000)
    invalid_quote = _contract("502", "call", "0.50").model_copy(
        update={"bid": Decimal("0"), "ask": Decimal("0")}
    )
    malformed = _contract("502.5", "call", "0.50").model_copy(
        update={"contract_id": "fabricated-contract"}
    )
    non_0dte = _contract(
        "503", "call", "0.50", expiry=date(2026, 8, 6)
    )
    excessive = _contract("504", "call", "1.00", bid="0.20")
    valid = _contract("505", "call", "0.50")
    universe = _build(
        [stale, invalid_quote, malformed, non_0dte, excessive, valid]
    )
    assert [row.contract_id for row in universe.contracts] == [valid.contract_id]
    assert universe.exclusion_counts["stale_contract"] == 1
    assert universe.exclusion_counts["invalid_bid_ask"] == 1
    assert universe.exclusion_counts["invalid_contract_id"] == 1
    assert universe.exclusion_counts["non_0dte_contract"] == 1
    assert universe.exclusion_counts["excessive_spread"] == 1


def test_no_first_eighty_row_bias_and_stratification_keeps_cheap_contract() -> None:
    contracts = [
        _contract(str(450 + index), "call", "1.00")
        for index in range(100)
    ]
    cheap = _contract("560", "call", "0.10", bid="0.09")
    universe = _build([*contracts, cheap], budget="500", maximum=20)
    assert universe.stratified is True
    assert cheap.contract_id in {row.contract_id for row in universe.contracts}


def test_two_hundred_dollar_budget_keeps_affordable_premiums() -> None:
    premiums = ("0.10", "0.20", "0.50", "1.00", "2.25")
    contracts = [
        _contract(str(500 + index), "call", premium)
        for index, premium in enumerate(premiums)
    ]
    universe = _build(contracts, budget="200")
    asks = {row.ask for row in universe.contracts}
    assert asks == {
        Decimal("0.10"),
        Decimal("0.20"),
        Decimal("0.50"),
        Decimal("1.00"),
    }
    assert universe.exclusion_counts["unaffordable_contract"] == 1

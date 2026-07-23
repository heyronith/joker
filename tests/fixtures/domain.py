"""Shared test fixtures for domain objects."""

from __future__ import annotations

from datetime import date, datetime, timezone

from joker.schemas.domain import (
    DailyState,
    MarketSnapshot,
    OptionContract,
    OptionQuote,
    TradeCandidate,
)


def make_contract(**kwargs) -> OptionContract:
    defaults = {
        "expiration": date.today(),
        "strike": 550.0,
        "option_type": "call",
        "is_0dte": True,
    }
    defaults.update(kwargs)
    return OptionContract(**defaults)


def make_quote(contract: OptionContract | None = None, **kwargs) -> OptionQuote:
    contract = contract or make_contract()
    defaults = {
        "contract": contract,
        "bid": 1.0,
        "ask": 1.1,
        "timestamp": datetime.now(timezone.utc),
    }
    defaults.update(kwargs)
    return OptionQuote(**defaults)


def make_candidate(**kwargs) -> TradeCandidate:
    contract = make_contract()
    quote = make_quote(contract)
    defaults = {
        "run_id": "test-run",
        "setup_id": "setup-1",
        "contract": contract,
        "quote": quote,
        "direction": "long_call",
        "entry_limit_price": 1.05,
        "stop_price": 0.50,
        "take_profit_price": 2.0,
        "quantity": 1,
    }
    defaults.update(kwargs)
    return TradeCandidate(**defaults)


def make_daily_state(**kwargs) -> DailyState:
    defaults = {
        "trading_day": date.today(),
        "run_id": "test-run",
        "mode": "PAPER",
    }
    defaults.update(kwargs)
    return DailyState(**defaults)


def make_snapshot(**kwargs) -> MarketSnapshot:
    defaults = {
        "symbol": "SPY",
        "timestamp": datetime.now(timezone.utc),
        "price": 550.0,
    }
    defaults.update(kwargs)
    return MarketSnapshot(**defaults)

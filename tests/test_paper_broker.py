"""Phase 4 paper broker tests."""

from __future__ import annotations

import pytest

from joker.broker.interface import BrokerError, PaperBroker
from joker.schemas.domain import OrderIntent
from tests.fixtures.domain import make_contract


def test_limit_order_fill_simulation() -> None:
    broker = PaperBroker(slippage_pct=0.0)
    intent = OrderIntent(
        candidate_id="c1",
        contract=make_contract(),
        side="buy",
        order_type="limit",
        limit_price=2.0,
    )
    order = broker.submit_order(intent)
    assert order.status == "filled"
    assert len(broker.list_positions()) == 1


def test_non_filled_order_remains_open() -> None:
    broker = PaperBroker(slippage_pct=50.0)
    intent = OrderIntent(
        candidate_id="c2",
        contract=make_contract(),
        side="buy",
        order_type="limit",
        limit_price=0.01,
    )
    order = broker.submit_order(intent)
    assert order.status == "open"


def test_cancellation_works() -> None:
    broker = PaperBroker(slippage_pct=50.0)
    intent = OrderIntent(
        candidate_id="c3",
        contract=make_contract(),
        side="buy",
        limit_price=0.01,
    )
    order = broker.submit_order(intent)
    cancelled = broker.cancel_order(order.order_id)
    assert cancelled.status == "cancelled"


def test_position_updates_on_fill() -> None:
    broker = PaperBroker(slippage_pct=0.0)
    intent = OrderIntent(
        candidate_id="c4",
        contract=make_contract(),
        side="buy",
        limit_price=1.5,
    )
    broker.submit_order(intent)
    positions = broker.list_positions()
    assert len(positions) == 1
    assert positions[0].is_open


def test_pnl_on_close() -> None:
    broker = PaperBroker(slippage_pct=0.0)
    contract = make_contract()
    buy = OrderIntent(candidate_id="c5", contract=contract, side="buy", limit_price=1.0)
    broker.submit_order(buy)
    sell = OrderIntent(candidate_id="c5", contract=contract, side="sell", limit_price=1.5)
    broker.submit_order(sell)
    assert broker.get_daily_pnl() == 50.0


def test_paper_broker_has_no_webull_import() -> None:
    import joker.broker.interface as mod

    source_file = mod.__file__
    assert source_file
    content = open(source_file).read()
    assert "webull" not in content.lower()

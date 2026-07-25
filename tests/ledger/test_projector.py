"""Partial fills, cancel, reject, realized PnL — fills are sole position source."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from joker.ledger.projector import LedgerProjector, OrderStatus
from joker.ledger.schemas import LedgerEventType, make_ledger_event

NOW = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)


def _evt(etype, **kw):
    base = dict(
        broker_account_id="a",
        client_order_id="c1",
        contract_id="SPY:c",
        side="buy",
        quantity=Decimal("1"),
        exchange_timestamp=NOW,
        session_id="s",
        idempotency_key=str(etype) + str(kw.get("quantity", "")) + str(kw.get("side", "")),
    )
    base.update(kw)
    return make_ledger_event(etype, **base)


def test_one_contract_buy_opens_exactly_one() -> None:
    p = LedgerProjector()
    state = p.project(
        [
            _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("1"), idempotency_key="s"),
            _evt(LedgerEventType.BROKER_ORDER_ACCEPTED, quantity=Decimal("1"), idempotency_key="a"),
            _evt(
                LedgerEventType.FINAL_FILL,
                quantity=Decimal("1"),
                price=Decimal("2.00"),
                idempotency_key="f",
            ),
        ]
    )
    pos = state.positions["SPY:c"]
    assert pos.quantity == Decimal("1")
    assert pos.open is True
    assert pos.avg_price == Decimal("2.00")


def test_one_contract_sell_closes_to_zero() -> None:
    p = LedgerProjector()
    state = p.project(
        [
            _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("1"), idempotency_key="s1"),
            _evt(
                LedgerEventType.FINAL_FILL,
                quantity=Decimal("1"),
                price=Decimal("2.00"),
                idempotency_key="f1",
            ),
            _evt(
                LedgerEventType.ORDER_SUBMISSION_REQUESTED,
                client_order_id="c2",
                side="sell",
                quantity=Decimal("1"),
                idempotency_key="s2",
            ),
            _evt(
                LedgerEventType.FINAL_FILL,
                client_order_id="c2",
                side="sell",
                quantity=Decimal("1"),
                price=Decimal("2.50"),
                idempotency_key="f2",
            ),
        ]
    )
    pos = state.positions["SPY:c"]
    assert pos.quantity == Decimal("0")
    assert pos.open is False
    assert pos.realized_pnl == Decimal("50")  # (2.50-2.00)*1*100


def test_partial_fills_accumulate() -> None:
    p = LedgerProjector()
    events = [
        _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("3"), idempotency_key="s"),
        _evt(LedgerEventType.BROKER_ORDER_ACCEPTED, quantity=Decimal("3"), idempotency_key="a"),
        _evt(
            LedgerEventType.PARTIAL_FILL,
            quantity=Decimal("1"),
            price=Decimal("2.0"),
            idempotency_key="p1",
        ),
        _evt(
            LedgerEventType.FINAL_FILL,
            quantity=Decimal("2"),
            price=Decimal("2.5"),
            idempotency_key="f1",
        ),
    ]
    state = p.project(events)
    order = state.orders["c1"]
    assert order.filled_qty == Decimal("3")
    assert order.avg_fill_price is not None
    assert abs(order.avg_fill_price - (Decimal("7") / Decimal("3"))) < Decimal("0.0001")
    assert state.positions["SPY:c"].quantity == Decimal("3")


def test_overfill_cannot_inflate_position() -> None:
    p = LedgerProjector()
    state = p.project(
        [
            _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("1"), idempotency_key="s"),
            _evt(
                LedgerEventType.FINAL_FILL,
                quantity=Decimal("5"),
                price=Decimal("1.0"),
                idempotency_key="f",
            ),
        ]
    )
    assert state.orders["c1"].filled_qty == Decimal("1")
    assert state.positions["SPY:c"].quantity == Decimal("1")


def test_position_ledger_events_do_not_double_count_fills() -> None:
    p = LedgerProjector()
    state = p.project(
        [
            _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("1"), idempotency_key="s"),
            _evt(
                LedgerEventType.FINAL_FILL,
                quantity=Decimal("1"),
                price=Decimal("2.0"),
                idempotency_key="f",
            ),
            # Legacy position event must not inflate quantity.
            _evt(
                LedgerEventType.POSITION_OPENED,
                quantity=Decimal("1"),
                price=Decimal("2.0"),
                idempotency_key="po",
            ),
        ]
    )
    assert state.positions["SPY:c"].quantity == Decimal("1")


def test_duplicate_events_noop() -> None:
    p = LedgerProjector()
    e = _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("1"), idempotency_key="s")
    state = p.project([e, e])
    assert state.orders["c1"].submitted_qty == Decimal("1")


def test_cancel_and_reject() -> None:
    p = LedgerProjector()
    state = p.project(
        [
            _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, idempotency_key="s"),
            _evt(LedgerEventType.CANCELLATION, quantity=Decimal("0"), idempotency_key="c"),
        ]
    )
    assert state.orders["c1"].status == OrderStatus.CANCELLED
    state2 = p.project(
        [
            _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, idempotency_key="s2"),
            _evt(LedgerEventType.REJECTION, quantity=Decimal("0"), idempotency_key="r"),
        ]
    )
    assert state2.orders["c1"].status == OrderStatus.REJECTED


def test_replay_deterministic() -> None:
    p = LedgerProjector()
    events = [
        _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("2"), idempotency_key="s"),
        _evt(
            LedgerEventType.PARTIAL_FILL,
            quantity=Decimal("1"),
            price=Decimal("1.0"),
            idempotency_key="p",
        ),
        _evt(
            LedgerEventType.FINAL_FILL,
            quantity=Decimal("1"),
            price=Decimal("1.2"),
            idempotency_key="f",
        ),
    ]
    a = p.project(events)
    b = p.project(events)
    assert a.model_dump() == b.model_dump()

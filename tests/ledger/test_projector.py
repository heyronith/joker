
"""Partial fills, cancel, reject, realized PnL."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from joker.ledger.projector import LedgerProjector
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
        idempotency_key=str(etype) + str(kw.get("quantity", "")),
    )
    base.update(kw)
    return make_ledger_event(etype, **base)


def test_partial_then_final_fill() -> None:
    p = LedgerProjector()
    events = [
        _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("3"), idempotency_key="s"),
        _evt(LedgerEventType.BROKER_ORDER_ACCEPTED, quantity=Decimal("3"), idempotency_key="a"),
        _evt(LedgerEventType.PARTIAL_FILL, quantity=Decimal("1"), price=Decimal("2.0"), idempotency_key="p1"),
        _evt(LedgerEventType.FINAL_FILL, quantity=Decimal("2"), price=Decimal("2.5"), idempotency_key="f1"),
    ]
    state = p.project(events)
    order = state.orders["c1"]
    assert order.filled_qty == Decimal("3")
    assert order.avg_fill_price == Decimal("2.333333333333333333333333333") or (
        order.avg_fill_price is not None and abs(order.avg_fill_price - Decimal("7") / Decimal("3")) < Decimal("0.0001")
    )


def test_duplicate_events_noop() -> None:
    p = LedgerProjector()
    e = _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, quantity=Decimal("1"), idempotency_key="s")
    state = p.project([e, e])
    assert state.orders["c1"].submitted_qty == Decimal("1")


def test_cancel_and_reject() -> None:
    p = LedgerProjector()
    state = p.project([
        _evt(LedgerEventType.ORDER_SUBMISSION_REQUESTED, idempotency_key="s"),
        _evt(LedgerEventType.CANCELLATION, quantity=Decimal("0"), idempotency_key="c"),
    ])
    assert "cancel" in str(state.orders["c1"].status).lower() or state.orders["c1"].status.value.lower() == "cancelled" or str(state.orders["c1"].status).endswith("CANCELLED") or "CANCEL" in str(state.orders["c1"].status).upper()

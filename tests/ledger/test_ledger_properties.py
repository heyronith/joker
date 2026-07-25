
"""Hypothesis properties for ledger projection."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from hypothesis import given, settings, strategies as st

from joker.ledger.projector import LedgerProjector
from joker.ledger.schemas import LedgerEventType, make_ledger_event

NOW = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)


@given(
    submitted=st.decimals(min_value=1, max_value=20, places=0),
    fill_a=st.decimals(min_value=1, max_value=10, places=0),
    fill_b=st.decimals(min_value=1, max_value=10, places=0),
    price=st.decimals(min_value="0.5", max_value="5", places=2),
)
@settings(max_examples=40, deadline=None)
def test_filled_never_exceeds_submitted_without_correction(submitted, fill_a, fill_b, price) -> None:
    p = LedgerProjector()
    events = [
        make_ledger_event(
            LedgerEventType.ORDER_SUBMISSION_REQUESTED,
            broker_account_id="a", client_order_id="c", contract_id="x", side="buy",
            quantity=submitted, exchange_timestamp=NOW, idempotency_key="s", session_id="s",
        ),
        make_ledger_event(
            LedgerEventType.PARTIAL_FILL,
            broker_account_id="a", client_order_id="c", contract_id="x", side="buy",
            quantity=fill_a, price=price, exchange_timestamp=NOW, idempotency_key="p1", session_id="s",
        ),
        make_ledger_event(
            LedgerEventType.FINAL_FILL,
            broker_account_id="a", client_order_id="c", contract_id="x", side="buy",
            quantity=fill_b, price=price, exchange_timestamp=NOW, idempotency_key="f1", session_id="s",
        ),
    ]
    state = p.project(events)
    order = state.orders["c"]
    assert order.filled_qty <= order.submitted_qty


@given(n=st.integers(min_value=1, max_value=5))
@settings(max_examples=20, deadline=None)
def test_replay_duplicates_stable(n) -> None:
    p = LedgerProjector()
    e = make_ledger_event(
        LedgerEventType.ORDER_SUBMISSION_REQUESTED,
        broker_account_id="a", client_order_id="c", contract_id="x", side="buy",
        quantity=Decimal("2"), exchange_timestamp=NOW, idempotency_key="s", session_id="s",
    )
    state1 = p.project([e] * n)
    state2 = p.project([e] * n)
    assert state1.orders["c"].submitted_qty == state2.orders["c"].submitted_qty == Decimal("2")

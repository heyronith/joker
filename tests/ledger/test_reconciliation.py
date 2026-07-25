
"""Reconciliation mismatch findings."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.ledger.projector import ProjectionState
from joker.ledger.reconciliation import (
    BrokerOpenOrderView,
    BrokerReconciler,
    ReconciliationOutcome,
)


def test_unknown_broker_order_finding() -> None:
    report = BrokerReconciler().reconcile(
        session_id="s1",
        projection=ProjectionState(),
        broker_orders=[
            BrokerOpenOrderView(
                broker_order_id="b1",
                client_order_id="unknown",
                contract_id="c1",
                side="buy",
                quantity=1,
                status="open",
            )
        ],
        broker_positions=[],
        exchange_timestamp=datetime.now(timezone.utc),
    )
    outcomes = {f.outcome for f in report.findings}
    assert ReconciliationOutcome.UNKNOWN_BROKER_ORDER in outcomes

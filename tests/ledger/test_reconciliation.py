"""Reconciliation mismatch findings and correction classification."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from joker.ledger.projector import OrderLifecycle, OrderStatus, PositionState, ProjectionState
from joker.ledger.reconciliation import (
    BrokerOpenOrderView,
    BrokerPositionView,
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


def test_position_quantity_mismatch_correction_kind() -> None:
    projection = ProjectionState(
        positions={
            "c1": PositionState(
                contract_id="c1",
                quantity=Decimal("1"),
                avg_price=Decimal("1.0"),
                open=True,
            )
        }
    )
    report = BrokerReconciler().reconcile(
        session_id="s1",
        projection=projection,
        broker_orders=[],
        broker_positions=[
            BrokerPositionView(contract_id="c1", quantity=Decimal("2"), avg_price=Decimal("1.0"))
        ],
        exchange_timestamp=datetime.now(timezone.utc),
    )
    corrections = report.correction_events(broker_account_id="acct")
    assert corrections
    kinds = {c.metadata.get("correction_kind") for c in corrections}
    assert "position_quantity" in kinds


def test_order_fill_qty_mismatch_correction_kind() -> None:
    projection = ProjectionState(
        orders={
            "o1": OrderLifecycle(
                client_order_id="o1",
                status=OrderStatus.PARTIALLY_FILLED,
                submitted_qty=Decimal("2"),
                filled_qty=Decimal("1"),
                side="buy",
                contract_id="c1",
                broker_order_id="b1",
            )
        }
    )
    report = BrokerReconciler().reconcile(
        session_id="s1",
        projection=projection,
        broker_orders=[
            BrokerOpenOrderView(
                broker_order_id="b1",
                client_order_id="o1",
                contract_id="c1",
                side="buy",
                quantity=Decimal("2"),
                filled_qty=Decimal("2"),
                status="partially_filled",
            )
        ],
        broker_positions=[],
        exchange_timestamp=datetime.now(timezone.utc),
    )
    corrections = report.correction_events(broker_account_id="acct")
    assert any(c.metadata.get("correction_kind") == "order_fill_qty" for c in corrections)

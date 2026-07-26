"""Helpers for Task 3 tests that need Task 1 projection truth."""

from __future__ import annotations

from decimal import Decimal

from joker.ledger.projector import (
    OrderLifecycle,
    OrderStatus,
    PositionState,
    ProjectionState,
)


class FakeExecutionProjection:
    def __init__(self, projection: ProjectionState) -> None:
        self._projection = projection

    async def project_session(self) -> ProjectionState:
        return self._projection


def closed_trade_projection(
    *,
    contract_id: str,
    entry_id: str = "entry-1",
    exit_id: str = "exit-1",
    entry_price: Decimal = Decimal("1.00"),
    exit_price: Decimal = Decimal("1.50"),
    qty: Decimal = Decimal("1"),
    realised_pnl: Decimal = Decimal("50"),
    remaining_mismatch: bool = False,
    session_id: str = "s",
    position_lifecycle_id: str | None = None,
) -> ProjectionState:
    exit_qty = qty if not remaining_mismatch else qty - Decimal("1")
    lifecycle = position_lifecycle_id or f"{session_id}:{entry_id}:{contract_id}"
    return ProjectionState(
        orders={
            entry_id: OrderLifecycle(
                client_order_id=entry_id,
                status=OrderStatus.FILLED,
                submitted_qty=qty,
                filled_qty=qty,
                avg_fill_price=entry_price,
                side="buy",
                contract_id=contract_id,
                position_lifecycle_id=lifecycle,
                originating_entry_client_order_id=entry_id,
            ),
            exit_id: OrderLifecycle(
                client_order_id=exit_id,
                status=OrderStatus.FILLED,
                submitted_qty=exit_qty,
                filled_qty=exit_qty,
                avg_fill_price=exit_price,
                side="sell",
                contract_id=contract_id,
                position_lifecycle_id=lifecycle,
                originating_entry_client_order_id=entry_id,
                parent_client_order_id=entry_id,
            ),
        },
        positions={
            contract_id: PositionState(
                contract_id=contract_id,
                quantity=Decimal("0"),
                avg_price=entry_price,
                realized_pnl=realised_pnl,
                open=False,
                position_lifecycle_id=lifecycle,
                originating_entry_client_order_id=entry_id,
            )
        },
    )

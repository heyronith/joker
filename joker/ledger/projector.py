"""Deterministic ledger projector: orders, positions, P&L from append-only events."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Iterable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from joker.ledger.schemas import LedgerEvent, LedgerEventType, Side

_MULTIPLIER = Decimal("100")
_FILL_TYPES = frozenset({LedgerEventType.PARTIAL_FILL, LedgerEventType.FINAL_FILL})


class OrderStatus(StrEnum):
    """Projected order lifecycle status."""

    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderLifecycle(BaseModel):
    """Projected state for a single client order."""

    model_config = ConfigDict(frozen=True)

    client_order_id: str
    status: OrderStatus
    submitted_qty: Decimal = Decimal("0")
    filled_qty: Decimal = Decimal("0")
    avg_fill_price: Decimal | None = None
    fees: Decimal = Decimal("0")
    side: Side
    contract_id: str
    broker_order_id: str | None = None
    position_lifecycle_id: str | None = None
    originating_entry_client_order_id: str | None = None
    parent_client_order_id: str | None = None
    causation_event_id: str | None = None


class PositionState(BaseModel):
    """Projected position for a contract. open is True only when quantity != 0."""

    model_config = ConfigDict(frozen=True)

    contract_id: str
    quantity: Decimal = Decimal("0")
    avg_price: Decimal | None = None
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    open: bool = False
    position_id: str | None = None
    position_lifecycle_id: str | None = None
    configuration_version_id: str | None = None


class ProjectionState(BaseModel):
    """Full projected ledger view. Deterministic for a given event sequence."""

    model_config = ConfigDict(frozen=True)

    orders: dict[str, OrderLifecycle] = Field(default_factory=dict)
    positions: dict[str, PositionState] = Field(default_factory=dict)
    capital_usage: Decimal = Decimal("0")
    seen_event_ids: frozenset[UUID] = Field(default_factory=frozenset)
    seen_idempotency_keys: frozenset[str] = Field(default_factory=frozenset)


class LedgerProjector:
    """Project immutable ledger events into order and position state.

    Safety rules:
    - Never derive a fill from a limit/request price.
    - Never open a position until filled quantity is positive.
    - Never close a position until a closing fill confirms quantity reduction.
    - Duplicate events (by ledger_event_id or idempotency_key) are no-ops.
    - Partial fills accumulate a quantity-weighted average fill price.
    - Replaying the same sequence yields identical ProjectionState.
    """

    def project(self, events: Iterable[LedgerEvent]) -> ProjectionState:
        orders: dict[str, _MutableOrder] = {}
        positions: dict[str, _MutablePosition] = {}
        seen_ids: set[UUID] = set()
        seen_keys: set[str] = set()

        for event in events:
            if event.ledger_event_id in seen_ids or event.idempotency_key in seen_keys:
                continue
            seen_ids.add(event.ledger_event_id)
            seen_keys.add(event.idempotency_key)
            self._apply(event, orders, positions)

        frozen_orders = {k: v.to_model() for k, v in orders.items()}
        frozen_positions = {k: v.to_model() for k, v in positions.items()}
        capital = Decimal("0")
        for pos in frozen_positions.values():
            if pos.open and pos.avg_price is not None:
                capital += abs(pos.quantity) * pos.avg_price * _MULTIPLIER

        return ProjectionState(
            orders=frozen_orders,
            positions=frozen_positions,
            capital_usage=capital,
            seen_event_ids=frozenset(seen_ids),
            seen_idempotency_keys=frozenset(seen_keys),
        )

    def _apply(
        self,
        event: LedgerEvent,
        orders: dict[str, _MutableOrder],
        positions: dict[str, _MutablePosition],
    ) -> None:
        et = event.event_type

        if et == LedgerEventType.ORDER_SUBMISSION_REQUESTED:
            order = orders.get(event.client_order_id)
            if order is None:
                orders[event.client_order_id] = _MutableOrder(
                    client_order_id=event.client_order_id,
                    status=OrderStatus.SUBMITTED,
                    submitted_qty=event.quantity,
                    side=event.side,
                    contract_id=event.contract_id,
                    broker_order_id=event.broker_order_id,
                )
            else:
                order.submitted_qty = event.quantity
                order.side = event.side
                order.contract_id = event.contract_id
                if event.broker_order_id:
                    order.broker_order_id = event.broker_order_id
                if order.status not in {
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                }:
                    order.status = OrderStatus.SUBMITTED
            self._add_fees(order=orders.get(event.client_order_id), event=event, positions=positions)
            return

        if et == LedgerEventType.BROKER_ORDER_ACCEPTED:
            order = self._ensure_order(event, orders)
            if event.broker_order_id:
                order.broker_order_id = event.broker_order_id
            if event.quantity > 0 and order.submitted_qty == 0:
                order.submitted_qty = event.quantity
            if order.status not in {
                OrderStatus.FILLED,
                OrderStatus.CANCELLED,
                OrderStatus.REJECTED,
                OrderStatus.PARTIALLY_FILLED,
            }:
                order.status = OrderStatus.ACCEPTED
            self._add_fees(order=order, event=event, positions=positions)
            return

        if et in _FILL_TYPES:
            if event.price is None or event.quantity <= 0:
                return
            order = self._ensure_order(event, orders)
            if event.broker_order_id:
                order.broker_order_id = event.broker_order_id
            allow_overfill = bool(event.metadata.get("allow_overfill"))
            applied_qty = self._accumulate_fill(
                order,
                qty=event.quantity,
                price=event.price,
                allow_overfill=allow_overfill,
            )
            if applied_qty <= 0:
                self._add_fees(order=order, event=event, positions=positions)
                return
            if et == LedgerEventType.FINAL_FILL or (
                order.submitted_qty > 0 and order.filled_qty >= order.submitted_qty
            ):
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            # Fills are the sole source of position quantity / realised P&L.
            self._apply_fill_to_position(
                event,
                positions,
                applied_qty=applied_qty,
            )
            self._add_fees(order=order, event=event, positions=positions)
            return

        if et == LedgerEventType.CANCELLATION:
            order = self._ensure_order(event, orders)
            if order.status not in {OrderStatus.FILLED, OrderStatus.REJECTED}:
                order.status = OrderStatus.CANCELLED
            self._add_fees(order=order, event=event, positions=positions)
            return

        if et == LedgerEventType.REJECTION:
            order = self._ensure_order(event, orders)
            if order.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED}:
                order.status = OrderStatus.REJECTED
            self._add_fees(order=order, event=event, positions=positions)
            return

        if et == LedgerEventType.FEE_RECORDED:
            self._add_fees(
                order=orders.get(event.client_order_id),
                event=event,
                positions=positions,
            )
            return

        # POSITION_OPENED / RESIZED / CLOSED ledger events are legacy/compat only.
        # Authoritative quantity and realised P&L come from PARTIAL_FILL / FINAL_FILL.
        # Explicit reconciliation corrections may still adjust state below.
        if et in {
            LedgerEventType.POSITION_OPENED,
            LedgerEventType.POSITION_RESIZED,
            LedgerEventType.POSITION_CLOSED,
        }:
            # Ignore quantity mutation from position lifecycle events to prevent
            # double-counting fills. Fees may still be recorded.
            self._add_fees(order=None, event=event, positions=positions)
            return

        if et == LedgerEventType.RECONCILIATION_CORRECTION:
            self._apply_correction(event, orders, positions)

    def _apply_correction(
        self,
        event: LedgerEvent,
        orders: dict[str, _MutableOrder],
        positions: dict[str, _MutablePosition],
    ) -> None:
        kind = str(event.metadata.get("correction_kind", ""))
        if kind == "order_status" and event.client_order_id in orders:
            status_raw = event.metadata.get("status")
            if isinstance(status_raw, str):
                try:
                    orders[event.client_order_id].status = OrderStatus(status_raw)
                except ValueError:
                    pass
        if kind == "order_fill_qty" and event.client_order_id in orders:
            order = orders[event.client_order_id]
            order.filled_qty = event.quantity
            if event.price is not None:
                order.avg_fill_price = event.price
            if order.submitted_qty > 0 and order.filled_qty >= order.submitted_qty:
                order.status = OrderStatus.FILLED
            elif order.filled_qty > 0:
                order.status = OrderStatus.PARTIALLY_FILLED
        if kind == "position_quantity":
            pos = self._ensure_position(event.contract_id, positions)
            if "absolute_quantity" in event.metadata:
                pos.quantity = Decimal(str(event.metadata["absolute_quantity"]))
            else:
                pos.quantity = event.quantity if event.side == "buy" else -event.quantity
            if event.price is not None:
                pos.avg_price = event.price
            pos.open = pos.quantity != 0
            if pos.quantity == 0:
                pos.avg_price = None
        self._add_fees(
            order=orders.get(event.client_order_id),
            event=event,
            positions=positions,
        )

    def _ensure_order(
        self,
        event: LedgerEvent,
        orders: dict[str, _MutableOrder],
    ) -> _MutableOrder:
        order = orders.get(event.client_order_id)
        if order is None:
            submitted = (
                event.quantity
                if event.event_type == LedgerEventType.ORDER_SUBMISSION_REQUESTED
                else (event.quantity if event.event_type in _FILL_TYPES else Decimal("0"))
            )
            order = _MutableOrder(
                client_order_id=event.client_order_id,
                status=OrderStatus.SUBMITTED,
                submitted_qty=submitted,
                side=event.side,
                contract_id=event.contract_id,
                broker_order_id=event.broker_order_id,
            )
            orders[event.client_order_id] = order
        return order

    @staticmethod
    def _ensure_position(
        contract_id: str,
        positions: dict[str, _MutablePosition],
    ) -> _MutablePosition:
        pos = positions.get(contract_id)
        if pos is None:
            pos = _MutablePosition(contract_id=contract_id)
            positions[contract_id] = pos
        return pos

    @staticmethod
    def _accumulate_fill(
        order: _MutableOrder,
        *,
        qty: Decimal,
        price: Decimal,
        allow_overfill: bool = False,
    ) -> Decimal:
        """Accumulate fill on the order; return quantity actually applied after clamp."""
        prior_qty = order.filled_qty
        applied = qty
        # Hard rule: filled quantity never exceeds submitted unless explicit correction.
        if not allow_overfill and order.submitted_qty > 0:
            remaining = order.submitted_qty - prior_qty
            if remaining <= 0:
                return Decimal("0")
            applied = min(qty, remaining)
        if applied <= 0:
            return Decimal("0")
        new_qty = prior_qty + applied
        if prior_qty <= 0 or order.avg_fill_price is None:
            order.avg_fill_price = price
        else:
            order.avg_fill_price = (
                (order.avg_fill_price * prior_qty) + (price * applied)
            ) / new_qty
        order.filled_qty = new_qty
        return applied

    def _apply_fill_to_position(
        self,
        event: LedgerEvent,
        positions: dict[str, _MutablePosition],
        *,
        applied_qty: Decimal,
    ) -> None:
        assert event.price is not None
        if applied_qty <= 0:
            return
        pos = self._ensure_position(event.contract_id, positions)
        signed = applied_qty if event.side == "buy" else -applied_qty

        if pos.quantity == 0:
            pos.quantity = signed
            pos.avg_price = event.price
            pos.open = True
            if event.position_id:
                pos.position_id = event.position_id
            return

        same_direction = (pos.quantity > 0 and signed > 0) or (pos.quantity < 0 and signed < 0)
        if same_direction:
            self._resize_position(pos, delta_qty=signed, price=event.price)
            return

        close_qty = min(abs(pos.quantity), abs(signed))
        if close_qty > 0 and pos.avg_price is not None:
            self._realize(pos, close_qty=close_qty, exit_price=event.price)
        # Closing fills cannot drive quantity past zero without an explicit correction.
        if pos.quantity > 0:
            pos.quantity = pos.quantity - close_qty
        else:
            pos.quantity = pos.quantity + close_qty
        if pos.quantity == 0:
            pos.open = False
            pos.avg_price = None
        else:
            pos.open = True

    @staticmethod
    def _resize_position(pos: _MutablePosition, *, delta_qty: Decimal, price: Decimal) -> None:
        new_qty = pos.quantity + delta_qty
        if pos.quantity == 0 or pos.avg_price is None:
            pos.quantity = new_qty
            pos.avg_price = price if new_qty != 0 else None
            pos.open = new_qty != 0
            return
        abs_prior = abs(pos.quantity)
        abs_delta = abs(delta_qty)
        pos.avg_price = ((pos.avg_price * abs_prior) + (price * abs_delta)) / (abs_prior + abs_delta)
        pos.quantity = new_qty
        pos.open = new_qty != 0
        if new_qty == 0:
            pos.avg_price = None

    @staticmethod
    def _realize(pos: _MutablePosition, *, close_qty: Decimal, exit_price: Decimal) -> None:
        if pos.avg_price is None or close_qty <= 0:
            return
        if pos.quantity > 0:
            pnl = (exit_price - pos.avg_price) * close_qty * _MULTIPLIER
        else:
            pnl = (pos.avg_price - exit_price) * close_qty * _MULTIPLIER
        pos.realized_pnl += pnl

    @staticmethod
    def _add_fees(
        *,
        order: _MutableOrder | None,
        event: LedgerEvent,
        positions: dict[str, _MutablePosition],
    ) -> None:
        if event.fees is None or event.fees == 0:
            return
        if order is not None:
            order.fees += event.fees
        pos = positions.get(event.contract_id)
        if pos is not None:
            pos.fees += event.fees
        elif event.event_type in {
            LedgerEventType.POSITION_OPENED,
            LedgerEventType.POSITION_RESIZED,
            LedgerEventType.POSITION_CLOSED,
            LedgerEventType.FEE_RECORDED,
        }:
            pos = LedgerProjector._ensure_position(event.contract_id, positions)
            pos.fees += event.fees


class _MutableOrder:
    __slots__ = (
        "client_order_id",
        "status",
        "submitted_qty",
        "filled_qty",
        "avg_fill_price",
        "fees",
        "side",
        "contract_id",
        "broker_order_id",
    )

    def __init__(
        self,
        *,
        client_order_id: str,
        status: OrderStatus,
        submitted_qty: Decimal,
        side: Side,
        contract_id: str,
        broker_order_id: str | None,
    ) -> None:
        self.client_order_id = client_order_id
        self.status = status
        self.submitted_qty = submitted_qty
        self.filled_qty = Decimal("0")
        self.avg_fill_price: Decimal | None = None
        self.fees = Decimal("0")
        self.side = side
        self.contract_id = contract_id
        self.broker_order_id = broker_order_id

    def to_model(self) -> OrderLifecycle:
        return OrderLifecycle(
            client_order_id=self.client_order_id,
            status=self.status,
            submitted_qty=self.submitted_qty,
            filled_qty=self.filled_qty,
            avg_fill_price=self.avg_fill_price,
            fees=self.fees,
            side=self.side,
            contract_id=self.contract_id,
            broker_order_id=self.broker_order_id,
        )


class _MutablePosition:
    __slots__ = (
        "contract_id",
        "quantity",
        "avg_price",
        "realized_pnl",
        "fees",
        "open",
        "position_id",
    )

    def __init__(self, *, contract_id: str) -> None:
        self.contract_id = contract_id
        self.quantity = Decimal("0")
        self.avg_price: Decimal | None = None
        self.realized_pnl = Decimal("0")
        self.fees = Decimal("0")
        self.open = False
        self.position_id: str | None = None

    def to_model(self) -> PositionState:
        return PositionState(
            contract_id=self.contract_id,
            quantity=self.quantity,
            avg_price=self.avg_price,
            realized_pnl=self.realized_pnl,
            fees=self.fees,
            open=self.open and self.quantity != 0,
            position_id=self.position_id,
        )

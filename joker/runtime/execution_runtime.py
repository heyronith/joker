"""Execution runtime — explicit broker commands, ledger, reconciliation.

Must not decide whether a trade is desirable. Only executes commanded actions
and records broker truth. Position quantity and realised P&L come only from
verified PARTIAL_FILL / FINAL_FILL ledger events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from joker.broker.interface import BrokerClient, BrokerSubmissionUnknown
from joker.broker.order_update import BrokerOrderUpdate
from joker.events.bus import InProcessAsyncEventBus
from joker.events.schemas import EventType, make_event
from joker.ledger.projector import LedgerProjector, ProjectionState
from joker.ledger.reconciliation import (
    BrokerOpenOrderView,
    BrokerPositionView,
    BrokerReconciler,
    ReconciliationReport,
)
from joker.ledger.schemas import LedgerEvent, LedgerEventType, make_ledger_event
from joker.ledger.store import SqliteLedgerStore
from joker.schemas.domain import BrokerOrder, OptionContract, OrderIntent, Position
from joker.time.clock import ExchangeClock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionCommand:
    """Explicit execution instruction — never inferred from market state."""

    client_order_id: str
    intent: OrderIntent
    broker_account_id: str = "default"
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DailyPnlView:
    """Daily P&L with explicit availability (never fabricate broker zeros)."""

    available: bool
    value: float | None


@dataclass(frozen=True)
class UnresolvedReconciliation:
    """Typed unresolved recovery state — system must not claim recovery."""

    report: ReconciliationReport
    prevents_recovery_claim: bool = True


def contract_id_for(contract: OptionContract) -> str:
    """Stable contract id for ledger / reconciliation keys."""
    return (
        f"{contract.symbol}:{contract.expiration.isoformat()}:"
        f"{contract.strike}:{contract.option_type}"
    )


def _position_contract_id(position: Position) -> str:
    return contract_id_for(position.contract)


def _verified_fill_price(broker: BrokerClient, order: BrokerOrder) -> Decimal | None:
    """Extract a verified fill price from broker state when available."""
    getter = getattr(broker, "get_fill_price", None)
    if callable(getter):
        price = getter(order.order_id)
        if price is not None:
            return Decimal(str(price))
    fills = getattr(broker, "_fills", None)
    if isinstance(fills, dict):
        for fill in fills.values():
            if getattr(fill, "order_id", None) == order.order_id:
                return Decimal(str(fill.price))
    return None


class ExecutionRuntime:
    """Broker interaction + ledger truth. No desirability / strategy logic."""

    def __init__(
        self,
        *,
        broker: BrokerClient,
        ledger_store: SqliteLedgerStore,
        projector: LedgerProjector | None = None,
        reconciler: BrokerReconciler | None = None,
        event_bus: InProcessAsyncEventBus,
        clock: ExchangeClock | None = None,
        session_id: str,
        broker_account_id: str = "default",
    ) -> None:
        self._broker = broker
        self._ledger = ledger_store
        self._projector = projector or LedgerProjector()
        self._reconciler = reconciler or BrokerReconciler()
        self._bus = event_bus
        self._clock = clock
        self._session_id = session_id
        self._broker_account_id = broker_account_id
        self._correlation_id = uuid4()
        self._client_to_broker: dict[str, str] = {}
        self._unresolved: UnresolvedReconciliation | None = None

    @property
    def client_to_broker_map(self) -> dict[str, str]:
        return dict(self._client_to_broker)

    @property
    def unresolved_reconciliation(self) -> UnresolvedReconciliation | None:
        return self._unresolved

    def _now(self) -> datetime:
        if self._clock is not None:
            return self._clock.now()
        return datetime.now(timezone.utc)

    async def restore_order_mappings(self) -> dict[str, str]:
        """Rebuild client-order → broker-order map from persisted ledger events."""
        events = await self._ledger.get_by_session(self._session_id)
        mapping: dict[str, str] = {}
        for event in events:
            if event.broker_order_id:
                mapping[event.client_order_id] = event.broker_order_id
        self._client_to_broker = mapping
        return dict(mapping)

    async def submit_execution_command(self, command: ExecutionCommand) -> BrokerOrder:
        """Submit an explicit command through the broker and append ledger events.

        Idempotent at the ExecutionRuntime boundary: if ``client_order_id`` already
        maps to an accepted/open/filled/cancelled broker lifecycle, return or
        reconcile that order instead of calling the broker again.
        """
        await self.restore_order_mappings()
        existing = await self._resolve_existing_submission(command.client_order_id)
        if existing is not None:
            logger.info(
                "execution_submit_idempotent_reuse",
                extra={
                    "client_order_id": command.client_order_id,
                    "broker_order_id": existing.order_id,
                    "status": existing.status,
                },
            )
            return existing

        now = self._now()
        intent = command.intent
        cid = contract_id_for(intent.contract)
        requested = make_ledger_event(
            LedgerEventType.ORDER_SUBMISSION_REQUESTED,
            broker_account_id=command.broker_account_id or self._broker_account_id,
            client_order_id=command.client_order_id,
            contract_id=cid,
            side=intent.side,
            quantity=Decimal(intent.quantity),
            exchange_timestamp=now,
            idempotency_key=f"submit-req:{command.client_order_id}",
            session_id=self._session_id,
            price=Decimal(str(intent.limit_price)) if intent.limit_price is not None else None,
            metadata={
                "intent_id": intent.intent_id,
                "order_type": intent.order_type,
                **dict(command.provenance),
            },
        )
        await self._ledger.append(requested)
        await self._publish_order_event(EventType.ORDER_SUBMITTED, command.client_order_id, now)

        # Broker may already hold the order if we crashed after accept but before
        # the ACCEPT ledger event was written — reconcile by intent_id.
        recovered = self._find_broker_order_by_intent(intent.intent_id)
        if recovered is not None:
            self._client_to_broker[command.client_order_id] = recovered.order_id
            accept_key = f"accept:{command.client_order_id}:{recovered.order_id}"
            accepted = make_ledger_event(
                LedgerEventType.BROKER_ORDER_ACCEPTED,
                broker_account_id=command.broker_account_id or self._broker_account_id,
                client_order_id=command.client_order_id,
                contract_id=cid,
                side=intent.side,
                quantity=Decimal(recovered.quantity),
                exchange_timestamp=self._now(),
                idempotency_key=accept_key,
                session_id=self._session_id,
                broker_order_id=recovered.order_id,
                price=(
                    Decimal(str(recovered.limit_price))
                    if recovered.limit_price is not None
                    else None
                ),
            )
            await self._ledger.append(accepted)
            return recovered

        try:
            order = self._broker.submit_order(intent)
        except BrokerSubmissionUnknown as exc:
            # Truthfully unknown — never append REJECTION / publish ORDER_REJECTED.
            # Preserve client_order_id mapping absence until broker confirms; reservation
            # is retained by the gateway. Duplicate placement is blocked by journal +
            # _resolve_existing_submission / SUBMISSION_UNKNOWN ledger identity.
            unknown = make_ledger_event(
                LedgerEventType.SUBMISSION_UNKNOWN,
                broker_account_id=command.broker_account_id or self._broker_account_id,
                client_order_id=command.client_order_id,
                contract_id=cid,
                side=intent.side,
                quantity=Decimal(intent.quantity),
                exchange_timestamp=self._now(),
                idempotency_key=f"unknown:{command.client_order_id}",
                session_id=self._session_id,
                metadata={"error": str(exc), "client_order_id": exc.client_order_id},
            )
            await self._ledger.append(unknown)
            await self._publish_order_event(
                EventType.ORDER_SUBMISSION_UNKNOWN,
                command.client_order_id,
                self._now(),
                extra={"error": str(exc)},
            )
            raise
        except Exception as exc:
            rejected = make_ledger_event(
                LedgerEventType.REJECTION,
                broker_account_id=command.broker_account_id or self._broker_account_id,
                client_order_id=command.client_order_id,
                contract_id=cid,
                side=intent.side,
                quantity=Decimal(intent.quantity),
                exchange_timestamp=self._now(),
                idempotency_key=f"reject:{command.client_order_id}:{type(exc).__name__}",
                session_id=self._session_id,
                metadata={"error": str(exc)},
            )
            await self._ledger.append(rejected)
            await self._publish_order_event(
                EventType.ORDER_REJECTED,
                command.client_order_id,
                self._now(),
                extra={"error": str(exc)},
            )
            raise

        self._client_to_broker[command.client_order_id] = order.order_id
        accepted = make_ledger_event(
            LedgerEventType.BROKER_ORDER_ACCEPTED,
            broker_account_id=command.broker_account_id or self._broker_account_id,
            client_order_id=command.client_order_id,
            contract_id=cid,
            side=intent.side,
            quantity=Decimal(order.quantity),
            exchange_timestamp=self._now(),
            idempotency_key=f"accept:{command.client_order_id}:{order.order_id}",
            session_id=self._session_id,
            broker_order_id=order.order_id,
            price=Decimal(str(order.limit_price)) if order.limit_price is not None else None,
        )
        await self._ledger.append(accepted)
        await self._publish_order_event(
            EventType.ORDER_ACCEPTED,
            command.client_order_id,
            self._now(),
            extra={"broker_order_id": order.order_id, "status": order.status},
        )

        if order.status == "filled":
            fill_price = _verified_fill_price(self._broker, order)
            if fill_price is not None:
                await self.record_verified_fill(
                    order,
                    client_order_id=command.client_order_id,
                    fill_price=fill_price,
                    fill_qty=Decimal(order.quantity),
                    final=True,
                )
        return order

    async def _resolve_existing_submission(
        self, client_order_id: str
    ) -> BrokerOrder | None:
        """Return an existing broker lifecycle for client_order_id if recoverable."""
        broker_order_id = self._client_to_broker.get(client_order_id)
        if broker_order_id:
            order = self._broker.get_order(broker_order_id)
            if order is not None:
                return order

        events = await self._ledger.get_by_session(self._session_id)
        recovered_broker_id: str | None = None
        for event in events:
            if event.client_order_id != client_order_id:
                continue
            if event.broker_order_id:
                recovered_broker_id = event.broker_order_id
                self._client_to_broker[client_order_id] = event.broker_order_id
        if recovered_broker_id:
            order = self._broker.get_order(recovered_broker_id)
            if order is not None:
                return order
        return None

    def _find_broker_order_by_intent(self, intent_id: str) -> BrokerOrder | None:
        """Locate a broker order already created for this intent (crash recovery)."""
        for order in self._broker.list_open_orders():
            if order.intent_id == intent_id:
                return order
        finder = getattr(self._broker, "find_order_by_intent_id", None)
        if callable(finder):
            found = finder(intent_id)
            if found is not None:
                return found
        # PaperBroker and similar keep filled orders in an internal map.
        orders_map = getattr(self._broker, "_orders", None)
        if isinstance(orders_map, dict):
            for order in orders_map.values():
                if getattr(order, "intent_id", None) == intent_id:
                    return order
        return None
    async def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
        """Cancel via broker adapter, append ledger cancellation, reconcile."""
        await self.restore_order_mappings()
        broker_order_id = self._client_to_broker.get(client_order_id)
        if broker_order_id is None:
            raise RuntimeError(
                f"No broker mapping for client_order_id={client_order_id!r}; "
                "cannot cancel through ExecutionRuntime"
            )
        order = self._broker.cancel_order(broker_order_id)
        await self.on_broker_update(order, client_order_id=client_order_id)
        await self.run_reconciliation()
        return order

    async def cancel_order_by_broker_id(self, broker_order_id: str) -> BrokerOrder:
        """Cancel using broker order id after resolving the client mapping."""
        await self.restore_order_mappings()
        client_order_id: str | None = None
        for cid, bid in self._client_to_broker.items():
            if bid == broker_order_id:
                client_order_id = cid
                break
        if client_order_id is None:
            # Fall back: cancel at broker, then map from order.intent_id.
            order = self._broker.cancel_order(broker_order_id)
            client_order_id = order.intent_id
            self._client_to_broker[client_order_id] = broker_order_id
            await self.on_broker_update(order, client_order_id=client_order_id)
            await self.run_reconciliation()
            return order
        return await self.cancel_order(client_order_id=client_order_id)

    async def poll_order_status(self, client_order_id: str) -> BrokerOrder | None:
        """Poll broker for order status and write ledger events on transitions."""
        broker_order_id = self._client_to_broker.get(client_order_id)
        if broker_order_id is None:
            # Attempt reconstruction from ledger if map was lost (e.g. after restart).
            await self.restore_order_mappings()
            broker_order_id = self._client_to_broker.get(client_order_id)
        if broker_order_id is None:
            return None
        order = self._broker.get_order(broker_order_id)
        if order is None:
            return None
        await self.on_broker_update(order, client_order_id=client_order_id)
        return order

    async def on_broker_update(
        self,
        order: BrokerOrder | BrokerOrderUpdate,
        *,
        client_order_id: str | None = None,
    ) -> list[LedgerEvent]:
        """Apply a broker order update into the ledger (idempotent keys)."""
        if isinstance(order, BrokerOrderUpdate):
            return await self._on_broker_order_update(order, client_order_id=client_order_id)

        client_id = client_order_id or order.intent_id
        self._client_to_broker.setdefault(client_id, order.order_id)
        written: list[LedgerEvent] = []
        now = self._now()
        cid = contract_id_for(order.contract)

        if order.status == "cancelled":
            event = make_ledger_event(
                LedgerEventType.CANCELLATION,
                broker_account_id=self._broker_account_id,
                client_order_id=client_id,
                contract_id=cid,
                side=order.side,
                quantity=Decimal(order.quantity),
                exchange_timestamp=now,
                idempotency_key=f"cancel:{client_id}:{order.order_id}",
                session_id=self._session_id,
                broker_order_id=order.order_id,
            )
            if await self._ledger.append(event):
                written.append(event)
                await self._publish_order_event(EventType.ORDER_CANCELLED, client_id, now)
        elif order.status == "rejected":
            event = make_ledger_event(
                LedgerEventType.REJECTION,
                broker_account_id=self._broker_account_id,
                client_order_id=client_id,
                contract_id=cid,
                side=order.side,
                quantity=Decimal(order.quantity),
                exchange_timestamp=now,
                idempotency_key=f"reject:{client_id}:{order.order_id}",
                session_id=self._session_id,
                broker_order_id=order.order_id,
            )
            if await self._ledger.append(event):
                written.append(event)
                await self._publish_order_event(EventType.ORDER_REJECTED, client_id, now)
        elif order.status in {"partially_filled", "filled"}:
            raw_filled = getattr(order, "filled_quantity", None)
            if raw_filled is None or int(raw_filled) <= 0:
                if order.status == "filled":
                    cumulative = int(order.quantity)
                else:
                    # Partial without authoritative cumulative — fail closed.
                    logger.error(
                        "partial_fill_truth_unavailable",
                        extra={
                            "client_order_id": client_id,
                            "broker_order_id": order.order_id,
                        },
                    )
                    return written
            else:
                cumulative = int(raw_filled)
            already = await self._persisted_fill_quantity(client_id)
            delta = cumulative - already
            if delta > 0:
                fill_price = _verified_fill_price(self._broker, order)
                if fill_price is None and order.average_fill_price is not None:
                    fill_price = Decimal(str(order.average_fill_price))
                if fill_price is not None:
                    fill_events = await self.record_verified_fill(
                        order,
                        client_order_id=client_id,
                        fill_price=fill_price,
                        fill_qty=Decimal(delta),
                        cumulative_filled_quantity=Decimal(cumulative),
                        final=order.status == "filled",
                    )
                    written.extend(fill_events)
                else:
                    logger.error(
                        "fill_price_unavailable_no_mutation",
                        extra={
                            "client_order_id": client_id,
                            "cumulative_filled_quantity": cumulative,
                        },
                    )
            # delta <= 0: already persisted — do not republish ORDER_FILLED.
        return written

    async def _on_broker_order_update(
        self,
        update: BrokerOrderUpdate,
        *,
        client_order_id: str | None = None,
    ) -> list[LedgerEvent]:
        client_id = client_order_id or update.client_order_id
        self._client_to_broker.setdefault(client_id, update.broker_order_id)
        order = self._broker.get_order(update.broker_order_id) or self._broker.get_order(
            client_id
        )
        if order is None:
            # Reconstruct a minimal BrokerOrder for fill recording when broker
            # detail is momentarily unavailable but the typed update is authoritative.
            return []
        patched = order.model_copy(
            update={
                "status": update.status,
                "filled_quantity": update.cumulative_filled_quantity,
                "remaining_quantity": update.remaining_quantity,
                "average_fill_price": (
                    float(update.average_fill_price)
                    if update.average_fill_price is not None
                    else order.average_fill_price
                ),
            }
        )
        return await self.on_broker_update(patched, client_order_id=client_id)

    async def _persisted_fill_quantity(self, client_order_id: str) -> int:
        events = await self._ledger.get_by_session(self._session_id)
        total = Decimal("0")
        for event in events:
            if event.client_order_id != client_order_id:
                continue
            if event.event_type in {
                LedgerEventType.PARTIAL_FILL,
                LedgerEventType.FINAL_FILL,
            }:
                total += Decimal(event.quantity or 0)
        return int(total)

    async def resolve_submission_unknown(
        self,
        client_order_id: str,
        *,
        side: str = "buy",
        quantity: Decimal | None = None,
        contract_id: str | None = None,
    ) -> LedgerEvent | None:
        """Transition SUBMISSION_UNKNOWN → ACCEPT or REJECT from broker truth."""
        order = self._broker.get_order(client_order_id)
        if order is None:
            return None
        now = self._now()
        cid = contract_id or contract_id_for(order.contract)
        qty = quantity if quantity is not None else Decimal(order.quantity)
        if order.status == "rejected":
            event = make_ledger_event(
                LedgerEventType.REJECTION,
                broker_account_id=self._broker_account_id,
                client_order_id=client_order_id,
                contract_id=cid,
                side=order.side or side,  # type: ignore[arg-type]
                quantity=qty,
                exchange_timestamp=now,
                idempotency_key=f"reject:{client_order_id}:{order.order_id}",
                session_id=self._session_id,
                broker_order_id=order.order_id,
                metadata={"resolved_from": "submission_unknown"},
            )
            await self._ledger.append(event)
            await self._publish_order_event(EventType.ORDER_REJECTED, client_order_id, now)
            return event
        if order.status in {"open", "pending", "partially_filled", "filled", "cancelled"}:
            self._client_to_broker[client_order_id] = order.order_id
            event = make_ledger_event(
                LedgerEventType.BROKER_ORDER_ACCEPTED,
                broker_account_id=self._broker_account_id,
                client_order_id=client_order_id,
                contract_id=cid,
                side=order.side or side,  # type: ignore[arg-type]
                quantity=qty,
                exchange_timestamp=now,
                idempotency_key=f"accept:{client_order_id}:{order.order_id}",
                session_id=self._session_id,
                broker_order_id=order.order_id,
                price=(
                    Decimal(str(order.limit_price))
                    if order.limit_price is not None
                    else None
                ),
                metadata={"resolved_from": "submission_unknown"},
            )
            await self._ledger.append(event)
            await self._publish_order_event(
                EventType.ORDER_ACCEPTED,
                client_order_id,
                now,
                extra={"broker_order_id": order.order_id, "status": order.status},
            )
            if order.status in {"partially_filled", "filled"}:
                await self.on_broker_update(order, client_order_id=client_order_id)
            return event
        return None

    async def record_verified_fill(
        self,
        order: BrokerOrder,
        *,
        client_order_id: str,
        fill_price: Decimal,
        fill_qty: Decimal | None = None,
        cumulative_filled_quantity: Decimal | None = None,
        final: bool = True,
    ) -> list[LedgerEvent]:
        """Record a fill with a verified price (never derived from limit).

        Position domain events are published by comparing projection before/after.
        No separate POSITION_* ledger events are appended for fill quantities.
        Idempotency keys include the new authoritative cumulative filled quantity
        so two equal-sized partial fills at the same price are both persisted.
        """
        before = await self.project_session()
        qty = fill_qty if fill_qty is not None else Decimal(order.quantity)
        event_type = LedgerEventType.FINAL_FILL if final else LedgerEventType.PARTIAL_FILL
        domain_type = EventType.ORDER_FILLED if final else EventType.ORDER_PARTIALLY_FILLED
        now = self._now()
        cid = contract_id_for(order.contract)
        self._client_to_broker.setdefault(client_order_id, order.order_id)
        if cumulative_filled_quantity is not None:
            cum_key = cumulative_filled_quantity
        else:
            already = await self._persisted_fill_quantity(client_order_id)
            cum_key = Decimal(already) + qty
        event = make_ledger_event(
            event_type,
            broker_account_id=self._broker_account_id,
            client_order_id=client_order_id,
            contract_id=cid,
            side=order.side,
            quantity=qty,
            exchange_timestamp=now,
            idempotency_key=(
                f"fill:{client_order_id}:{order.order_id}:{event_type.value}"
                f":{cum_key}:{fill_price}"
            ),
            session_id=self._session_id,
            broker_order_id=order.order_id,
            price=fill_price,
            metadata={"cumulative_filled_quantity": str(cum_key)},
        )
        written: list[LedgerEvent] = []
        if await self._ledger.append(event):
            written.append(event)
            remaining_qty = Decimal("0") if final else max(
                Decimal("0"), Decimal(order.quantity) - qty
            )
            await self._publish_order_event(
                domain_type,
                client_order_id,
                now,
                extra={
                    "broker_order_id": order.order_id,
                    "price": str(fill_price),
                    "qty": str(qty),
                    "fill_qty": str(qty),
                    "fill_price": str(fill_price),
                    "remaining_quantity": str(int(remaining_qty)),
                    "ledger_event_id": str(event.ledger_event_id),
                    "idempotency_key": event.idempotency_key,
                    "contract_id": cid,
                },
            )
            after = await self.project_session()
            await self._publish_position_transitions(
                before=before,
                after=after,
                client_order_id=client_order_id,
                exchange_timestamp=now,
            )
        return written

    async def _publish_position_transitions(
        self,
        *,
        before: ProjectionState,
        after: ProjectionState,
        client_order_id: str,
        exchange_timestamp: datetime,
    ) -> None:
        """Publish domain POSITION_* events from projection deltas (not ledger)."""
        all_ids = set(before.positions) | set(after.positions)
        for contract_id in all_ids:
            prev = before.positions.get(contract_id)
            curr = after.positions.get(contract_id)
            prev_qty = prev.quantity if prev is not None else Decimal("0")
            curr_qty = curr.quantity if curr is not None else Decimal("0")
            if prev_qty == 0 and curr_qty != 0:
                await self._publish_order_event(
                    EventType.POSITION_OPENED,
                    client_order_id,
                    exchange_timestamp,
                    extra={"contract_id": contract_id, "quantity": str(curr_qty)},
                )
            elif prev_qty != 0 and curr_qty == 0:
                await self._publish_order_event(
                    EventType.POSITION_CLOSED,
                    client_order_id,
                    exchange_timestamp,
                    extra={
                        "contract_id": contract_id,
                        "realized_pnl": str(curr.realized_pnl if curr else "0"),
                    },
                )
            elif prev_qty != curr_qty:
                await self._publish_order_event(
                    EventType.POSITION_CHANGED,
                    client_order_id,
                    exchange_timestamp,
                    extra={
                        "contract_id": contract_id,
                        "quantity": str(curr_qty),
                        "prior_quantity": str(prev_qty),
                    },
                )

    async def project_session(self) -> ProjectionState:
        events = await self._ledger.get_by_session(self._session_id)
        return self._projector.project(events)

    async def run_reconciliation(self) -> ReconciliationReport:
        """Compare projected ledger vs broker; never silently overwrite history."""
        projection = await self.project_session()
        broker_orders = [
            BrokerOpenOrderView(
                broker_order_id=o.order_id,
                client_order_id=o.intent_id,
                contract_id=contract_id_for(o.contract),
                side=o.side,
                quantity=Decimal(o.quantity),
                filled_qty=Decimal(
                    int(getattr(o, "filled_quantity", 0) or 0)
                    or (o.quantity if o.status == "filled" else 0)
                ),
                status=o.status,
            )
            for o in self._broker.list_open_orders()
        ]
        # Include recently known non-open orders that still matter for fill mismatch.
        for client_id, broker_id in self._client_to_broker.items():
            order = self._broker.get_order(broker_id)
            if order is None:
                continue
            if any(b.broker_order_id == broker_id for b in broker_orders):
                continue
            if order.status in {"open", "partially_filled"}:
                broker_orders.append(
                    BrokerOpenOrderView(
                        broker_order_id=order.order_id,
                        client_order_id=client_id,
                        contract_id=contract_id_for(order.contract),
                        side=order.side,
                        quantity=Decimal(order.quantity),
                        filled_qty=Decimal(
                            int(getattr(order, "filled_quantity", 0) or 0)
                            or (order.quantity if order.status == "filled" else 0)
                        ),
                        status=order.status,
                    )
                )
        broker_positions = [
            BrokerPositionView(
                contract_id=_position_contract_id(p),
                quantity=Decimal(p.quantity if p.is_open else 0),
                avg_price=Decimal(str(p.avg_entry_price)),
            )
            for p in self._broker.list_positions()
            if p.is_open
        ]
        report = self._reconciler.reconcile(
            session_id=self._session_id,
            projection=projection,
            broker_orders=broker_orders,
            broker_positions=broker_positions,
            exchange_timestamp=self._now(),
        )
        if not report.is_consistent:
            await self._bus.publish(
                make_event(
                    EventType.RECONCILIATION_REQUIRED,
                    session_id=self._session_id,
                    source="execution_runtime",
                    exchange_timestamp=self._now(),
                    correlation_id=self._correlation_id,
                    payload={
                        "report_id": str(report.report_id),
                        "is_consistent": False,
                        "finding_count": len(report.findings),
                    },
                )
            )
            logger.warning(
                "reconciliation_mismatch",
                extra={
                    "session_id": self._session_id,
                    "report_id": str(report.report_id),
                    "findings": len(report.findings),
                },
            )
        else:
            self._unresolved = None
            logger.info(
                "reconciliation_consistent",
                extra={
                    "session_id": self._session_id,
                    "report_id": str(report.report_id),
                },
            )
        return report

    async def apply_reconciliation_corrections(
        self,
        report: ReconciliationReport,
        *,
        mark_unresolved_if_still_mismatched: bool = True,
    ) -> list[LedgerEvent]:
        """Append approved reconciliation corrections to the ledger (append-only)."""
        corrections = report.correction_events(broker_account_id=self._broker_account_id)
        written: list[LedgerEvent] = []
        for event in corrections:
            if await self._ledger.append(event):
                written.append(event)
        if mark_unresolved_if_still_mismatched:
            follow_up = await self.run_reconciliation()
            if not follow_up.is_consistent:
                self._unresolved = UnresolvedReconciliation(report=follow_up)
            else:
                self._unresolved = None
        return written

    def get_daily_pnl(self) -> DailyPnlView:
        """Return broker daily P&L with availability — never invent zeros."""
        available, value = self._broker.get_daily_pnl_available()
        return DailyPnlView(available=available, value=value)

    def claims_recovery(self) -> bool:
        """False when an unresolved reconciliation blocks recovery claims."""
        return self._unresolved is None or not self._unresolved.prevents_recovery_claim

    async def _publish_order_event(
        self,
        event_type: EventType,
        client_order_id: str,
        exchange_timestamp: datetime,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"client_order_id": client_order_id}
        if extra:
            payload.update(extra)
        await self._bus.publish(
            make_event(
                event_type,
                session_id=self._session_id,
                source="execution_runtime",
                exchange_timestamp=exchange_timestamp,
                correlation_id=self._correlation_id,
                payload=payload,
            )
        )

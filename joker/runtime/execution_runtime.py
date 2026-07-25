"""Execution runtime — explicit broker commands, ledger, reconciliation.

Must not decide whether a trade is desirable. Only executes commanded actions
and records broker truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import uuid4

from joker.broker.interface import BrokerClient
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


@dataclass(frozen=True)
class DailyPnlView:
    """Daily P&L with explicit availability (never fabricate broker zeros)."""

    available: bool
    value: float | None


def contract_id_for(contract: OptionContract) -> str:
    """Stable contract id for ledger / reconciliation keys."""
    return (
        f"{contract.symbol}:{contract.expiration.isoformat()}:"
        f"{contract.strike}:{contract.option_type}"
    )


def _position_contract_id(position: Position) -> str:
    return contract_id_for(position.contract)


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

    def _now(self) -> datetime:
        if self._clock is not None:
            return self._clock.now()
        return datetime.now(timezone.utc)

    async def submit_execution_command(self, command: ExecutionCommand) -> BrokerOrder:
        """Submit an explicit command through the broker and append ledger events.

        Does not evaluate whether the trade is desirable.
        """
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
            metadata={"intent_id": intent.intent_id, "order_type": intent.order_type},
        )
        await self._ledger.append(requested)
        await self._publish_order_event(EventType.ORDER_SUBMITTED, command.client_order_id, now)

        try:
            order = self._broker.submit_order(intent)
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

        # Paper broker may fill immediately — record fill without inventing price.
        if order.status == "filled":
            await self._record_fill_from_order(order, command.client_order_id, final=True)
        return order

    async def poll_order_status(self, client_order_id: str) -> BrokerOrder | None:
        """Poll broker for order status and write ledger events on transitions."""
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
        order: BrokerOrder,
        *,
        client_order_id: str | None = None,
    ) -> list[LedgerEvent]:
        """Apply a broker order update into the ledger (idempotent keys)."""
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
        elif order.status == "filled":
            fill_events = await self._record_fill_from_order(order, client_id, final=True)
            written.extend(fill_events)
        return written

    async def _record_fill_from_order(
        self,
        order: BrokerOrder,
        client_order_id: str,
        *,
        final: bool,
        fill_price: Decimal | None = None,
        fill_qty: Decimal | None = None,
    ) -> list[LedgerEvent]:
        """Record fill(s) only when a concrete fill price is known.

        Never treats limit_price as a fill. If fill_price is omitted and the
        broker order does not expose a fill, no FINAL_FILL is written (caller
        must supply price from a verified fill record).
        """
        # PaperBroker fills immediately but BrokerOrder has no fill price field.
        # Prefer explicit fill_price; otherwise refuse to invent from limit.
        if fill_price is None:
            logger.warning(
                "fill_price_unavailable",
                extra={
                    "session_id": self._session_id,
                    "client_order_id": client_order_id,
                    "broker_order_id": order.order_id,
                    "status": order.status,
                },
            )
            # Still publish ORDER_FILLED domain event for status visibility, but
            # do not invent ledger fill economics from limit_price.
            await self._publish_order_event(
                EventType.ORDER_FILLED if final else EventType.ORDER_PARTIALLY_FILLED,
                client_order_id,
                self._now(),
                extra={"broker_order_id": order.order_id, "fill_price_available": False},
            )
            return []

        qty = fill_qty if fill_qty is not None else Decimal(order.quantity)
        event_type = LedgerEventType.FINAL_FILL if final else LedgerEventType.PARTIAL_FILL
        domain_type = EventType.ORDER_FILLED if final else EventType.ORDER_PARTIALLY_FILLED
        now = self._now()
        cid = contract_id_for(order.contract)
        event = make_ledger_event(
            event_type,
            broker_account_id=self._broker_account_id,
            client_order_id=client_order_id,
            contract_id=cid,
            side=order.side,
            quantity=qty,
            exchange_timestamp=now,
            idempotency_key=f"fill:{client_order_id}:{order.order_id}:{event_type.value}:{qty}",
            session_id=self._session_id,
            broker_order_id=order.order_id,
            price=fill_price,
        )
        written: list[LedgerEvent] = []
        if await self._ledger.append(event):
            written.append(event)
            await self._publish_order_event(
                domain_type,
                client_order_id,
                now,
                extra={"broker_order_id": order.order_id, "price": str(fill_price), "qty": str(qty)},
            )
            if final and order.side == "buy":
                pos_event = make_ledger_event(
                    LedgerEventType.POSITION_OPENED,
                    broker_account_id=self._broker_account_id,
                    client_order_id=client_order_id,
                    contract_id=cid,
                    side=order.side,
                    quantity=qty,
                    exchange_timestamp=now,
                    idempotency_key=f"pos-open:{client_order_id}:{order.order_id}",
                    session_id=self._session_id,
                    broker_order_id=order.order_id,
                    price=fill_price,
                )
                if await self._ledger.append(pos_event):
                    written.append(pos_event)
                    await self._publish_order_event(EventType.POSITION_OPENED, client_order_id, now)
            elif final and order.side == "sell":
                pos_event = make_ledger_event(
                    LedgerEventType.POSITION_CLOSED,
                    broker_account_id=self._broker_account_id,
                    client_order_id=client_order_id,
                    contract_id=cid,
                    side=order.side,
                    quantity=qty,
                    exchange_timestamp=now,
                    idempotency_key=f"pos-close:{client_order_id}:{order.order_id}",
                    session_id=self._session_id,
                    broker_order_id=order.order_id,
                    price=fill_price,
                )
                if await self._ledger.append(pos_event):
                    written.append(pos_event)
                    await self._publish_order_event(EventType.POSITION_CLOSED, client_order_id, now)
        return written

    async def record_verified_fill(
        self,
        order: BrokerOrder,
        *,
        client_order_id: str,
        fill_price: Decimal,
        fill_qty: Decimal | None = None,
        final: bool = True,
    ) -> list[LedgerEvent]:
        """Record a fill with a verified price (never derived from limit)."""
        return await self._record_fill_from_order(
            order,
            client_order_id,
            final=final,
            fill_price=fill_price,
            fill_qty=fill_qty,
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
                status=o.status,
            )
            for o in self._broker.list_open_orders()
        ]
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
            logger.info(
                "reconciliation_consistent",
                extra={
                    "session_id": self._session_id,
                    "report_id": str(report.report_id),
                },
            )
        return report

    def get_daily_pnl(self) -> DailyPnlView:
        """Return broker daily P&L with availability — never invent zeros."""
        available, value = self._broker.get_daily_pnl_available()
        return DailyPnlView(available=available, value=value)

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

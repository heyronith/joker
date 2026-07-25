"""Broker reconciliation: compare projected ledger state to broker open state."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Sequence
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from joker.ledger.exceptions import ReconciliationError
from joker.ledger.projector import OrderLifecycle, OrderStatus, ProjectionState
from joker.ledger.schemas import LedgerEvent, LedgerEventType, Side, make_ledger_event


class ReconciliationOutcome(StrEnum):
    """Typed reconciliation finding outcomes."""

    CONSISTENT = "consistent"
    MISSING_BROKER_ORDER = "missing_broker_order"
    UNKNOWN_BROKER_ORDER = "unknown_broker_order"
    QUANTITY_MISMATCH = "quantity_mismatch"
    STATUS_MISMATCH = "status_mismatch"
    MISSING_BROKER_POSITION = "missing_broker_position"
    UNEXPECTED_BROKER_POSITION = "unexpected_broker_position"


class BrokerOpenOrderView(BaseModel):
    """Broker-reported open (or working) order snapshot for reconciliation."""

    model_config = ConfigDict(frozen=True)

    broker_order_id: str
    client_order_id: str | None = None
    contract_id: str
    side: Side
    quantity: Decimal
    filled_qty: Decimal = Decimal("0")
    status: str


class BrokerPositionView(BaseModel):
    """Broker-reported position snapshot for reconciliation."""

    model_config = ConfigDict(frozen=True)

    contract_id: str
    quantity: Decimal
    avg_price: Decimal | None = None


class ReconciliationFinding(BaseModel):
    """Single typed mismatch or consistent check result."""

    model_config = ConfigDict(frozen=True)

    outcome: ReconciliationOutcome
    entity_type: str
    entity_id: str
    detail: str
    projected_quantity: Decimal | None = None
    broker_quantity: Decimal | None = None
    projected_status: str | None = None
    broker_status: str | None = None


class ReconciliationReport(BaseModel):
    """Full reconciliation report. Does not mutate ledger history."""

    model_config = ConfigDict(frozen=True)

    report_id: UUID = Field(default_factory=uuid4)
    session_id: str
    exchange_timestamp: datetime
    findings: tuple[ReconciliationFinding, ...] = ()
    is_consistent: bool = True

    def correction_events(
        self,
        *,
        broker_account_id: str,
        source_event_id: UUID | None = None,
    ) -> list[LedgerEvent]:
        """Build RECONCILIATION_CORRECTION events for non-consistent findings.

        Caller decides whether to append via SqliteLedgerStore.append.
        History is never overwritten in place.
        """
        corrections: list[LedgerEvent] = []
        for finding in self.findings:
            if finding.outcome == ReconciliationOutcome.CONSISTENT:
                continue
            side: Side = "buy"
            qty = finding.broker_quantity or finding.projected_quantity or Decimal("0")
            if qty < 0:
                side = "sell"
                qty = abs(qty)
            client_order_id = (
                finding.entity_id
                if finding.entity_type == "order"
                else f"recon-{finding.entity_id}"
            )
            contract_id = finding.entity_id
            meta: dict = {
                "correction_kind": _correction_kind(finding.outcome),
                "outcome": finding.outcome.value,
                "entity_type": finding.entity_type,
                "entity_id": finding.entity_id,
                "detail": finding.detail,
            }
            if finding.broker_status is not None:
                meta["status"] = finding.broker_status
            if finding.entity_type == "position" and finding.broker_quantity is not None:
                meta["absolute_quantity"] = str(finding.broker_quantity)
            corrections.append(
                make_ledger_event(
                    LedgerEventType.RECONCILIATION_CORRECTION,
                    broker_account_id=broker_account_id,
                    client_order_id=client_order_id,
                    contract_id=contract_id,
                    side=side,
                    quantity=qty,
                    exchange_timestamp=self.exchange_timestamp,
                    idempotency_key=(
                        f"recon:{self.report_id}:{finding.outcome.value}:{finding.entity_id}"
                    ),
                    session_id=self.session_id,
                    source_event_id=source_event_id,
                    metadata=meta,
                )
            )
        return corrections


def _correction_kind(outcome: ReconciliationOutcome) -> str:
    if outcome in {
        ReconciliationOutcome.STATUS_MISMATCH,
        ReconciliationOutcome.MISSING_BROKER_ORDER,
        ReconciliationOutcome.UNKNOWN_BROKER_ORDER,
    }:
        return "order_status"
    if outcome == ReconciliationOutcome.QUANTITY_MISMATCH:
        return "order_fill_qty"
    return "position_quantity"


_ACTIVE_ORDER_STATUSES = frozenset(
    {
        OrderStatus.SUBMITTED,
        OrderStatus.ACCEPTED,
        OrderStatus.PARTIALLY_FILLED,
    }
)

_STATUS_ALIASES: dict[str, set[str]] = {
    "submitted": {"submitted", "pending", "new", "open"},
    "accepted": {"accepted", "working", "open", "new", "partially_filled"},
    "partially_filled": {"partially_filled", "partial", "open", "working"},
    "filled": {"filled", "complete", "completed"},
    "cancelled": {"cancelled", "canceled"},
    "rejected": {"rejected"},
}


def _statuses_compatible(projected: OrderStatus, broker_status: str) -> bool:
    aliases = _STATUS_ALIASES.get(projected.value, {projected.value})
    return broker_status.strip().lower() in aliases


class BrokerReconciler:
    """Compare projected ledger state against broker open orders and positions.

    Never silently overwrites history. Returns a report; corrections are explicit
    LedgerEvent values the caller may append.
    """

    def reconcile(
        self,
        *,
        session_id: str,
        projection: ProjectionState,
        broker_orders: Sequence[BrokerOpenOrderView],
        broker_positions: Sequence[BrokerPositionView],
        exchange_timestamp: datetime | None = None,
    ) -> ReconciliationReport:
        if not session_id.strip():
            raise ReconciliationError("session_id is required")

        ts = exchange_timestamp or datetime.now(timezone.utc)
        if ts.tzinfo is None or ts.utcoffset() is None:
            raise ReconciliationError("exchange_timestamp must be timezone-aware")

        findings: list[ReconciliationFinding] = []

        projected_active = {
            oid: order
            for oid, order in projection.orders.items()
            if order.status in _ACTIVE_ORDER_STATUSES
        }
        broker_by_client: dict[str, BrokerOpenOrderView] = {}
        broker_by_broker_id: dict[str, BrokerOpenOrderView] = {}
        for bo in broker_orders:
            broker_by_broker_id[bo.broker_order_id] = bo
            if bo.client_order_id:
                broker_by_client[bo.client_order_id] = bo

        matched_broker_ids: set[str] = set()

        for client_order_id, order in projected_active.items():
            broker_order = broker_by_client.get(client_order_id)
            if broker_order is None and order.broker_order_id:
                broker_order = broker_by_broker_id.get(order.broker_order_id)
            if broker_order is None:
                findings.append(
                    ReconciliationFinding(
                        outcome=ReconciliationOutcome.MISSING_BROKER_ORDER,
                        entity_type="order",
                        entity_id=client_order_id,
                        detail="projected active order not found at broker",
                        projected_quantity=order.submitted_qty - order.filled_qty,
                        projected_status=order.status.value,
                    )
                )
                continue
            matched_broker_ids.add(broker_order.broker_order_id)
            findings.extend(self._compare_order(order, broker_order))

        for bo in broker_orders:
            if bo.broker_order_id in matched_broker_ids:
                continue
            known = False
            if bo.client_order_id and bo.client_order_id in projection.orders:
                known = True
            if not known and any(
                o.broker_order_id == bo.broker_order_id for o in projection.orders.values()
            ):
                known = True
            if not known:
                findings.append(
                    ReconciliationFinding(
                        outcome=ReconciliationOutcome.UNKNOWN_BROKER_ORDER,
                        entity_type="order",
                        entity_id=bo.client_order_id or bo.broker_order_id,
                        detail="broker open order not present in projected ledger",
                        broker_quantity=bo.quantity - bo.filled_qty,
                        broker_status=bo.status,
                    )
                )

        projected_open_positions = {
            cid: pos for cid, pos in projection.positions.items() if pos.open and pos.quantity != 0
        }
        broker_pos_by_contract = {p.contract_id: p for p in broker_positions}

        for contract_id, pos in projected_open_positions.items():
            bp = broker_pos_by_contract.get(contract_id)
            if bp is None:
                findings.append(
                    ReconciliationFinding(
                        outcome=ReconciliationOutcome.MISSING_BROKER_POSITION,
                        entity_type="position",
                        entity_id=contract_id,
                        detail="projected open position missing at broker",
                        projected_quantity=pos.quantity,
                    )
                )
                continue
            if bp.quantity != pos.quantity:
                findings.append(
                    ReconciliationFinding(
                        outcome=ReconciliationOutcome.QUANTITY_MISMATCH,
                        entity_type="position",
                        entity_id=contract_id,
                        detail="position quantity differs between ledger and broker",
                        projected_quantity=pos.quantity,
                        broker_quantity=bp.quantity,
                    )
                )
            else:
                findings.append(
                    ReconciliationFinding(
                        outcome=ReconciliationOutcome.CONSISTENT,
                        entity_type="position",
                        entity_id=contract_id,
                        detail="position quantity matches",
                        projected_quantity=pos.quantity,
                        broker_quantity=bp.quantity,
                    )
                )

        for bp in broker_positions:
            if bp.quantity == 0:
                continue
            if bp.contract_id not in projected_open_positions:
                findings.append(
                    ReconciliationFinding(
                        outcome=ReconciliationOutcome.UNEXPECTED_BROKER_POSITION,
                        entity_type="position",
                        entity_id=bp.contract_id,
                        detail="broker position not open in projected ledger",
                        broker_quantity=bp.quantity,
                    )
                )

        is_consistent = all(f.outcome == ReconciliationOutcome.CONSISTENT for f in findings)
        if not findings:
            is_consistent = True

        return ReconciliationReport(
            session_id=session_id,
            exchange_timestamp=ts,
            findings=tuple(findings),
            is_consistent=is_consistent,
        )

    @staticmethod
    def _compare_order(
        order: OrderLifecycle,
        broker_order: BrokerOpenOrderView,
    ) -> list[ReconciliationFinding]:
        findings: list[ReconciliationFinding] = []
        if order.filled_qty != broker_order.filled_qty:
            findings.append(
                ReconciliationFinding(
                    outcome=ReconciliationOutcome.QUANTITY_MISMATCH,
                    entity_type="order",
                    entity_id=order.client_order_id,
                    detail="filled/remaining quantity mismatch",
                    projected_quantity=order.filled_qty,
                    broker_quantity=broker_order.filled_qty,
                    projected_status=order.status.value,
                    broker_status=broker_order.status,
                )
            )
        if not _statuses_compatible(order.status, broker_order.status):
            findings.append(
                ReconciliationFinding(
                    outcome=ReconciliationOutcome.STATUS_MISMATCH,
                    entity_type="order",
                    entity_id=order.client_order_id,
                    detail="order status mismatch",
                    projected_status=order.status.value,
                    broker_status=broker_order.status,
                    projected_quantity=order.filled_qty,
                    broker_quantity=broker_order.filled_qty,
                )
            )
        if not findings:
            findings.append(
                ReconciliationFinding(
                    outcome=ReconciliationOutcome.CONSISTENT,
                    entity_type="order",
                    entity_id=order.client_order_id,
                    detail="order matches broker",
                    projected_status=order.status.value,
                    broker_status=broker_order.status,
                    projected_quantity=order.filled_qty,
                    broker_quantity=broker_order.filled_qty,
                )
            )
        return findings

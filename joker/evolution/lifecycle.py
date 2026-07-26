"""Exact position-lifecycle resolution for episode compilation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from joker.ledger.projector import OrderLifecycle, OrderStatus, ProjectionState


class _ProvenanceLookup(Protocol):
    async def get_by_client_order_id(self, client_order_id: str) -> Any: ...

    async def get_latest_by_contract_id(self, contract_id: str) -> Any: ...


class ResolvedPositionLifecycle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    position_lifecycle_id: str
    entry_cycle_id: str | None = None
    configuration_version_id: UUID | None = None
    entry_orders: tuple[OrderLifecycle, ...] = ()
    replacement_orders: tuple[OrderLifecycle, ...] = ()
    reduction_orders: tuple[OrderLifecycle, ...] = ()
    exit_orders: tuple[OrderLifecycle, ...] = ()
    fill_event_ids: tuple[UUID, ...] = ()
    position_event_ids: tuple[UUID, ...] = ()
    initial_snapshot_id: UUID | None = None
    terminal_snapshot_id: UUID | None = None
    original_strategy_id: UUID | None = None
    proposal_id: UUID | None = None
    decision_id: UUID | None = None
    realised_pnl: Decimal | None = None
    total_fees: Decimal = Decimal("0")
    quantity: Decimal = Decimal("0")
    findings: tuple[str, ...] = ()


@dataclass
class PositionLifecycleResolver:
    """Resolve a closed trade to one originating entry lifecycle, not all contract orders."""

    provenance: _ProvenanceLookup | None = None

    async def resolve_closed_lifecycle(
        self,
        *,
        session_id: str,
        terminal_event_id: str,
        contract_id: str,
        client_order_id: str | None,
        projection: ProjectionState,
        configuration_version_id: UUID | None = None,
        known_entry_cycle_id: str | None = None,
        known_snapshot_id: UUID | None = None,
    ) -> ResolvedPositionLifecycle:
        findings: list[str] = []
        entry_cycle_id = known_entry_cycle_id
        snapshot_id = known_snapshot_id
        proposal_id = None
        decision_id = None
        strategy_id = None
        originating_entry_id: str | None = None

        if self.provenance is not None:
            record = None
            if client_order_id:
                record = await self.provenance.get_by_client_order_id(client_order_id)
            if record is None and contract_id:
                record = await self.provenance.get_latest_by_contract_id(contract_id)
            if record is not None:
                entry_cycle_id = entry_cycle_id or getattr(record, "cycle_id", None)
                if getattr(record, "snapshot_id", None):
                    try:
                        snapshot_id = snapshot_id or UUID(str(record.snapshot_id))
                    except Exception:
                        findings.append("invalid_provenance_snapshot_id")
                for attr, target in (
                    ("proposal_id", "proposal_id"),
                    ("decision_id", "decision_id"),
                    ("strategy_id", "strategy_id"),
                ):
                    raw = getattr(record, attr, None)
                    if raw:
                        try:
                            if target == "proposal_id":
                                proposal_id = UUID(str(raw))
                            elif target == "decision_id":
                                decision_id = UUID(str(raw))
                            else:
                                strategy_id = UUID(str(raw))
                        except Exception:
                            pass
                if getattr(record, "kind", None) == "entry":
                    originating_entry_id = getattr(record, "client_order_id", None)
                elif getattr(record, "parent_client_order_id", None):
                    originating_entry_id = str(record.parent_client_order_id)

        filled = [
            o
            for o in projection.orders.values()
            if o.contract_id == contract_id
            and o.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
        ]
        # Stable chronological order by client_order_id (paper ids are sequential).
        filled_sorted = sorted(filled, key=lambda o: o.client_order_id)
        buys = [o for o in filled_sorted if o.side == "buy"]
        sells = [o for o in filled_sorted if o.side == "sell"]

        exit_order: OrderLifecycle | None = None
        if client_order_id and client_order_id in projection.orders:
            exit_order = projection.orders[client_order_id]
        elif sells:
            exit_order = sells[-1]

        entry_order: OrderLifecycle | None = None
        if originating_entry_id and originating_entry_id in projection.orders:
            entry_order = projection.orders[originating_entry_id]
        elif exit_order is not None:
            # Pair this exit with the unmatched buy immediately preceding it.
            prior_buys = [
                b for b in buys if b.client_order_id <= exit_order.client_order_id
            ]
            prior_sells = [
                s
                for s in sells
                if s.client_order_id < exit_order.client_order_id
            ]
            # Consume buys with earlier sells so we isolate this round-trip.
            remaining = list(prior_buys)
            for sell in prior_sells:
                if remaining:
                    remaining.pop(0)
            entry_order = remaining[0] if remaining else (buys[-1] if buys else None)
        elif buys:
            entry_order = buys[-1]

        if entry_order is None:
            findings.append("missing_entry_order_for_lifecycle")

        # Include replacement ancestry hanging off the entry.
        replacements: list[OrderLifecycle] = []
        reductions: list[OrderLifecycle] = []
        if entry_order is not None:
            for order in filled_sorted:
                parent = getattr(order, "parent_order_id", None) or getattr(
                    order, "replaced_order_id", None
                )
                if parent == entry_order.client_order_id and order.side == "buy":
                    replacements.append(order)
                if (
                    parent == entry_order.client_order_id
                    and order.side == "sell"
                    and (exit_order is None or order.client_order_id != exit_order.client_order_id)
                ):
                    reductions.append(order)

        exit_orders = (exit_order,) if exit_order is not None else ()
        entry_orders = (entry_order,) if entry_order is not None else ()
        if replacements:
            entry_orders = (*entry_orders, *replacements)

        entry_id = entry_order.client_order_id if entry_order is not None else (
            client_order_id or terminal_event_id
        )
        lifecycle_id = f"{session_id}:{entry_id}:{contract_id}"

        entry_qty = sum((o.filled_qty for o in entry_orders), Decimal("0"))
        exit_qty = sum(
            (o.filled_qty for o in (*reductions, *exit_orders)), Decimal("0")
        )
        fees = sum(
            (o.fees for o in (*entry_orders, *reductions, *exit_orders)),
            Decimal("0"),
        )
        if entry_qty != exit_qty:
            findings.append("quantity_identity_mismatch")

        realised = None
        entry_px = _vwap(entry_orders)
        exit_px = _vwap(tuple((*reductions, *exit_orders)))
        if entry_px is not None and exit_px is not None and entry_qty > 0:
            realised = ((exit_px - entry_px) * entry_qty * Decimal("100")) - fees

        if snapshot_id is None:
            findings.append("missing_initial_snapshot")

        return ResolvedPositionLifecycle(
            position_lifecycle_id=lifecycle_id,
            entry_cycle_id=entry_cycle_id,
            configuration_version_id=configuration_version_id,
            entry_orders=entry_orders,
            replacement_orders=tuple(replacements),
            reduction_orders=tuple(reductions),
            exit_orders=exit_orders,
            initial_snapshot_id=snapshot_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            original_strategy_id=strategy_id,
            realised_pnl=realised,
            total_fees=fees,
            quantity=entry_qty,
            findings=tuple(dict.fromkeys(findings)),
        )


def _vwap(orders: tuple[OrderLifecycle, ...]) -> Decimal | None:
    qty = Decimal("0")
    notional = Decimal("0")
    for order in orders:
        if order.avg_fill_price is None or order.filled_qty <= 0:
            continue
        qty += order.filled_qty
        notional += order.avg_fill_price * order.filled_qty
    if qty <= 0:
        return None
    return (notional / qty).quantize(Decimal("0.0001"))

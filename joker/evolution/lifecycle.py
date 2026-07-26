"""Exact position-lifecycle resolution for episode compilation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from joker.evolution.lifecycle_id import make_position_lifecycle_id
from joker.ledger.projector import OrderLifecycle, OrderStatus, ProjectionState


class _ProvenanceLookup(Protocol):
    async def get_by_client_order_id(self, client_order_id: str) -> Any: ...

    async def get_latest_by_contract_id(self, contract_id: str) -> Any: ...

    async def list_by_lifecycle_id(self, position_lifecycle_id: str) -> list[Any]: ...


class ResolvedPositionLifecycle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    position_lifecycle_id: str
    entry_cycle_id: str | None = None
    configuration_version_id: UUID | None = None
    originating_entry_client_order_id: str | None = None
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
    remaining_quantity: Decimal = Decimal("0")
    findings: tuple[str, ...] = ()
    legacy_inferred: bool = False


@dataclass
class PositionLifecycleResolver:
    """Resolve closed trades by persisted lifecycle ID, not broker-ID ordering."""

    provenance: _ProvenanceLookup | None = None

    async def resolve_closed_lifecycle(
        self,
        *,
        session_id: str,
        terminal_event_id: str,
        projection: ProjectionState,
        position_lifecycle_id: str | None = None,
        contract_id: str | None = None,
        client_order_id: str | None = None,
        configuration_version_id: UUID | None = None,
        known_entry_cycle_id: str | None = None,
        known_snapshot_id: UUID | None = None,
        allow_legacy_inference: bool = True,
    ) -> ResolvedPositionLifecycle:
        findings: list[str] = []
        legacy = False
        lifecycle_id = position_lifecycle_id
        entry_cycle_id = known_entry_cycle_id
        snapshot_id = known_snapshot_id
        proposal_id = None
        decision_id = None
        strategy_id = None
        originating_entry_id: str | None = None

        if self.provenance is not None and client_order_id:
            record = await self.provenance.get_by_client_order_id(client_order_id)
            if record is not None:
                extra = getattr(record, "extra", None) or {}
                lifecycle_id = lifecycle_id or extra.get("position_lifecycle_id")
                originating_entry_id = extra.get("originating_entry_client_order_id")
                entry_cycle_id = entry_cycle_id or getattr(record, "cycle_id", None)
                if getattr(record, "snapshot_id", None) and snapshot_id is None:
                    try:
                        snapshot_id = UUID(str(record.snapshot_id))
                    except Exception:
                        findings.append("invalid_provenance_snapshot_id")
                for attr, setter in (
                    ("proposal_id", "proposal"),
                    ("decision_id", "decision"),
                    ("strategy_id", "strategy"),
                ):
                    raw = getattr(record, attr, None)
                    if not raw:
                        continue
                    try:
                        value = UUID(str(raw))
                    except Exception:
                        continue
                    if setter == "proposal":
                        proposal_id = value
                    elif setter == "decision":
                        decision_id = value
                    else:
                        strategy_id = value
                if getattr(record, "kind", None) == "entry":
                    originating_entry_id = originating_entry_id or record.client_order_id
                contract_id = contract_id or getattr(record, "contract_id", None)

        # Prefer lifecycle stamped on the terminal order itself.
        if (
            not lifecycle_id
            and client_order_id
            and client_order_id in projection.orders
        ):
            terminal = projection.orders[client_order_id]
            lifecycle_id = getattr(terminal, "position_lifecycle_id", None)
            originating_entry_id = originating_entry_id or getattr(
                terminal, "originating_entry_client_order_id", None
            )

        orders_by_lifecycle: list[OrderLifecycle] = []
        if lifecycle_id:
            for order in projection.orders.values():
                oid = getattr(order, "position_lifecycle_id", None)
                if oid == lifecycle_id:
                    orders_by_lifecycle.append(order)
            if self.provenance is not None and hasattr(
                self.provenance, "list_by_lifecycle_id"
            ):
                try:
                    records = await self.provenance.list_by_lifecycle_id(lifecycle_id)
                except Exception:
                    records = []
                for record in records:
                    order = projection.orders.get(record.client_order_id)
                    if order is not None and order not in orders_by_lifecycle:
                        orders_by_lifecycle.append(order)

        if not orders_by_lifecycle and allow_legacy_inference:
            legacy = True
            findings.append("legacy_lifecycle_inference")
            return await self._legacy_infer(
                session_id=session_id,
                terminal_event_id=terminal_event_id,
                contract_id=contract_id or "",
                client_order_id=client_order_id,
                projection=projection,
                configuration_version_id=configuration_version_id,
                known_entry_cycle_id=entry_cycle_id,
                known_snapshot_id=snapshot_id,
                proposal_id=proposal_id,
                decision_id=decision_id,
                strategy_id=strategy_id,
                originating_entry_id=originating_entry_id,
                findings=findings,
            )

        if not lifecycle_id:
            findings.append("missing_position_lifecycle_id")
            lifecycle_id = make_position_lifecycle_id(
                session_id=session_id,
                originating_entry_client_order_id=originating_entry_id
                or client_order_id
                or terminal_event_id,
                contract_id=contract_id or "unknown",
            )

        filled = [
            o
            for o in orders_by_lifecycle
            if o.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
        ]
        buys = [o for o in filled if o.side == "buy"]
        sells = [o for o in filled if o.side == "sell"]
        replacements = [
            o
            for o in buys
            if getattr(o, "parent_client_order_id", None)
            and getattr(o, "parent_client_order_id", None) != originating_entry_id
        ]
        entry_orders = tuple(
            o
            for o in buys
            if not getattr(o, "parent_client_order_id", None)
            or getattr(o, "originating_entry_client_order_id", None)
            == (originating_entry_id or o.client_order_id)
        ) or tuple(buys)
        if originating_entry_id is None and entry_orders:
            originating_entry_id = entry_orders[0].client_order_id

        exit_orders: list[OrderLifecycle] = []
        reductions: list[OrderLifecycle] = []
        if client_order_id and client_order_id in projection.orders:
            terminal = projection.orders[client_order_id]
            exit_orders = [terminal]
            reductions = [
                s for s in sells if s.client_order_id != terminal.client_order_id
            ]
        else:
            exit_orders = list(sells)

        entry_qty = sum((o.filled_qty for o in entry_orders), Decimal("0"))
        exit_qty = sum(
            (o.filled_qty for o in (*reductions, *exit_orders)), Decimal("0")
        )
        remaining = entry_qty - exit_qty
        fees = sum(
            (o.fees for o in (*entry_orders, *replacements, *reductions, *exit_orders)),
            Decimal("0"),
        )
        if remaining != 0:
            findings.append("quantity_identity_mismatch")

        buy_cost = sum(
            (
                (o.avg_fill_price or Decimal("0")) * o.filled_qty * Decimal("100")
                for o in (*entry_orders, *replacements)
                if o.filled_qty > 0
            ),
            Decimal("0"),
        )
        sell_proceeds = sum(
            (
                (o.avg_fill_price or Decimal("0")) * o.filled_qty * Decimal("100")
                for o in (*reductions, *exit_orders)
                if o.filled_qty > 0
            ),
            Decimal("0"),
        )
        realised = (sell_proceeds - buy_cost - fees) if entry_qty > 0 else None
        if snapshot_id is None:
            findings.append("missing_initial_snapshot")

        return ResolvedPositionLifecycle(
            position_lifecycle_id=lifecycle_id,
            entry_cycle_id=entry_cycle_id,
            configuration_version_id=configuration_version_id,
            originating_entry_client_order_id=originating_entry_id,
            entry_orders=tuple(entry_orders),
            replacement_orders=tuple(replacements),
            reduction_orders=tuple(reductions),
            exit_orders=tuple(exit_orders),
            initial_snapshot_id=snapshot_id,
            proposal_id=proposal_id,
            decision_id=decision_id,
            original_strategy_id=strategy_id,
            realised_pnl=realised,
            total_fees=fees,
            quantity=entry_qty,
            remaining_quantity=remaining,
            findings=tuple(dict.fromkeys(findings)),
            legacy_inferred=legacy,
        )

    async def _legacy_infer(self, **kwargs: Any) -> ResolvedPositionLifecycle:
        """Legacy fallback — marks episode incomplete for promotion exclusion."""
        session_id = kwargs["session_id"]
        contract_id = kwargs["contract_id"]
        client_order_id = kwargs["client_order_id"]
        projection: ProjectionState = kwargs["projection"]
        findings: list[str] = list(kwargs["findings"])
        filled = [
            o
            for o in projection.orders.values()
            if o.contract_id == contract_id
            and o.status in {OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED}
        ]
        buys = [o for o in filled if o.side == "buy"]
        sells = [o for o in filled if o.side == "sell"]
        exit_order = None
        if client_order_id and client_order_id in projection.orders:
            exit_order = projection.orders[client_order_id]
        elif sells:
            exit_order = sells[-1]
        entry_order = buys[0] if buys else None
        entry_id = (
            entry_order.client_order_id
            if entry_order is not None
            else (client_order_id or kwargs["terminal_event_id"])
        )
        lifecycle_id = make_position_lifecycle_id(
            session_id=session_id,
            originating_entry_client_order_id=entry_id,
            contract_id=contract_id or "unknown",
        )
        entry_orders = (entry_order,) if entry_order is not None else ()
        exit_orders = (exit_order,) if exit_order is not None else ()
        entry_qty = sum((o.filled_qty for o in entry_orders), Decimal("0"))
        exit_qty = sum((o.filled_qty for o in exit_orders), Decimal("0"))
        fees = sum((o.fees for o in (*entry_orders, *exit_orders)), Decimal("0"))
        buy_cost = sum(
            (
                (o.avg_fill_price or Decimal("0")) * o.filled_qty * Decimal("100")
                for o in entry_orders
            ),
            Decimal("0"),
        )
        sell_proceeds = sum(
            (
                (o.avg_fill_price or Decimal("0")) * o.filled_qty * Decimal("100")
                for o in exit_orders
            ),
            Decimal("0"),
        )
        realised = (sell_proceeds - buy_cost - fees) if entry_qty > 0 else None
        if kwargs["known_snapshot_id"] is None:
            findings.append("missing_initial_snapshot")
        if entry_qty != exit_qty:
            findings.append("quantity_identity_mismatch")
        return ResolvedPositionLifecycle(
            position_lifecycle_id=lifecycle_id,
            entry_cycle_id=kwargs["known_entry_cycle_id"],
            configuration_version_id=kwargs["configuration_version_id"],
            originating_entry_client_order_id=entry_id,
            entry_orders=entry_orders,
            exit_orders=exit_orders,
            initial_snapshot_id=kwargs["known_snapshot_id"],
            proposal_id=kwargs["proposal_id"],
            decision_id=kwargs["decision_id"],
            original_strategy_id=kwargs["strategy_id"],
            realised_pnl=realised,
            total_fees=fees,
            quantity=entry_qty,
            remaining_quantity=entry_qty - exit_qty,
            findings=tuple(dict.fromkeys(findings)),
            legacy_inferred=True,
        )

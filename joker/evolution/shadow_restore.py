"""Restore durable shadow execution into isolated replay runtimes."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.evolution.replay_execution import (
    ReplayExecutionRuntime,
    ReplayFill,
    ReplayOrder,
    ReplayPosition,
)
from joker.evolution.replay_market import ReplayEpisodeTruth, ReplayPositionSeed
from joker.evolution.replay_position_runtime import ReplayPositionRuntime
from joker.evolution.shadow_ledger import ShadowLedger


class RestoredShadowRuntime:
    def __init__(
        self,
        *,
        assignment_id: UUID,
        challenger_version_id: UUID,
        position_runtime: ReplayPositionRuntime,
        last_snapshot_id: str | None,
        submitted_keys: set[str],
        cash: Decimal,
        realised_pnl: Decimal,
        cursor: dict[str, Any] | None = None,
    ) -> None:
        self.assignment_id = assignment_id
        self.challenger_version_id = challenger_version_id
        self.position_runtime = position_runtime
        self.last_snapshot_id = last_snapshot_id
        self.submitted_keys = submitted_keys
        self.cash = cash
        self.realised_pnl = realised_pnl
        self.cursor = cursor or {}


class ShadowExecutionRestorer:
    def __init__(self, ledger: ShadowLedger) -> None:
        self._ledger = ledger

    async def restore_assignment(
        self,
        assignment: Any,
        *,
        truth: ReplayEpisodeTruth | None = None,
    ) -> RestoredShadowRuntime:
        await self._ledger.initialize()
        assignment_id = assignment.assignment_id
        challenger_id = assignment.challenger_version_id
        checkpoint = await self._ledger.load_checkpoint(assignment_id)
        cursor = (checkpoint or {}).get("cursor") or {}
        last_snapshot = (checkpoint or {}).get("last_snapshot_id")
        cash = Decimal(str(cursor.get("cash", truth.starting_cash if truth else "25000")))
        realised = Decimal(str(cursor.get("realised_pnl", "0")))
        submitted = set(cursor.get("submitted_keys") or [])

        if truth is None:
            truth = ReplayEpisodeTruth(
                episode_id=uuid4(),
                session_id=f"shadow-restore:{assignment_id}",
                initial_snapshot_id=uuid4(),
                starting_cash=cash,
                starting_positions=(),
            )
        execution = ReplayExecutionRuntime(truth=truth)
        orders: dict[str, ReplayOrder] = {}
        for raw in await self._ledger.list_orders(assignment_id):
            payload = raw.get("payload") or {}
            orders[raw["client_order_id"]] = ReplayOrder(
                client_order_id=raw["client_order_id"],
                contract_id=raw["contract_id"],
                side=raw["side"],
                quantity=Decimal(str(raw["quantity"])),
                limit_price=(
                    Decimal(str(payload["limit_price"]))
                    if payload.get("limit_price") is not None
                    else None
                ),
                status=raw["status"],
                filled_qty=Decimal(str(payload.get("filled_qty", "0"))),
                avg_fill_price=(
                    Decimal(str(payload["avg_fill_price"]))
                    if payload.get("avg_fill_price") is not None
                    else None
                ),
                fees=Decimal(str(payload.get("fees", "0"))),
                parent_order_id=payload.get("parent_order_id"),
            )
        fills: list[ReplayFill] = []
        for raw in await self._ledger.list_fills(assignment_id):
            payload = raw.get("payload") or {}
            fills.append(
                ReplayFill(
                    fill_id=raw["fill_id"],
                    client_order_id=raw["client_order_id"],
                    contract_id=str(payload.get("contract_id") or ""),
                    side=str(payload.get("side") or "buy"),
                    quantity=Decimal(str(raw["quantity"])),
                    price=Decimal(str(raw["price"])),
                    fees=Decimal(str(raw["fee"])),
                )
            )
        positions: dict[str, ReplayPosition] = {}
        for raw in await self._ledger.list_open_positions(assignment_id):
            positions[raw["contract_id"]] = ReplayPosition(
                contract_id=raw["contract_id"],
                quantity=Decimal(str(raw["quantity"])),
                avg_price=Decimal(str(raw["average_price"])),
                realised_pnl=Decimal(str(raw["realised_pnl"])),
                configuration_version_id=UUID(str(raw["configuration_version_id"])),
                position_lifecycle_id=raw["position_lifecycle_id"],
            )
            # Also include closed positions so restart does not recreate them.
        for raw in await self._ledger.list_positions(assignment_id, status="closed"):
            if raw["contract_id"] in positions:
                continue
            positions[raw["contract_id"]] = ReplayPosition(
                contract_id=raw["contract_id"],
                quantity=Decimal("0"),
                avg_price=Decimal(str(raw["average_price"])),
                realised_pnl=Decimal(str(raw["realised_pnl"])),
                configuration_version_id=UUID(str(raw["configuration_version_id"])),
                position_lifecycle_id=raw["position_lifecycle_id"],
            )

        execution.restore_state(
            cash=cash,
            orders=orders,
            positions=positions,
            fills=fills,
            submitted_keys=submitted,
        )
        runtime = ReplayPositionRuntime(
            execution=execution,
            configuration_version_id=challenger_id,
        )
        if any(p.quantity > 0 for p in positions.values()):
            runtime.traded = True
            open_cid = next(cid for cid, p in positions.items() if p.quantity > 0)
            runtime.selected_contract_id = open_cid
            runtime.mark("restored_open_position")
        return RestoredShadowRuntime(
            assignment_id=assignment_id,
            challenger_version_id=challenger_id,
            position_runtime=runtime,
            last_snapshot_id=last_snapshot,
            submitted_keys=submitted,
            cash=cash,
            realised_pnl=realised,
            cursor=cursor if isinstance(cursor, dict) else {},
        )

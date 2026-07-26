"""Replay/shadow position cognition against isolated ReplayExecutionRuntime."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.evolution.replay_execution import ReplayExecutionRuntime


@dataclass
class ReplayPositionRuntime:
    """Drive entry/exit fills for one configuration sample without live broker."""

    execution: ReplayExecutionRuntime
    configuration_version_id: UUID
    stage: str = "truth_loaded"
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    selected_contract_id: str | None = None
    direction: str = "none"
    traded: bool = False
    open_at_end: bool = False
    mfe: Decimal = Decimal("0")
    mae: Decimal = Decimal("0")
    safety_findings: list[str] = field(default_factory=list)
    model_call_ids: list[str] = field(default_factory=list)
    replay_event_ids: list[str] = field(default_factory=list)

    def mark(self, stage: str) -> None:
        self.stage = stage

    def simulate_entry_from_meta(
        self,
        *,
        action: str,
        contract_id: str | None,
        quantity: Decimal = Decimal("1"),
        bid: Decimal | None = None,
        ask: Decimal | None = None,
        fill_fraction: Decimal = Decimal("1"),
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if self.stage not in {"truth_loaded", "entry_graph_completed"}:
            return {"skipped": True, "stage": self.stage}
        self.mark("entry_graph_completed")
        if action not in {"execute", "probe", "EXECUTE", "PROBE"} or not contract_id:
            self.traded = False
            self.mark("replay_finalised")
            return {"traded": False, "action": action}
        if bid is not None and ask is not None:
            self.execution.allow_contract(contract_id, bid=bid, ask=ask)
        try:
            order_id = f"replay-entry:{self.configuration_version_id}:{contract_id}"
            lifecycle = f"replay:{self.configuration_version_id}:{contract_id}"
            order = self.execution.submit_order(
                client_order_id=order_id,
                contract_id=contract_id,
                side="buy",
                quantity=quantity,
                fill_fraction=fill_fraction,
                idempotency_key=idempotency_key or order_id,
                configuration_version_id=self.configuration_version_id,
                lifecycle_id=lifecycle,
            )
        except Exception as exc:  # noqa: BLE001
            self.safety_findings.append(str(exc))
            self.traded = False
            self.mark("replay_finalised")
            return {"traded": False, "error": str(exc)}
        self.entry_order_id = order.client_order_id
        self.selected_contract_id = contract_id
        self.direction = "long"
        self.traded = order.filled_qty > 0
        self.replay_event_ids.append(str(uuid4()))
        self.mark("entry_order_simulated")
        if self.traded:
            self.mark("position_graph_started")
        else:
            self.mark("replay_finalised")
        return {
            "traded": self.traded,
            "order_status": order.status,
            "fill_price": str(order.avg_fill_price) if order.avg_fill_price else None,
            "filled_qty": str(order.filled_qty),
        }

    def simulate_exit(
        self,
        *,
        bid: Decimal | None = None,
        ask: Decimal | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        if not self.traded or not self.selected_contract_id:
            self.mark("replay_finalised")
            return {"exited": False}
        if self.stage in {"position_graph_completed", "replay_finalised"}:
            return {"exited": True, "idempotent": True}
        pos = self.execution.positions.get(self.selected_contract_id)
        if pos is None or pos.quantity <= 0:
            self.open_at_end = False
            self.mark("replay_finalised")
            return {"exited": False, "already_flat": True}
        if bid is not None and ask is not None:
            self.execution.allow_contract(self.selected_contract_id, bid=bid, ask=ask)
        order_id = f"replay-exit:{self.configuration_version_id}:{self.selected_contract_id}"
        order = self.execution.submit_order(
            client_order_id=order_id,
            contract_id=self.selected_contract_id,
            side="sell",
            quantity=pos.quantity,
            idempotency_key=idempotency_key or order_id,
            configuration_version_id=self.configuration_version_id,
            lifecycle_id=pos.position_lifecycle_id,
        )
        self.exit_order_id = order.client_order_id
        self.open_at_end = (
            self.execution.positions.get(self.selected_contract_id, pos).quantity > 0
        )
        self.replay_event_ids.append(str(uuid4()))
        self.mark("position_graph_completed")
        self.mark("replay_finalised")
        return {
            "exited": order.filled_qty > 0,
            "exit_price": str(order.avg_fill_price) if order.avg_fill_price else None,
            "realised_pnl": str(self.execution.realised_pnl()),
        }

    def outcome_payload(self) -> dict[str, Any]:
        entry = (
            self.execution.orders.get(self.entry_order_id)
            if self.entry_order_id
            else None
        )
        exit_order = (
            self.execution.orders.get(self.exit_order_id) if self.exit_order_id else None
        )
        return {
            "realised_pnl": str(self.execution.realised_pnl()),
            "selected_contract": self.selected_contract_id,
            "direction": self.direction,
            "entry_action": "execute" if self.traded else "no_trade",
            "entry_order": self.entry_order_id,
            "fill_price": str(entry.avg_fill_price) if entry and entry.avg_fill_price else None,
            "quantity": str(entry.filled_qty) if entry else "0",
            "exit_price": (
                str(exit_order.avg_fill_price)
                if exit_order and exit_order.avg_fill_price
                else None
            ),
            "mfe": str(self.mfe),
            "mae": str(self.mae),
            "model_call_ids": list(self.model_call_ids),
            "configuration_version_id": str(self.configuration_version_id),
            "replay_event_ids": list(self.replay_event_ids),
            "safety_findings": list(self.safety_findings),
            "traded": self.traded,
            "open_at_end": self.open_at_end,
            "stage": self.stage,
            "projection": self.execution.projection(),
            "broker_submit": False,
            "execution_runtime": False,
            "fill_model_version": self.execution.truth.fill_model_version,
            "random_seed": self.execution.truth.random_seed,
        }

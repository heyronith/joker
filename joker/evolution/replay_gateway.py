"""Isolated OrderActionGateway backed by ReplayExecutionRuntime."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import uuid4

from joker.evolution.lifecycle_id import make_position_lifecycle_id
from joker.evolution.replay_execution import ReplayExecutionError, ReplayExecutionRuntime
from joker.runtime.order_action_gateway import (
    OrderActionKind,
    OrderActionRequest,
    OrderActionResult,
)


@dataclass
class ReplayOrderActionGateway:
    """Same action-request contract as production; no broker / ExecutionRuntime."""

    execution: ReplayExecutionRuntime
    session_id: str
    configuration_version_id: str | None = None

    async def submit(self, request: OrderActionRequest) -> OrderActionResult:
        try:
            if request.action == OrderActionKind.CANCEL:
                # Represented as cancel of replace parent when provided.
                target = request.replace_of_client_order_id or request.client_order_id
                self.execution.cancel_order(target)
                return OrderActionResult(
                    submitted=True, client_order_id=target, broker_order=None
                )

            side = request.side
            qty = Decimal(str(request.quantity))
            limit = (
                Decimal(str(request.limit_price))
                if request.limit_price is not None
                else None
            )
            lifecycle = request.position_lifecycle_id
            originating = request.originating_entry_client_order_id
            if request.action in {OrderActionKind.ENTRY, OrderActionKind.PROBE}:
                originating = originating or request.client_order_id
                lifecycle = lifecycle or make_position_lifecycle_id(
                    session_id=self.session_id,
                    originating_entry_client_order_id=originating,
                    contract_id=request.contract_id,
                )
                # Ensure contract is on current surface quotes.
                if request.contract_id not in self.execution.quotes:
                    raise ReplayExecutionError(
                        f"contract_not_on_frozen_surface:{request.contract_id}"
                    )

            if request.action == OrderActionKind.REPLACE:
                order = self.execution.replace_order(
                    parent_order_id=request.replace_of_client_order_id
                    or request.client_order_id,
                    client_order_id=request.client_order_id,
                    quantity=qty,
                    limit_price=limit,
                    idempotency_key=request.client_order_id,
                )
            else:
                order = self.execution.submit_order(
                    client_order_id=request.client_order_id,
                    contract_id=request.contract_id,
                    side=side,
                    quantity=qty,
                    limit_price=limit,
                    idempotency_key=request.client_order_id,
                    parent_order_id=request.replace_of_client_order_id,
                    lifecycle_id=lifecycle,
                )
            if order.status == "rejected":
                return OrderActionResult(
                    submitted=False,
                    client_order_id=request.client_order_id,
                    blocked_reason="replay_order_rejected",
                )
            return OrderActionResult(
                submitted=True,
                client_order_id=order.client_order_id,
                broker_order=None,
            )
        except ReplayExecutionError as exc:
            return OrderActionResult(
                submitted=False,
                client_order_id=request.client_order_id,
                blocked_reason=str(exc),
            )

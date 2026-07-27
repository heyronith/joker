"""Task 2 order-management cognition during isolated replay."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

from joker.agents.cognitive.order_management import OrderManagerAgent
from joker.cognition.prompt_overrides import pinned_applied_configuration
from joker.cognition.schemas import AgentRole, OrderManagementDecision
from joker.evolution.replay_execution import ReplayExecutionRuntime, ReplayOrder
from joker.evolution.replay_gateway import ReplayOrderActionGateway
from joker.evolution.replay_truth import ReplayMarketFrame
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.runtime.order_action_gateway import OrderActionKind, OrderActionRequest


class ReplayOrderManagementRunner:
    """Run production OrderManagerAgent against replay gateway only."""

    def __init__(
        self,
        *,
        deps: CognitiveGraphDeps,
        agent: OrderManagerAgent | None = None,
    ) -> None:
        self._deps = deps
        self._agent = agent or OrderManagerAgent()

    async def manage(
        self,
        *,
        frame: ReplayMarketFrame | None,
        order: ReplayOrder,
        execution: ReplayExecutionRuntime,
        applied_configuration: Any,
        parent_cycle_id: str,
        gateway: ReplayOrderActionGateway,
    ) -> OrderManagementDecision:
        if self._deps.execution_runtime is not None:
            raise RuntimeError("replay_om_has_production_execution_runtime")
        if getattr(self._deps, "submit_callback", None) is not None:
            raise RuntimeError("replay_om_has_production_submit_callback")
        if gateway.execution is not execution:
            raise RuntimeError("replay_om_gateway_execution_mismatch")

        order_projection = {
            "client_order_id": order.client_order_id,
            "contract_id": order.contract_id,
            "side": order.side,
            "quantity": int(order.quantity),
            "filled_quantity": int(order.filled_qty),
            "status": order.status,
            "limit_price": float(order.limit_price) if order.limit_price is not None else None,
        }
        snapshot_id = str(frame.snapshot_id) if frame is not None else None
        context = None
        if snapshot_id and self._deps.snapshot_repo is not None:
            try:
                snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
                    self._deps, snapshot_id
                )
                context = await assemble_role_context(
                    self._deps,
                    agent_role=AgentRole.ORDER_MANAGER,
                    session_id=self._deps.session_id,
                    cycle_id=f"{parent_cycle_id}:om:{order.client_order_id}",
                    snapshot=snapshot,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    order_projection=order_projection,
                )
            except Exception:
                context = None
        if context is None:
            # Fail closed without inventing market context — still invoke agent
            # with a minimal assembled package from the first available snapshot.
            raise RuntimeError("replay_om_missing_snapshot_context")

        with pinned_applied_configuration(applied_configuration):
            decision = await self._agent.manage(
                context,
                self._deps.router,
                client_order_id=order.client_order_id,
                order_projection=order_projection,
            )

        await self._apply(
            decision,
            order=order,
            gateway=gateway,
            snapshot_id=snapshot_id or str(uuid4()),
        )
        return decision

    async def _apply(
        self,
        decision: OrderManagementDecision,
        *,
        order: ReplayOrder,
        gateway: ReplayOrderActionGateway,
        snapshot_id: str,
    ) -> None:
        action = str(decision.action)
        if action in {"continue_waiting", "hold", "wait"}:
            return
        remaining = max(int(order.quantity - order.filled_qty), 1)
        if action in {"cancel", "abandon"}:
            await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.CANCEL,
                    snapshot_id=snapshot_id,
                    contract_id=order.contract_id,
                    side=order.side,  # type: ignore[arg-type]
                    quantity=remaining,
                    client_order_id=f"replay-om-cancel:{order.client_order_id}",
                    replace_of_client_order_id=order.client_order_id,
                    cycle_id=f"om-cancel:{order.client_order_id}",
                )
            )
            return
        if action in {"replace", "reduce_quantity"}:
            new_limit = decision.new_limit_price or order.limit_price
            new_qty = decision.new_quantity or remaining
            await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.REPLACE,
                    snapshot_id=snapshot_id,
                    contract_id=order.contract_id,
                    side=order.side,  # type: ignore[arg-type]
                    quantity=int(new_qty),
                    client_order_id=f"replay-om-replace:{order.client_order_id}",
                    replace_of_client_order_id=order.client_order_id,
                    limit_price=float(new_limit) if new_limit is not None else None,
                    cycle_id=f"om-replace:{order.client_order_id}",
                )
            )
            return
        if action == "exit":
            await gateway.submit(
                OrderActionRequest(
                    action=OrderActionKind.EXIT,
                    snapshot_id=snapshot_id,
                    contract_id=order.contract_id,
                    side="sell",
                    quantity=remaining,
                    client_order_id=f"replay-om-exit:{order.client_order_id}",
                    cycle_id=f"om-exit:{order.client_order_id}",
                )
            )

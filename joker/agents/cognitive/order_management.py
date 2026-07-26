"""Working-order management agent."""

from __future__ import annotations

from typing import Any

from joker.agents.cognitive.base import CognitiveAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, OrderManagementDecision
from joker.models.router import ModelRouter


class OrderManagerAgent(CognitiveAgent[OrderManagementDecision]):
    role = AgentRole.ORDER_MANAGER
    output_type = OrderManagementDecision

    async def manage(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        client_order_id: str,
        order_projection: dict[str, Any] | None = None,
    ) -> OrderManagementDecision:
        """Produce an order-management decision for a working order."""
        extra: dict[str, Any] = {"client_order_id": client_order_id}
        if order_projection is not None:
            extra["order_projection"] = order_projection
        return await self.run(context, router, extra_payload=extra)


async def run_order_manager(
    *,
    state,
    router: ModelRouter,
    context: ContextPackage,
    client_order_id: str,
) -> OrderManagementDecision:
    """Graph-facing order manager wrapper."""
    agent = OrderManagerAgent()
    return await agent.manage(context, router, client_order_id=client_order_id)

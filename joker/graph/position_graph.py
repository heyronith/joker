"""Position management LangGraph — independent of new-entry decision cycles."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.execution import parse_contract_id
from joker.agents.cognitive.position import PositionDecisionAgent, PositionThesisAgent
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, PositionAction, PositionThesisVersion
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_error, append_trace, trace_update
from joker.runtime.execution_runtime import ExecutionCommand
from joker.schemas.domain import OrderIntent


def build_position_graph(deps: CognitiveGraphDeps):
    """Compile the full position graph.

    hydrate_position_truth
    → load_original_strategy
    → load_prior_thesis
    → position_thesis_agent
    → position_execution_critic
    → position_decision_agent
    → route_position_action
    """

    async def hydrate_position_truth(state: CognitiveGraphState) -> dict[str, Any]:
        snapshot_id = state.get("snapshot_id")
        position_id = state.get("_position_id")  # type: ignore[typeddict-item]
        if not snapshot_id or not position_id:
            return append_error(
                state,
                node_name="hydrate_position_truth",
                error_code="missing_position_inputs",
                message="snapshot_id and position_id required",
            )
        snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
            deps, snapshot_id
        )
        projection = None
        if deps.projection_loader is not None:
            projection = await deps.projection_loader()
        position_projection = _extract_position(projection, str(position_id))
        order_projection = _extract_orders(projection)
        cycle_id = state.get("cycle_id") or str(uuid4())
        context = await assemble_role_context(
            deps,
            agent_role=AgentRole.POSITION_THESIS,
            session_id=state.get("session_id") or deps.session_id,
            cycle_id=cycle_id,
            snapshot=snapshot,
            data_quality=data_quality,
            option_surface_slice=surface_slice,
            order_projection=order_projection,
            position_projection=position_projection,
        )
        return {
            "cycle_id": cycle_id,
            "_context_package": context,
            "_position_projection": position_projection,
            "_order_projection": order_projection,
            "_trading_date": snapshot.trading_date.isoformat(),
            **trace_update(
                append_trace(state, node_name="hydrate_position_truth", status="completed")
            ),
        }

    async def load_original_strategy(state: CognitiveGraphState) -> dict[str, Any]:
        strategy_id = state.get("_original_strategy_id")  # type: ignore[typeddict-item]
        strategy = None
        if strategy_id and deps.strategy_repo is not None:
            try:
                strategy = await deps.strategy_repo.get_by_id(UUID(str(strategy_id)))
            except Exception:
                strategy = None
        return {
            "_original_strategy": strategy,
            **trace_update(
                append_trace(state, node_name="load_original_strategy", status="completed")
            ),
        }

    async def load_prior_thesis(state: CognitiveGraphState) -> dict[str, Any]:
        position_id = str(state.get("_position_id") or "")
        prior = None
        if deps.position_thesis_repo is not None and position_id:
            theses = await deps.position_thesis_repo.list_by_session(
                state.get("session_id") or deps.session_id
            )
            matching = [t for t in theses if t.position_id == position_id]
            if matching:
                prior = matching[-1]
        return {
            "_prior_thesis": prior,
            **trace_update(
                append_trace(state, node_name="load_prior_thesis", status="completed")
            ),
        }

    async def position_thesis_agent(state: CognitiveGraphState) -> dict[str, Any]:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return append_error(
                state,
                node_name="position_thesis_agent",
                error_code="missing_context",
                message="context package required",
            )
        position_id = str(state.get("_position_id") or "")
        contract_id = str(state.get("_contract_id") or "")
        strategy_raw = state.get("_original_strategy_id")
        strategy_id = (
            UUID(str(strategy_raw)) if strategy_raw else uuid4()
        )
        prior = state.get("_prior_thesis")  # type: ignore[typeddict-item]
        agent = PositionThesisAgent()
        thesis = await agent.reassess(
            context,
            deps.router,
            position_id=position_id,
            contract_id=contract_id,
            original_strategy_id=strategy_id,
            position_projection=state.get("_position_projection"),  # type: ignore[arg-type]
            prior_version=prior if isinstance(prior, PositionThesisVersion) else None,
        )
        if deps.position_thesis_repo is not None:
            await deps.position_thesis_repo.append(thesis)
        return {
            "_position_thesis": thesis,
            **trace_update(
                append_trace(
                    state,
                    node_name="position_thesis_agent",
                    status="completed",
                    artifact_ids=(thesis.thesis_version_id,),
                )
            ),
        }

    async def position_execution_critic(state: CognitiveGraphState) -> dict[str, Any]:
        # Lightweight deterministic critic notes attached for the decision agent.
        thesis = state.get("_position_thesis")  # type: ignore[typeddict-item]
        notes = {
            "has_thesis": isinstance(thesis, PositionThesisVersion),
            "open_orders": bool(state.get("_order_projection")),
        }
        return {
            "_position_critic_notes": notes,
            **trace_update(
                append_trace(
                    state, node_name="position_execution_critic", status="completed"
                )
            ),
        }

    async def position_decision_agent(state: CognitiveGraphState) -> dict[str, Any]:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        thesis = state.get("_position_thesis")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage) or not isinstance(
            thesis, PositionThesisVersion
        ):
            return append_error(
                state,
                node_name="position_decision_agent",
                error_code="missing_thesis",
                message="thesis required for position decision",
            )
        decision_context = context
        # Re-assemble with POSITION_DECISION role so context hash reflects the role.
        snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
            deps, context.snapshot_id
        )
        decision_context = await assemble_role_context(
            deps,
            agent_role=AgentRole.POSITION_DECISION,
            session_id=context.session_id,
            cycle_id=context.cycle_id,
            snapshot=snapshot,
            data_quality=data_quality,
            option_surface_slice=surface_slice,
            order_projection=state.get("_order_projection"),  # type: ignore[arg-type]
            position_projection=state.get("_position_projection"),  # type: ignore[arg-type]
            session_artifact_summaries=(
                {
                    "critic_notes": state.get("_position_critic_notes"),
                    "thesis_id": str(thesis.thesis_version_id),
                },
            ),
        )
        agent = PositionDecisionAgent()
        decision = await agent.decide(
            decision_context,
            deps.router,
            latest_thesis=thesis,
            position_projection=state.get("_position_projection"),  # type: ignore[arg-type]
        )
        if deps.position_thesis_repo is not None:
            await deps.position_thesis_repo.append(decision)
        return {
            "_position_decision": decision,
            **trace_update(
                append_trace(
                    state,
                    node_name="position_decision_agent",
                    status="completed",
                    artifact_ids=(decision.thesis_version_id,),
                )
            ),
        }

    async def route_position_action(state: CognitiveGraphState) -> dict[str, Any]:
        decision = state.get("_position_decision")  # type: ignore[typeddict-item]
        if not isinstance(decision, PositionThesisVersion):
            return append_error(
                state,
                node_name="route_position_action",
                error_code="missing_decision",
                message="position decision missing",
            )
        action = decision.recommended_action
        result: dict[str, Any] = {
            "_position_action": action.value,
            **trace_update(
                append_trace(state, node_name="route_position_action", status="started")
            ),
        }
        if action in {
            PositionAction.HOLD,
            PositionAction.UPDATE_THESIS,
        }:
            result.update(
                trace_update(
                    append_trace(
                        state, node_name="route_position_action", status="completed"
                    )
                )
            )
            return result

        if deps.execution_runtime is None:
            return {
                **result,
                **append_error(
                    state,
                    node_name="route_position_action",
                    error_code="no_execution_runtime",
                    message="ExecutionRuntime required for order actions",
                ),
            }

        trading_date_raw = state.get("_trading_date")
        trading_date = (
            date.fromisoformat(str(trading_date_raw))
            if trading_date_raw
            else date.today()
        )

        if action == PositionAction.CANCEL_WORKING_ORDER:
            orders = state.get("_order_projection") or {}
            client_order_id = str(orders.get("client_order_id") or "")
            if client_order_id:
                await deps.execution_runtime.cancel_order(client_order_id=client_order_id)
                result["_position_command_id"] = f"cancel:{client_order_id}"
        elif action in {
            PositionAction.EXIT,
            PositionAction.REDUCE,
            PositionAction.ADD,
            PositionAction.REPLACE_WORKING_ORDER,
        }:
            if state.get("_position_command_id"):
                # Idempotent: do not duplicate execution commands after recovery.
                result.update(
                    trace_update(
                        append_trace(
                            state,
                            node_name="route_position_action",
                            status="skipped",
                        )
                    )
                )
                return result
            qty = decision.recommended_quantity or 1
            if action == PositionAction.REDUCE:
                qty = max(1, int(qty))
            side = "buy" if action == PositionAction.ADD else "sell"
            if action == PositionAction.REPLACE_WORKING_ORDER:
                orders = state.get("_order_projection") or {}
                prior = str(orders.get("client_order_id") or "")
                if prior:
                    await deps.execution_runtime.cancel_order(client_order_id=prior)
            contract = parse_contract_id(decision.contract_id, trading_date=trading_date)
            limit = (
                float(decision.recommended_limit_price)
                if decision.recommended_limit_price is not None
                else None
            )
            client_order_id = f"pos-{decision.thesis_version_id}-{action.value}"
            command = ExecutionCommand(
                client_order_id=client_order_id,
                intent=OrderIntent(
                    candidate_id=str(decision.thesis_version_id),
                    contract=contract,
                    side=side,
                    order_type="limit" if limit is not None else "market",
                    quantity=int(qty),
                    limit_price=limit,
                ),
            )
            await deps.execution_runtime.submit_execution_command(command)
            result["_position_command_id"] = client_order_id

        result.update(
            trace_update(
                append_trace(state, node_name="route_position_action", status="completed")
            )
        )
        return result

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("hydrate_position_truth", hydrate_position_truth)
    graph.add_node("load_original_strategy", load_original_strategy)
    graph.add_node("load_prior_thesis", load_prior_thesis)
    graph.add_node("position_thesis_agent", position_thesis_agent)
    graph.add_node("position_execution_critic", position_execution_critic)
    graph.add_node("position_decision_agent", position_decision_agent)
    graph.add_node("route_position_action", route_position_action)

    graph.add_edge(START, "hydrate_position_truth")
    graph.add_edge("hydrate_position_truth", "load_original_strategy")
    graph.add_edge("load_original_strategy", "load_prior_thesis")
    graph.add_edge("load_prior_thesis", "position_thesis_agent")
    graph.add_edge("position_thesis_agent", "position_execution_critic")
    graph.add_edge("position_execution_critic", "position_decision_agent")
    graph.add_edge("position_decision_agent", "route_position_action")
    graph.add_edge("route_position_action", END)

    compiled_kwargs = {}
    if deps.checkpointer is not None:
        compiled_kwargs["checkpointer"] = deps.checkpointer
    return graph.compile(**compiled_kwargs)


def _extract_position(projection: Any, position_id: str) -> dict[str, Any] | None:
    if projection is None:
        return {"position_id": position_id}
    positions = getattr(projection, "positions", None) or {}
    if isinstance(positions, dict):
        items = list(positions.items())
    else:
        items = [(getattr(p, "contract_id", None), p) for p in positions]
    for key, pos in items:
        pid = str(key)
        if isinstance(pos, dict):
            if str(pos.get("position_id") or pos.get("contract_id") or key) in {
                position_id,
                pid,
            }:
                return dict(pos)
            continue
        cid = getattr(pos, "contract_id", None) or key
        if str(cid) == position_id or pid == position_id:
            qty = getattr(pos, "quantity", None) or getattr(pos, "net_quantity", None)
            return {
                "position_id": position_id,
                "contract_id": str(cid),
                "quantity": str(qty) if qty is not None else None,
                "average_price": str(getattr(pos, "average_price", None) or ""),
            }
    for key, pos in items:
        qty = getattr(pos, "quantity", None) if not isinstance(pos, dict) else pos.get("quantity")
        try:
            if qty is not None and Decimal(str(qty)) != 0:
                if isinstance(pos, dict):
                    return dict(pos)
                return {
                    "position_id": position_id,
                    "contract_id": str(getattr(pos, "contract_id", key)),
                    "quantity": str(qty),
                }
        except Exception:
            continue
    return {"position_id": position_id}


def _extract_orders(projection: Any) -> dict[str, Any] | None:
    if projection is None:
        return None
    orders = getattr(projection, "orders", None) or {}
    if isinstance(orders, dict):
        order_iter = orders.values()
    else:
        order_iter = orders
    for order in order_iter:
        status = (
            order.get("status")
            if isinstance(order, dict)
            else getattr(order, "status", None)
        )
        status_val = getattr(status, "value", status)
        if status_val in {"submitted", "accepted", "partially_filled", "open", "WORKING", "working"}:
            if isinstance(order, dict):
                return dict(order)
            return {
                "client_order_id": getattr(order, "client_order_id", None),
                "status": status_val,
                "quantity": getattr(order, "quantity", None),
                "filled_quantity": getattr(order, "filled_quantity", None),
            }
    return None

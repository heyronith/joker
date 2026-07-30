"""Strategy construction LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.strategy import run_strategy_inventors
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_strategy_graph(deps: CognitiveGraphDeps):
    """Compile strategy construction subgraph."""

    async def run_strategy_node(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        # Goal-aware inventors receive sanitised ObjectiveContext (perception does not).
        objective_context = state.get("_objective_context")
        if objective_context and deps.snapshot_repo is not None and state.get("snapshot_id"):
            try:
                record, data_quality, _surface, surface_slice = await load_snapshot_truth(
                    deps, state["snapshot_id"]
                )
                context = await assemble_role_context(
                    deps,
                    agent_role=AgentRole.BULLISH_INVENTOR,
                    session_id=state.get("session_id") or deps.session_id,
                    cycle_id=state.get("cycle_id") or "",
                    snapshot=record,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    objective_context=objective_context,
                )
            except Exception:
                pass
        strategies = list(await run_strategy_inventors(deps.router, context))
        strategies = strategies[: deps.config.max_strategy_candidates]
        if deps.strategy_repo is not None:
            for strategy in strategies:
                await deps.strategy_repo.append(strategy)
        return {
            "strategies": strategies,
            **trace_update(
                append_trace(state, node_name="strategy_construction", status="started"),
                append_trace(
                    state,
                    node_name="strategy_construction",
                    status="completed",
                    artifact_ids=tuple(s.strategy_id for s in strategies),
                ),
            ),
        }

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("run_strategy_construction", run_strategy_node)
    graph.add_edge(START, "run_strategy_construction")
    graph.add_edge("run_strategy_construction", END)
    return graph.compile()

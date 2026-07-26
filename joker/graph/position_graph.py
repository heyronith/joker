"""Position management LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.position import run_position_cycle
from joker.cognition.context import ContextPackage
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_position_graph(deps: CognitiveGraphDeps):
    """Compile position management subgraph."""

    async def run_position_node(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        position_id = state.get("_position_id")  # type: ignore[typeddict-item]
        contract_id = state.get("_contract_id")  # type: ignore[typeddict-item]
        strategy_id = state.get("_original_strategy_id")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage) or not position_id:
            return {}
        thesis = await run_position_cycle(
            state=state,
            router=deps.router,
            context=context,
            position_id=str(position_id),
            contract_id=str(contract_id or ""),
            original_strategy_id=strategy_id,
        )
        return {
            "_position_thesis": thesis,
            **trace_update(
                append_trace(state, node_name="position_cycle", status="started"),
                append_trace(
                    state,
                    node_name="position_cycle",
                    status="completed",
                    artifact_ids=(thesis.thesis_version_id,),
                ),
            ),
        }

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("run_position_cycle", run_position_node)
    graph.add_edge(START, "run_position_cycle")
    graph.add_edge("run_position_cycle", END)
    return graph.compile()

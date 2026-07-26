"""Strategy construction LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.strategy import run_strategy_inventors
from joker.cognition.context import ContextPackage
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_strategy_graph(deps: CognitiveGraphDeps):
    """Compile strategy construction subgraph."""

    async def run_strategy_node(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        strategies = list(await run_strategy_inventors(deps.router, context))
        strategies = strategies[: deps.config.max_strategy_candidates]
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

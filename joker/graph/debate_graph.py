"""Debate LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.debate import run_debate_panel
from joker.cognition.context import ContextPackage
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_debate_graph(deps: CognitiveGraphDeps):
    """Compile debate subgraph."""

    async def run_debate_node(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        strategies = state.get("strategies") or []
        selected_ids = set(state.get("selected_strategy_ids") or [])
        if selected_ids:
            strategies = [s for s in strategies if str(s.strategy_id) in selected_ids]
        reviews = []
        for strategy in strategies:
            panel = await run_debate_panel(strategy, context, deps.router)
            reviews.extend(panel)
        return {
            "reviews": reviews,
            **trace_update(
                append_trace(state, node_name="debate_panel", status="started"),
                append_trace(
                    state,
                    node_name="debate_panel",
                    status="completed",
                    artifact_ids=tuple(r.review_id for r in reviews),
                ),
            ),
        }

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("run_debate_panel", run_debate_node)
    graph.add_edge(START, "run_debate_panel")
    graph.add_edge("run_debate_panel", END)
    return graph.compile()

"""Discovery LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.discovery import run_discovery_swarm
from joker.cognition.context import ContextPackage
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_discovery_graph(deps: CognitiveGraphDeps):
    """Compile discovery subgraph."""

    async def run_discovery_node(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        hypotheses = list(
            await run_discovery_swarm(
                deps.router,
                context,
                max_hypotheses=deps.config.max_hypotheses_per_cycle,
            )
        )
        if deps.hypothesis_repo is not None:
            for hyp in hypotheses:
                await deps.hypothesis_repo.append(hyp)
        return {
            "hypotheses": hypotheses,
            **trace_update(
                append_trace(state, node_name="discovery", status="started"),
                append_trace(
                    state,
                    node_name="discovery",
                    status="completed",
                    artifact_ids=tuple(h.hypothesis_id for h in hypotheses),
                ),
            ),
        }

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("run_discovery", run_discovery_node)
    graph.add_edge(START, "run_discovery")
    graph.add_edge("run_discovery", END)
    return graph.compile()

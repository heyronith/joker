"""Perception LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.perception import run_perception_swarm
from joker.cognition.context import ContextPackage
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_perception_graph(deps: CognitiveGraphDeps):
    """Compile perception subgraph: run_perception_swarm."""

    async def run_perception(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        evidence = list(await run_perception_swarm(deps.router, context))
        return {
            "evidence": evidence,
            **trace_update(
                append_trace(state, node_name="perception_swarm", status="started"),
                append_trace(
                    state,
                    node_name="perception_swarm",
                    status="completed",
                    artifact_ids=tuple(e.evidence_id for e in evidence),
                ),
            ),
        }

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("run_perception_swarm", run_perception)
    graph.add_edge(START, "run_perception_swarm")
    graph.add_edge("run_perception_swarm", END)
    return graph.compile()

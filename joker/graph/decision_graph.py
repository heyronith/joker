"""Meta-decision LangGraph subgraph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.decision import run_meta_decision
from joker.cognition.context import ContextPackage
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace


def build_decision_graph(deps: CognitiveGraphDeps):
    """Compile meta-decision subgraph."""

    async def run_meta_decision_node(state: CognitiveGraphState) -> dict:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        strategies = state.get("strategies") or []
        started = append_trace(state, node_name="meta_decision", status="started")
        decision = await run_meta_decision(
            state=state,
            router=deps.router,
            context=context,
            strategies=strategies,
        )
        done = append_trace(
            state,
            node_name="meta_decision",
            status="completed",
            artifact_ids=(decision.decision_id,),
        )
        return {"meta_decision": decision, **started, **done}

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("run_meta_decision", run_meta_decision_node)
    graph.add_edge(START, "run_meta_decision")
    graph.add_edge("run_meta_decision", END)
    return graph.compile()

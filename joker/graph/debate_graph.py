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
        from joker.objectives.portfolio_review import (
            build_portfolio_review_context,
            portfolio_review_from_debate,
        )

        full_chain_settings = getattr(deps, "full_chain_optimizer_settings", None)
        review_limit = int(
            getattr(full_chain_settings, "top_candidates_for_agent_review", 10)
            or 10
        )
        portfolio_context = build_portfolio_review_context(
            state=dict(state),
            limit=review_limit,
        )
        reviews = []
        for strategy in strategies:
            panel = await run_debate_panel(
                strategy,
                context,
                deps.router,
                portfolio_review_context=(
                    portfolio_context.model_dump(mode="json")
                    if portfolio_context is not None
                    else None
                ),
            )
            reviews.extend(panel)
        if deps.debate_repo is not None:
            session_id = state.get("session_id") or deps.session_id
            for review in reviews:
                await deps.debate_repo.append(review, session_id=session_id)
        portfolio_reviews = (
            [
                portfolio_review_from_debate(review, portfolio_context).model_dump(
                    mode="json"
                )
                for review in reviews
            ]
            if portfolio_context is not None
            else []
        )
        return {
            "reviews": reviews,
            "_portfolio_review_context": (
                portfolio_context.model_dump(mode="json")
                if portfolio_context is not None
                else None
            ),
            "_portfolio_debate_reviews": portfolio_reviews,
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

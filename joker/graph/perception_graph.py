"""Perception LangGraph subgraph with role-specific context packages."""

from __future__ import annotations

import asyncio

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.perception import PERCEPTION_ROLES, perception_agent_for
from joker.cognition.schemas import AgentEvidence
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_trace, trace_update


def build_perception_graph(deps: CognitiveGraphDeps):
    """Compile perception subgraph: role-specific contexts + parallel agents."""

    async def run_perception(state: CognitiveGraphState) -> dict:
        base_context = state.get("_context_package")  # type: ignore[typeddict-item]
        snapshot_id = state.get("snapshot_id")
        cycle_id = state.get("cycle_id") or ""
        session_id = state.get("session_id") or deps.session_id
        if not snapshot_id:
            return {}

        snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
            deps, snapshot_id
        )
        order_projection = None
        position_projection = None
        if base_context is not None and hasattr(base_context, "order_projection"):
            order_projection = base_context.order_projection
            position_projection = base_context.position_projection

        async def _run_one(role) -> AgentEvidence:
            role_context = await assemble_role_context(
                deps,
                agent_role=role,
                session_id=session_id,
                cycle_id=cycle_id,
                snapshot=snapshot,
                data_quality=data_quality,
                option_surface_slice=surface_slice,
                order_projection=order_projection,
                position_projection=position_projection,
            )
            agent = perception_agent_for(role)
            evidence = await agent.run(role_context, deps.router)
            if deps.evidence_repo is not None:
                await deps.evidence_repo.append(evidence)
            return evidence

        evidence = list(await asyncio.gather(*[_run_one(role) for role in PERCEPTION_ROLES]))
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
    compiled_kwargs = {}
    if deps.checkpointer is not None:
        compiled_kwargs["checkpointer"] = deps.checkpointer
    return graph.compile(**compiled_kwargs)

"""Parent cognitive decision LangGraph (Task 2 spec section 19)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.decision import run_meta_decision
from joker.agents.cognitive.execution import (
    run_entry_tactician,
    validate_and_compile_proposal,
)
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import (
    AgentRole,
    MarketAnomaly,
    MarketDirection,
    MarketStructureAssessment,
    MarketWorldModel,
    MetaDecisionAction,
    OptionsMicrostructureAssessment,
    RegimeHypothesis,
    TemporalAssessment,
    VolatilityAssessment,
)
from joker.events.schemas import EventType, make_event
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.debate_graph import build_debate_graph
from joker.graph.decision_graph import build_decision_graph
from joker.graph.discovery_graph import build_discovery_graph
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_error, append_trace, trace_update, utc_now
from joker.graph.perception_graph import build_perception_graph
from joker.graph.strategy_graph import build_strategy_graph

logger = logging.getLogger(__name__)


def _selected_strategy(state: CognitiveGraphState):
    meta = state.get("meta_decision")
    strategies = state.get("strategies") or []
    if meta is None or meta.selected_strategy_id is None:
        return None
    for strategy in strategies:
        if strategy.strategy_id == meta.selected_strategy_id:
            return strategy
    return strategies[0] if strategies else None


def build_cognitive_graph(deps: CognitiveGraphDeps):
    """Compile the parent cognitive decision graph with explicit subgraph nodes."""
    perception = build_perception_graph(deps)
    discovery = build_discovery_graph(deps)
    strategy_graph = build_strategy_graph(deps)
    debate = build_debate_graph(deps)

    async def validate_trigger(state: CognitiveGraphState) -> dict[str, Any]:
        if not state.get("trigger_event_id") or not state.get("snapshot_id"):
            return append_error(
                state,
                node_name="validate_trigger",
                error_code="invalid_trigger",
                message="missing trigger_event_id or snapshot_id",
                recoverable=False,
            )
        return trace_update(append_trace(state, node_name="validate_trigger", status="completed"))

    async def hydrate_context(state: CognitiveGraphState) -> dict[str, Any]:
        snapshot_id = state.get("snapshot_id")
        if not snapshot_id or deps.snapshot_repo is None:
            return append_error(
                state,
                node_name="hydrate_context",
                error_code="snapshot_unavailable",
                message="cannot hydrate context without snapshot repository",
            )
        record = await deps.snapshot_repo.get_by_id(UUID(str(snapshot_id)))
        if record is None:
            return append_error(
                state,
                node_name="hydrate_context",
                error_code="snapshot_not_found",
                message=f"snapshot {snapshot_id} not found",
            )
        cycle_id = state.get("cycle_id") or str(uuid4())
        context = deps.context_assembler.assemble(
            agent_role=AgentRole.MARKET_STRUCTURE,
            session_id=state.get("session_id") or deps.session_id,
            cycle_id=cycle_id,
            snapshot=record,
        )
        return {
            "cycle_id": cycle_id,
            "context_ref": context.context_id,
            "_context_package": context,
            "latest_known_snapshot_id": str(snapshot_id),
            **trace_update(append_trace(state, node_name="hydrate_context", status="completed")),
        }

    async def synthesise_world_model(state: CognitiveGraphState) -> dict[str, Any]:
        evidence = state.get("evidence") or []
        if not evidence:
            return append_error(
                state,
                node_name="synthesise_world_model",
                error_code="no_evidence",
                message="cannot synthesise world model without evidence",
            )
        snapshot_id = UUID(str(state.get("snapshot_id")))
        cycle_id = state.get("cycle_id") or ""
        session_id = state.get("session_id") or deps.session_id
        evidence_ids = tuple(e.evidence_id for e in evidence)
        model_call_id = uuid4()
        world_model = MarketWorldModel(
            session_id=session_id,
            snapshot_id=snapshot_id,
            prompt_version="task2-v1",
            model_call_id=model_call_id,
            cycle_id=cycle_id,
            regime_hypotheses=(
                RegimeHypothesis(
                    label="synthesised",
                    direction=MarketDirection.UNCERTAIN,
                    confidence=0.5,
                    supporting_evidence_ids=evidence_ids[:5],
                    rationale="Deterministic synthesis from perception evidence",
                ),
            ),
            market_structure=MarketStructureAssessment(
                primary_direction=MarketDirection.UNCERTAIN,
                structure_summary="Synthesised from perception swarm",
                supporting_evidence_ids=evidence_ids[:5],
                confidence=0.5,
            ),
            volatility_state=VolatilityAssessment(
                state=MarketDirection.UNCERTAIN,
                summary="Synthesised volatility assessment",
                supporting_evidence_ids=evidence_ids[:5],
                confidence=0.5,
            ),
            options_state=OptionsMicrostructureAssessment(
                liquidity_summary="Synthesised microstructure",
                spread_conditions="unknown",
                supporting_evidence_ids=evidence_ids[:5],
                confidence=0.5,
            ),
            temporal_state=TemporalAssessment(
                session_phase="regular",
                time_decay_context="0DTE context",
                supporting_evidence_ids=evidence_ids[:5],
                confidence=0.5,
            ),
            anomalies=tuple(
                MarketAnomaly(
                    description=e.claim,
                    severity="low",
                    supporting_evidence_ids=(e.evidence_id,),
                )
                for e in evidence
                if e.agent_role == AgentRole.ANOMALY
            ),
            evidence_ids=evidence_ids,
            overall_uncertainty=0.5,
            synthesizer_model_call_id=model_call_id,
        )
        if deps.world_model_repo is not None:
            await deps.world_model_repo.append(world_model)
        return {
            "world_model": world_model,
            **trace_update(
                append_trace(
                    state,
                    node_name="synthesise_world_model",
                    status="completed",
                    artifact_ids=(world_model.world_model_id,),
                )
            ),
        }

    async def select_debate_candidates(state: CognitiveGraphState) -> dict[str, Any]:
        strategies = state.get("strategies") or []
        limit = deps.config.max_strategy_candidates
        selected = [str(s.strategy_id) for s in strategies[:limit]]
        return {
            "selected_strategy_ids": selected,
            **trace_update(
                append_trace(state, node_name="select_debate_candidates", status="completed")
            ),
        }

    async def meta_decision_node(state: CognitiveGraphState) -> dict[str, Any]:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        strategies = state.get("strategies") or []
        decision = await run_meta_decision(
            state=state,
            router=deps.router,
            context=context,
            strategies=strategies,
        )
        if deps.decision_repo is not None:
            await deps.decision_repo.append_meta(decision)
        return {
            "meta_decision": decision,
            **trace_update(
                append_trace(
                    state,
                    node_name="meta_decision",
                    status="completed",
                    artifact_ids=(decision.decision_id,),
                )
            ),
        }

    def route_meta_decision(state: CognitiveGraphState) -> str:
        meta = state.get("meta_decision")
        if meta is None:
            return "persist_cycle"
        if state.get("stale_decision"):
            return "persist_stale"
        action = meta.action
        if action in {MetaDecisionAction.EXECUTE, MetaDecisionAction.PROBE}:
            return "entry_tactician"
        if action == MetaDecisionAction.DELAY:
            return "persist_pending_cycle"
        if action == MetaDecisionAction.REQUEST_MORE_EVIDENCE:
            return "persist_evidence_request"
        if action == MetaDecisionAction.SWITCH_STRATEGY:
            switches = int(state.get("strategy_switch_count") or 0)
            if switches >= deps.config.max_strategy_switches:
                return "persist_cycle"
            return "strategy_switch_revision"
        return "persist_cycle"

    async def strategy_switch_revision(state: CognitiveGraphState) -> dict[str, Any]:
        switches = int(state.get("strategy_switch_count") or 0) + 1
        meta = state.get("meta_decision")
        alternate_ids = list(meta.alternate_strategy_ids) if meta else []
        return {
            "strategy_switch_count": switches,
            "selected_strategy_ids": [str(i) for i in alternate_ids],
            "debate_round": int(state.get("debate_round") or 0) + 1,
            **trace_update(
                append_trace(state, node_name="strategy_switch_revision", status="completed")
            ),
        }

    def after_switch_route(state: CognitiveGraphState) -> str:
        debate_round = int(state.get("debate_round") or 0)
        if debate_round <= deps.config.max_debate_rounds:
            return "debate"
        return "meta_decision"

    async def entry_tactician_node(state: CognitiveGraphState) -> dict[str, Any]:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        meta = state.get("meta_decision")
        strategy = _selected_strategy(state)
        if not isinstance(context, ContextPackage) or meta is None or strategy is None:
            return append_error(
                state,
                node_name="entry_tactician",
                error_code="missing_inputs",
                message="meta_decision and strategy required for entry tactician",
            )
        proposal = await run_entry_tactician(
            state=state,
            router=deps.router,
            context=context,
            meta_decision=meta,
            strategy=strategy,
        )
        if deps.decision_repo is not None:
            await deps.decision_repo.append_proposal(proposal)
        return {
            "execution_proposal": proposal,
            **trace_update(
                append_trace(
                    state,
                    node_name="entry_tactician",
                    status="completed",
                    artifact_ids=(proposal.proposal_id,),
                )
            ),
        }

    async def validate_execution_proposal(state: CognitiveGraphState) -> dict[str, Any]:
        proposal = state.get("execution_proposal")
        if proposal is None:
            return append_error(
                state,
                node_name="validate_execution_proposal",
                error_code="missing_proposal",
                message="no execution proposal to validate",
            )
        latest = state.get("latest_known_snapshot_id")
        try:
            validate_and_compile_proposal(
                proposal,
                latest_snapshot_id=latest,
                evidence_ids=tuple(e.evidence_id for e in state.get("evidence") or []),
            )
        except Exception as exc:
            if "stale" in str(exc).lower():
                return {"stale_decision": True, **append_error(
                    state,
                    node_name="validate_execution_proposal",
                    error_code="stale_proposal",
                    message=str(exc),
                )}
            return append_error(
                state,
                node_name="validate_execution_proposal",
                error_code="validation_failed",
                message=str(exc),
            )
        return trace_update(append_trace(state, node_name="validate_execution_proposal", status="completed"))

    async def submit_execution_command(state: CognitiveGraphState) -> dict[str, Any]:
        proposal = state.get("execution_proposal")
        if proposal is None:
            return {}
        if state.get("execution_command_id"):
            return trace_update(append_trace(state, node_name="submit_execution_command", status="skipped"))
        try:
            provenanced = validate_and_compile_proposal(
                proposal,
                latest_snapshot_id=state.get("latest_known_snapshot_id"),
                evidence_ids=tuple(e.evidence_id for e in state.get("evidence") or []),
            )
        except Exception as exc:
            return append_error(
                state,
                node_name="submit_execution_command",
                error_code="submit_validation_failed",
                message=str(exc),
            )
        if deps.submit_callback is None:
            return append_error(
                state,
                node_name="submit_execution_command",
                error_code="no_submit_callback",
                message="execution submit callback not configured",
            )
        result = await deps.submit_callback(provenanced)
        command_id = provenanced.command.client_order_id
        return {
            "execution_command_id": command_id,
            "execution_result_ref": str(getattr(result, "order_id", command_id)),
            **trace_update(
                append_trace(state, node_name="submit_execution_command", status="completed")
            ),
        }

    async def persist_cycle(state: CognitiveGraphState) -> dict[str, Any]:
        await _publish_cycle_completed(deps, state, outcome="completed")
        return trace_update(append_trace(state, node_name="persist_cycle", status="completed"))

    async def persist_pending_cycle(state: CognitiveGraphState) -> dict[str, Any]:
        await _publish_cycle_completed(deps, state, outcome="delayed")
        return trace_update(append_trace(state, node_name="persist_pending_cycle", status="completed"))

    async def persist_evidence_request(state: CognitiveGraphState) -> dict[str, Any]:
        meta = state.get("meta_decision")
        pending = list(meta.requested_evidence) if meta else []
        await _publish_cycle_completed(deps, state, outcome="more_evidence_requested")
        return {
            "pending_evidence_requests": pending,
            **trace_update(
                append_trace(state, node_name="persist_evidence_request", status="completed")
            ),
        }

    async def persist_stale(state: CognitiveGraphState) -> dict[str, Any]:
        await _publish_cycle_completed(deps, state, outcome="stale")
        return trace_update(append_trace(state, node_name="persist_stale", status="completed"))

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("validate_trigger", validate_trigger)
    graph.add_node("hydrate_context", hydrate_context)
    graph.add_node("perception", perception)
    graph.add_node("synthesise_world_model", synthesise_world_model)
    graph.add_node("discovery", discovery)
    graph.add_node("strategy", strategy_graph)
    graph.add_node("select_debate_candidates", select_debate_candidates)
    graph.add_node("debate", debate)
    graph.add_node("meta_decision", meta_decision_node)
    graph.add_node("strategy_switch_revision", strategy_switch_revision)
    graph.add_node("entry_tactician", entry_tactician_node)
    graph.add_node("validate_execution_proposal", validate_execution_proposal)
    graph.add_node("submit_execution_command", submit_execution_command)
    graph.add_node("persist_cycle", persist_cycle)
    graph.add_node("persist_pending_cycle", persist_pending_cycle)
    graph.add_node("persist_evidence_request", persist_evidence_request)
    graph.add_node("persist_stale", persist_stale)

    graph.add_edge(START, "validate_trigger")
    graph.add_edge("validate_trigger", "hydrate_context")
    graph.add_edge("hydrate_context", "perception")
    graph.add_edge("perception", "synthesise_world_model")
    graph.add_edge("synthesise_world_model", "discovery")
    graph.add_edge("discovery", "strategy")
    graph.add_edge("strategy", "select_debate_candidates")
    graph.add_edge("select_debate_candidates", "debate")
    graph.add_edge("debate", "meta_decision")
    graph.add_conditional_edges(
        "meta_decision",
        route_meta_decision,
        {
            "entry_tactician": "entry_tactician",
            "persist_cycle": "persist_cycle",
            "persist_pending_cycle": "persist_pending_cycle",
            "persist_evidence_request": "persist_evidence_request",
            "persist_stale": "persist_stale",
            "strategy_switch_revision": "strategy_switch_revision",
        },
    )
    graph.add_conditional_edges(
        "strategy_switch_revision",
        after_switch_route,
        {"debate": "debate", "meta_decision": "meta_decision"},
    )

    def after_validate_route(state: CognitiveGraphState) -> str:
        if state.get("stale_decision"):
            return "persist_stale"
        errors = state.get("errors") or []
        if any(e.error_code == "validation_failed" for e in errors):
            return "persist_cycle"
        return "submit_execution_command"

    graph.add_conditional_edges(
        "validate_execution_proposal",
        after_validate_route,
        {
            "submit_execution_command": "submit_execution_command",
            "persist_stale": "persist_stale",
            "persist_cycle": "persist_cycle",
        },
    )
    graph.add_edge("entry_tactician", "validate_execution_proposal")
    graph.add_edge("submit_execution_command", "persist_cycle")
    graph.add_edge("persist_cycle", END)
    graph.add_edge("persist_pending_cycle", END)
    graph.add_edge("persist_evidence_request", END)
    graph.add_edge("persist_stale", END)

    return graph.compile()


async def _publish_cycle_completed(
    deps: CognitiveGraphDeps,
    state: CognitiveGraphState,
    *,
    outcome: str,
) -> None:
    if deps.event_bus is None:
        return
    now = deps.clock.now() if deps.clock else utc_now()
    await deps.event_bus.publish(
        make_event(
            EventType.COGNITIVE_CYCLE_COMPLETED,
            session_id=state.get("session_id") or deps.session_id,
            source="cognitive_graph",
            exchange_timestamp=now,
            payload={
                "cycle_id": state.get("cycle_id"),
                "snapshot_id": state.get("snapshot_id"),
                "outcome": outcome,
                "execution_command_id": state.get("execution_command_id"),
            },
        )
    )


def initial_cycle_state(
    *,
    session_id: str,
    run_id: str,
    cycle_id: str,
    trigger_event_id: str,
    trigger_event_type: str,
    snapshot_id: str,
) -> CognitiveGraphState:
    """Build initial state for a new-entry cognitive cycle."""
    return CognitiveGraphState(
        run_id=run_id,
        session_id=session_id,
        cycle_id=cycle_id,
        trigger_event_id=trigger_event_id,
        trigger_event_type=trigger_event_type,
        snapshot_id=snapshot_id,
        latest_known_snapshot_id=snapshot_id,
        started_at=datetime.now(timezone.utc),
        evidence=[],
        hypotheses=[],
        strategies=[],
        reviews=[],
        node_trace=[],
        errors=[],
        debate_round=0,
        strategy_switch_count=0,
        selected_strategy_ids=[],
        pending_evidence_requests=[],
        pending_hypothesis_ids=[],
        stale_decision=False,
    )

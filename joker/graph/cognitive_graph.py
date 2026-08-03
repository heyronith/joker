"""Parent cognitive decision LangGraph (Task 2 spec section 19)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from langgraph.graph import END, START, StateGraph

from joker.agents.cognitive.decision import run_meta_decision
from joker.agents.cognitive.execution import (
    build_truth_from_deps,
    run_entry_tactician,
    validate_and_compile_proposal,
)
from joker.agents.cognitive.world_model import run_world_model_synthesis
from joker.cognition.context import ContextPackage
from joker.cognition.schemas import AgentRole, MetaDecisionAction
from joker.events.schemas import EventType, make_event
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.context_hydrate import assemble_role_context, load_snapshot_truth
from joker.graph.debate_graph import build_debate_graph
from joker.graph.discovery_graph import build_discovery_graph
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_error, append_trace, trace_update, utc_now
from joker.graph.perception_graph import build_perception_graph
from joker.graph.strategy_graph import build_strategy_graph
from joker.graph.objective_nodes import (
    apply_objective_sizing_to_proposal,
    assess_goal_feasibility_node,
    deterministic_sizing_node,
    entry_blocked_by_objective,
    gate_objective_confirmed,
    load_objective_context,
    score_strategies_against_objective_node,
)

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
            return {
                "_block_new_entries": True,
                **append_error(
                    state,
                    node_name="validate_trigger",
                    error_code="invalid_trigger",
                    message="missing trigger_event_id or snapshot_id",
                    recoverable=False,
                ),
            }
        perm = getattr(deps, "entry_permission", None)
        if perm is not None and not bool(getattr(perm, "permitted", True)):
            reasons = getattr(perm, "reasons", ()) or ()
            return {
                "_block_new_entries": True,
                **append_error(
                    state,
                    node_name="validate_trigger",
                    error_code="entry_permission_blocked",
                    message=(
                        "entries blocked: " + (", ".join(reasons) or "reconciliation")
                    ),
                    recoverable=True,
                ),
                **trace_update(
                    append_trace(
                        state, node_name="validate_trigger", status="completed"
                    )
                ),
            }
        gate = await gate_objective_confirmed(deps, state)
        return {
            **(gate or {}),
            **trace_update(
                append_trace(state, node_name="validate_trigger", status="completed")
            ),
        }

    def after_validate_trigger(state: CognitiveGraphState) -> str:
        if entry_blocked_by_objective(state):
            return "persist_cycle"
        errors = state.get("errors") or []
        if any(getattr(e, "error_code", None) == "invalid_trigger" for e in errors):
            return "persist_cycle"
        return "hydrate_context"

    async def hydrate_context(state: CognitiveGraphState) -> dict[str, Any]:
        snapshot_id = state.get("snapshot_id")
        if not snapshot_id or deps.snapshot_repo is None:
            return append_error(
                state,
                node_name="hydrate_context",
                error_code="snapshot_unavailable",
                message="cannot hydrate context without snapshot repository",
            )
        try:
            record, data_quality, _surface, surface_slice = await load_snapshot_truth(
                deps, snapshot_id
            )
        except Exception as exc:
            return append_error(
                state,
                node_name="hydrate_context",
                error_code="snapshot_not_found",
                message=str(exc),
            )
        cycle_id = state.get("cycle_id") or str(uuid4())
        order_projection = None
        position_projection = None
        if deps.projection_loader is not None:
            projection = await deps.projection_loader()
            if projection is not None:
                order_projection = {"orders": [str(o) for o in getattr(projection, "orders", ())]}
                position_projection = {
                    "positions": [str(p) for p in getattr(projection, "positions", ())]
                }
        objective_context = await load_objective_context(deps)
        # Perception-neutral default package (MARKET_STRUCTURE never gets objective).
        context = await assemble_role_context(
            deps,
            agent_role=AgentRole.MARKET_STRUCTURE,
            session_id=state.get("session_id") or deps.session_id,
            cycle_id=cycle_id,
            snapshot=record,
            data_quality=data_quality,
            option_surface_slice=surface_slice,
            order_projection=order_projection,
            position_projection=position_projection,
            objective_context=None,
        )
        return {
            "cycle_id": cycle_id,
            "context_ref": context.context_id,
            "_context_package": context,
            "_data_quality": data_quality,
            "_option_surface_id": str(record.option_surface_id)
            if record.option_surface_id
            else None,
            "_objective_context": objective_context,
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
        # Idempotent recovery: reuse persisted world model for this cycle when present.
        if state.get("world_model") is not None:
            return trace_update(
                append_trace(
                    state,
                    node_name="synthesise_world_model",
                    status="skipped",
                    artifact_ids=(state["world_model"].world_model_id,),
                )
            )
        snapshot_id = state.get("snapshot_id")
        cycle_id = state.get("cycle_id") or ""
        session_id = state.get("session_id") or deps.session_id
        snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
            deps, str(snapshot_id)
        )
        context = await assemble_role_context(
            deps,
            agent_role=AgentRole.WORLD_MODEL_SYNTHESISER,
            session_id=session_id,
            cycle_id=cycle_id,
            snapshot=snapshot,
            data_quality=data_quality,
            option_surface_slice=surface_slice,
            session_artifact_summaries=tuple(
                {
                    "evidence_id": str(e.evidence_id),
                    "agent_role": e.agent_role.value,
                    "claim": e.claim,
                    "direction": e.direction.value,
                    "confidence": e.confidence,
                }
                for e in evidence
            ),
        )
        world_model = await run_world_model_synthesis(
            router=deps.router,
            context=context,
            evidence=evidence,
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
        if (
            state.get("_no_valid_strategy")
            or state.get("_meta_decision_override") == "abandon"
            or entry_blocked_by_objective(state)
        ):
            from uuid import uuid4 as _uuid4

            from joker.cognition.schemas import MetaDecision

            snapshot_raw = state.get("snapshot_id") or state.get("latest_known_snapshot_id")
            reason = "no valid objective strategy scores; retaining no-trade"
            if state.get("_meta_decision_override") == "abandon" or entry_blocked_by_objective(
                state
            ):
                reason = "objective gate blocks new entries; abandoning"
            decision = MetaDecision(
                session_id=state.get("session_id") or deps.session_id,
                snapshot_id=UUID(str(snapshot_raw)),
                prompt_version="objective-no-trade-v1",
                model_call_id=_uuid4(),
                cycle_id=str(state.get("cycle_id") or _uuid4()),
                action=MetaDecisionAction.ABANDON,
                confidence=1.0,
                rationale_summary=reason,
                selected_strategy_id=None,
            )
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
        # Re-assemble meta context with sanitised objective when available.
        objective_context = state.get("_objective_context")
        if objective_context and deps.snapshot_repo is not None and state.get("snapshot_id"):
            try:
                record, data_quality, _surface, surface_slice = await load_snapshot_truth(
                    deps, state["snapshot_id"]
                )
                context = await assemble_role_context(
                    deps,
                    agent_role=AgentRole.META_DECISION,
                    session_id=state.get("session_id") or deps.session_id,
                    cycle_id=state.get("cycle_id") or "",
                    snapshot=record,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    objective_context=objective_context,
                )
            except Exception:
                pass
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
        if state.get("_no_valid_strategy") or entry_blocked_by_objective(state):
            return "persist_cycle"
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
        from joker.cognition.prompt_overrides import get_active_debate_policy

        policy = get_active_debate_policy() or {}
        max_rounds = int(policy.get("maximum_rounds", deps.config.max_debate_rounds))
        if debate_round <= max_rounds:
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
        # Role-specific re-assemble with sanitised ObjectiveContext for goal-aware tactician.
        objective_context = state.get("_objective_context")
        if objective_context and deps.snapshot_repo is not None and state.get("snapshot_id"):
            try:
                record, data_quality, _surface, surface_slice = await load_snapshot_truth(
                    deps, state["snapshot_id"]
                )
                context = await assemble_role_context(
                    deps,
                    agent_role=AgentRole.ENTRY_TACTICIAN,
                    session_id=state.get("session_id") or deps.session_id,
                    cycle_id=state.get("cycle_id") or "",
                    snapshot=record,
                    data_quality=data_quality,
                    option_surface_slice=surface_slice,
                    objective_context=objective_context,
                )
            except Exception:
                pass
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

    async def apply_objective_sizing(state: CognitiveGraphState) -> dict[str, Any]:
        return await apply_objective_sizing_to_proposal(deps, state)

    async def validate_execution_proposal(state: CognitiveGraphState) -> dict[str, Any]:
        proposal = state.get("execution_proposal")
        if proposal is None:
            return append_error(
                state,
                node_name="validate_execution_proposal",
                error_code="missing_proposal",
                message="no execution proposal to validate",
            )
        try:
            snapshot, data_quality, surface, _slice = await load_snapshot_truth(
                deps, str(proposal.snapshot_id)
            )
            projection = None
            if deps.projection_loader is not None:
                projection = await deps.projection_loader()
            truth = build_truth_from_deps(
                snapshot=snapshot,
                data_quality=data_quality,
                option_surface=surface,
                projection=projection,
                already_submitted_proposal_ids=tuple(deps.submitted_proposal_ids),
            )
            validate_and_compile_proposal(
                proposal,
                truth=truth,
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
            snapshot, data_quality, surface, _slice = await load_snapshot_truth(
                deps, str(proposal.snapshot_id)
            )
            projection = None
            if deps.projection_loader is not None:
                projection = await deps.projection_loader()
            truth = build_truth_from_deps(
                snapshot=snapshot,
                data_quality=data_quality,
                option_surface=surface,
                projection=projection,
                already_submitted_proposal_ids=tuple(deps.submitted_proposal_ids),
            )
            provenanced = validate_and_compile_proposal(
                proposal,
                truth=truth,
                evidence_ids=tuple(e.evidence_id for e in state.get("evidence") or []),
            )
        except Exception as exc:
            return append_error(
                state,
                node_name="submit_execution_command",
                error_code="submit_validation_failed",
                message=str(exc),
            )
        from dataclasses import replace

        from joker.runtime.order_action_gateway import (
            OrderActionKind,
            ensure_order_action_gateway,
            provenanced_to_action_request,
        )

        gateway = ensure_order_action_gateway(deps)
        if gateway is not None:
            action = (
                OrderActionKind.PROBE
                if getattr(proposal, "action", None) == "probe"
                else OrderActionKind.ENTRY
            )
            causation = _resolve_entry_causation_event_id(state)
            action_request = provenanced_to_action_request(
                provenanced,
                action=action,
                causation_event_id=causation,
            )
            sizing = state.get("_sizing_decision") or {}
            estimate_id = sizing.get("estimate_id")
            if estimate_id:
                action_request = replace(action_request, estimate_id=str(estimate_id))
            gateway_result = await gateway.submit(action_request)
            if not gateway_result.submitted:
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="gateway_blocked",
                    message=gateway_result.blocked_reason or "order action blocked",
                )
            command_id = gateway_result.client_order_id
            result = gateway_result.broker_order
        else:
            if deps.submit_callback is None:
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="no_submit_callback",
                    message="execution submit callback not configured",
                )
            command_id = provenanced.command.client_order_id
            if deps.provenance_registry is not None:
                from joker.persistence.cognitive_execution_provenance import (
                    ExecutionProvenanceRecord,
                )
                from joker.runtime.execution_runtime import contract_id_for

                await deps.provenance_registry.record(
                    ExecutionProvenanceRecord(
                        client_order_id=command_id,
                        proposal_id=str(provenanced.proposal_id),
                        decision_id=str(provenanced.decision_id),
                        strategy_id=str(provenanced.strategy_id),
                        cycle_id=str(provenanced.cycle_id),
                        snapshot_id=str(provenanced.snapshot_id),
                        contract_id=contract_id_for(provenanced.command.intent.contract),
                        session_id=deps.session_id,
                        kind="entry",
                        causation_event_id=_resolve_entry_causation_event_id(state),
                    )
                )
            result = await deps.submit_callback(provenanced)
            deps.submitted_proposal_ids.add(str(proposal.proposal_id))
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

    async def assess_goal_feasibility(state: CognitiveGraphState) -> dict[str, Any]:
        return await assess_goal_feasibility_node(deps, state)

    async def score_strategies_against_objective(
        state: CognitiveGraphState,
    ) -> dict[str, Any]:
        return await score_strategies_against_objective_node(deps, state)

    async def deterministic_sizing(state: CognitiveGraphState) -> dict[str, Any]:
        return await deterministic_sizing_node(deps, state)

    graph = StateGraph(CognitiveGraphState)
    graph.add_node("validate_trigger", validate_trigger)
    graph.add_node("hydrate_context", hydrate_context)
    graph.add_node("perception", perception)
    graph.add_node("synthesise_world_model", synthesise_world_model)
    graph.add_node("discovery", discovery)
    graph.add_node("strategy", strategy_graph)
    graph.add_node("assess_goal_feasibility", assess_goal_feasibility)
    graph.add_node(
        "score_strategies_against_objective", score_strategies_against_objective
    )
    graph.add_node("select_debate_candidates", select_debate_candidates)
    graph.add_node("debate", debate)
    graph.add_node("meta_decision", meta_decision_node)
    graph.add_node("strategy_switch_revision", strategy_switch_revision)
    graph.add_node("deterministic_sizing", deterministic_sizing)
    graph.add_node("entry_tactician", entry_tactician_node)
    graph.add_node("apply_objective_sizing", apply_objective_sizing)
    graph.add_node("validate_execution_proposal", validate_execution_proposal)
    graph.add_node("submit_execution_command", submit_execution_command)
    graph.add_node("persist_cycle", persist_cycle)
    graph.add_node("persist_pending_cycle", persist_pending_cycle)
    graph.add_node("persist_evidence_request", persist_evidence_request)
    graph.add_node("persist_stale", persist_stale)

    graph.add_edge(START, "validate_trigger")
    graph.add_conditional_edges(
        "validate_trigger",
        after_validate_trigger,
        {
            "hydrate_context": "hydrate_context",
            "persist_cycle": "persist_cycle",
        },
    )
    graph.add_edge("hydrate_context", "perception")
    graph.add_edge("perception", "synthesise_world_model")
    graph.add_edge("synthesise_world_model", "discovery")
    graph.add_edge("discovery", "strategy")
    graph.add_edge("strategy", "assess_goal_feasibility")
    graph.add_edge("assess_goal_feasibility", "score_strategies_against_objective")
    graph.add_edge("score_strategies_against_objective", "select_debate_candidates")
    graph.add_edge("select_debate_candidates", "debate")
    graph.add_edge("debate", "meta_decision")
    graph.add_conditional_edges(
        "meta_decision",
        route_meta_decision,
        {
            "entry_tactician": "deterministic_sizing",
            "persist_cycle": "persist_cycle",
            "persist_pending_cycle": "persist_pending_cycle",
            "persist_evidence_request": "persist_evidence_request",
            "persist_stale": "persist_stale",
            "strategy_switch_revision": "strategy_switch_revision",
        },
    )
    graph.add_edge("deterministic_sizing", "entry_tactician")
    graph.add_conditional_edges(
        "strategy_switch_revision",
        after_switch_route,
        {"debate": "debate", "meta_decision": "meta_decision"},
    )

    def after_validate_route(state: CognitiveGraphState) -> str:
        if state.get("stale_decision"):
            return "persist_stale"
        errors = state.get("errors") or []
        if any(
            e.error_code
            in {
                "validation_failed",
                "sizing_rejected",
                "entries_blocked",
                "missing_proposal",
            }
            for e in errors
        ):
            return "persist_cycle"
        return "submit_execution_command"

    def after_apply_sizing(state: CognitiveGraphState) -> str:
        errors = state.get("errors") or []
        if any(
            e.error_code in {"sizing_rejected", "entries_blocked", "missing_proposal"}
            for e in errors
        ):
            return "persist_cycle"
        return "validate_execution_proposal"

    graph.add_conditional_edges(
        "validate_execution_proposal",
        after_validate_route,
        {
            "submit_execution_command": "submit_execution_command",
            "persist_stale": "persist_stale",
            "persist_cycle": "persist_cycle",
        },
    )
    graph.add_edge("entry_tactician", "apply_objective_sizing")
    graph.add_conditional_edges(
        "apply_objective_sizing",
        after_apply_sizing,
        {
            "validate_execution_proposal": "validate_execution_proposal",
            "persist_cycle": "persist_cycle",
        },
    )
    graph.add_edge("submit_execution_command", "persist_cycle")
    graph.add_edge("persist_cycle", END)
    graph.add_edge("persist_pending_cycle", END)
    graph.add_edge("persist_evidence_request", END)
    graph.add_edge("persist_stale", END)

    compiled_kwargs: dict[str, Any] = {}
    if deps.checkpointer is not None:
        compiled_kwargs["checkpointer"] = deps.checkpointer
    return graph.compile(**compiled_kwargs)


def _resolve_entry_causation_event_id(state: CognitiveGraphState) -> str | None:
    """Factual horizon-start event for entry provenance (never a fill).

    Preference order:
    1. explicit decision-completed event ID
    2. explicit execution-proposal event ID
    3. typed cognitive-cycle start/trigger event ID
    4. unavailable (None)
    """
    for key in (
        "decision_completed_event_id",
        "execution_proposal_event_id",
    ):
        raw = state.get(key)  # type: ignore[literal-required]
        if raw:
            return str(raw)
    proposal = state.get("execution_proposal")
    if proposal is not None:
        for attr in ("event_id", "proposal_event_id", "causation_event_id"):
            raw = getattr(proposal, attr, None)
            if raw:
                return str(raw)
    meta = state.get("meta_decision")
    if meta is not None:
        for attr in ("event_id", "decision_event_id", "causation_event_id"):
            raw = getattr(meta, attr, None)
            if raw:
                return str(raw)
    trigger = state.get("trigger_event_id")
    if trigger:
        # Explicit cycle-start/trigger anchor — not an inferred first window event.
        return str(trigger)
    return None


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

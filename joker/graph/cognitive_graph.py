"""Parent cognitive decision LangGraph (Task 2 spec section 19)."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

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
from joker.runtime.portfolio_recovery import PortfolioRecoveryCoordinator
from joker.runtime.recovery_mode import recovery_mode_value
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
                    message=("entries blocked: " + (", ".join(reasons) or "reconciliation")),
                    recoverable=True,
                ),
                **trace_update(
                    append_trace(state, node_name="validate_trigger", status="completed")
                ),
            }
        gate = await gate_objective_confirmed(deps, state)
        from joker.graph.observable_events import publish_graph_cycle_started

        await publish_graph_cycle_started(deps, {**state, **(gate or {})})
        return {
            **(gate or {}),
            **trace_update(append_trace(state, node_name="validate_trigger", status="completed")),
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
                raw_orders = getattr(projection, "orders", ()) or ()
                order_values = raw_orders.values() if isinstance(raw_orders, dict) else raw_orders
                order_projection = {
                    "orders": [
                        (
                            order.model_dump(mode="json")
                            if hasattr(order, "model_dump")
                            else dict(vars(order))
                            if hasattr(order, "__dict__")
                            else order
                        )
                        for order in order_values
                    ]
                }
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
            "_order_projection": order_projection,
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

    async def finalize_portfolio_review(
        state: CognitiveGraphState,
    ) -> dict[str, Any]:
        decision = state.get("_target_portfolio_decision")
        context = state.get("_portfolio_review_context")
        reviews = list(state.get("_portfolio_debate_reviews") or [])
        if not isinstance(decision, dict) or not isinstance(context, dict):
            return trace_update(
                append_trace(
                    state,
                    node_name="finalize_portfolio_review",
                    status="skipped",
                )
            )
        original = dict(decision)
        recommendations = {str(review.get("finalizer_recommendation") or "") for review in reviews}
        if "reoptimize" in recommendations:
            return {
                "_provisional_target_portfolio_decision": original,
                "_portfolio_review_finalization": {
                    "action": "request_reoptimization",
                    "preserved_decision_id": decision.get("decision_id"),
                    "review_ids": [review.get("review_id") for review in reviews],
                },
                "_block_new_entries": True,
                **append_error(
                    state,
                    node_name="finalize_portfolio_review",
                    error_code="target_attainment_recalculation_required",
                    message="portfolio reviewer requested full reoptimization",
                ),
            }
        if "wait" in recommendations:
            from joker.objectives.portfolio_review import (
                normalize_reviewer_forced_wait,
            )

            finalized, target_decision, audit = normalize_reviewer_forced_wait(
                portfolio_decision=original,
                legacy_decision=state.get("_target_attainment_decision"),
            )
            return {
                "_provisional_target_portfolio_decision": audit,
                "_portfolio_review_rejected_decision_audit": audit,
                "_target_portfolio_decision": finalized,
                "_target_authorized_positions": [],
                "_target_attainment_decision": target_decision,
                "_target_attainment_action": "wait",
                "_target_attainment_strategy_id": None,
                "_target_attainment_contract_id": None,
                "_target_attainment_quantity": 0,
                "_target_attainment_authoritative": True,
                "_quantity_grid": finalized.get("quantity_grid") or [],
                "_portfolio_grid": finalized.get("portfolio_evaluations") or [],
                "_sizing_decision": None,
                "execution_proposal": None,
                "execution_command_id": None,
                "_execution_command_ids": [],
                "execution_result_ref": None,
                "_portfolio_review_finalization": {
                    "action": "wait",
                    "preserved_decision_id": decision.get("decision_id"),
                    "review_ids": [review.get("review_id") for review in reviews],
                },
                **trace_update(
                    append_trace(
                        state,
                        node_name="finalize_portfolio_review",
                        status="completed",
                    )
                ),
            }
        return {
            "_provisional_target_portfolio_decision": original,
            "_portfolio_review_finalization": {
                "action": "preserve_exact_tuple",
                "preserved_decision_id": decision.get("decision_id"),
                "review_ids": [review.get("review_id") for review in reviews],
            },
            **trace_update(
                append_trace(
                    state,
                    node_name="finalize_portfolio_review",
                    status="completed",
                )
            ),
        }

    async def meta_decision_node(state: CognitiveGraphState) -> dict[str, Any]:
        context = state.get("_context_package")  # type: ignore[typeddict-item]
        if not isinstance(context, ContextPackage):
            return {}
        from uuid import uuid4 as _uuid4

        from joker.cognition.schemas import MetaDecision

        ta_authoritative = bool(state.get("_target_attainment_authoritative"))
        ta_action = str(state.get("_target_attainment_action") or "")
        ta_strategy = state.get("_target_attainment_strategy_id")

        if (
            state.get("_no_valid_strategy")
            or state.get("_meta_decision_override") == "abandon"
            or entry_blocked_by_objective(state)
            or (ta_authoritative and ta_action in {"wait", "block"})
        ):
            snapshot_raw = state.get("snapshot_id") or state.get("latest_known_snapshot_id")
            reason = "no valid objective strategy scores; retaining no-trade"
            if ta_authoritative and ta_action in {"wait", "block"}:
                reason = f"target_attainment_{ta_action}: " + ",".join(
                    (state.get("_target_attainment_decision") or {}).get("reason_codes")
                    or [ta_action]
                )
            elif state.get("_meta_decision_override") == "abandon" or entry_blocked_by_objective(
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
                "_meta_target_review": {
                    "role": "review_only",
                    "target_action": ta_action or "none",
                    "overrode_execution": True,
                },
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
        meta_review: dict[str, Any] = {
            "role": "review_only" if ta_authoritative else "selector",
            "llm_action": decision.action.value,
            "llm_selected_strategy_id": (
                str(decision.selected_strategy_id) if decision.selected_strategy_id else None
            ),
            "target_action": ta_action or None,
            "target_strategy_id": str(ta_strategy) if ta_strategy else None,
        }
        # Under target-attainment, meta may challenge (request evidence / abandon)
        # but cannot replace the executable strategy tuple.
        if ta_authoritative and ta_action == "enter" and ta_strategy:
            challenge = decision.action in {
                MetaDecisionAction.REQUEST_MORE_EVIDENCE,
                MetaDecisionAction.ABANDON,
            }
            meta_review["challenge"] = challenge
            if challenge:
                meta_review["result"] = "challenge_blocks_entry_pending_recalc"
                # Persist challenge; do not execute an alternate strategy.
                decision = decision.model_copy(
                    update={
                        "action": MetaDecisionAction.ABANDON,
                        "selected_strategy_id": None,
                        "rationale_summary": (
                            "meta_challenge_of_target_attainment: " + decision.rationale_summary
                        ),
                    }
                )
            else:
                # Force authoritative strategy; ignore LLM strategy substitution.
                decision = decision.model_copy(
                    update={
                        "action": MetaDecisionAction.EXECUTE,
                        "selected_strategy_id": UUID(str(ta_strategy)),
                        "rationale_summary": (
                            "target_attainment_authoritative_tuple; meta_support: "
                            + decision.rationale_summary
                        ),
                    }
                )
                meta_review["result"] = "target_tuple_enforced"
                meta_review["enforced_strategy_id"] = str(ta_strategy)
        if deps.decision_repo is not None:
            await deps.decision_repo.append_meta(decision)
        return {
            "meta_decision": decision,
            "_meta_target_review": meta_review,
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
        ta_authoritative = bool(state.get("_target_attainment_authoritative"))
        ta_action = str(state.get("_target_attainment_action") or "")
        if ta_authoritative and ta_action in {"wait", "block"}:
            return "persist_cycle"
        if ta_authoritative and ta_action == "enter":
            # Challenges become ABANDON above; only EXECUTE proceeds.
            if meta.action in {MetaDecisionAction.EXECUTE, MetaDecisionAction.PROBE}:
                return "entry_tactician"
            return "persist_cycle"
        action = meta.action
        if action in {MetaDecisionAction.EXECUTE, MetaDecisionAction.PROBE}:
            return "entry_tactician"
        if action == MetaDecisionAction.DELAY:
            return "persist_pending_cycle"
        if action == MetaDecisionAction.REQUEST_MORE_EVIDENCE:
            return "persist_evidence_request"
        if action == MetaDecisionAction.SWITCH_STRATEGY:
            if ta_authoritative:
                # Strategy switches are not an independent selector under TA.
                return "persist_cycle"
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
        # Enforce target-attainment strategy identity before tactician runs.
        if bool(state.get("_target_attainment_authoritative")):
            ta_sid = state.get("_target_attainment_strategy_id")
            if ta_sid is not None and str(strategy.strategy_id) != str(ta_sid):
                return append_error(
                    state,
                    node_name="entry_tactician",
                    error_code="target_attainment_strategy_mismatch",
                    message=(
                        "entry tactician refused: strategy differs from "
                        "authoritative target-attainment selection"
                    ),
                )
        proposal = await run_entry_tactician(
            state=state,
            router=deps.router,
            context=context,
            meta_decision=meta,
            strategy=strategy,
        )
        # Authoritative contract + quantity overwrite — tactician cannot change them.
        if bool(state.get("_target_attainment_authoritative")):
            ta_cid = state.get("_target_attainment_contract_id")
            ta_qty = int(state.get("_target_attainment_quantity") or 0)
            authorized_positions = list(state.get("_target_authorized_positions") or [])
            if not ta_cid or ta_qty < 1:
                return append_error(
                    state,
                    node_name="entry_tactician",
                    error_code="target_attainment_tuple_incomplete",
                    message="authoritative target-attainment tuple missing contract/quantity",
                )
            if not proposal.legs:
                return append_error(
                    state,
                    node_name="entry_tactician",
                    error_code="target_attainment_missing_legs",
                    message="entry tactician produced no legs for authoritative tuple",
                )
            template_leg = proposal.legs[0]
            if authorized_positions:
                portfolio_decision = dict(state.get("_target_portfolio_decision") or {})
                component_count = len(authorized_positions)
                new_legs = [
                    template_leg.model_copy(
                        update={
                            "leg_id": uuid4(),
                            "strategy_id": UUID(str(position["strategy_id"])),
                            "contract_id": str(position["contract_id"]),
                            "quantity": int(position["quantity"]),
                            "limit_price": Decimal(str(position["evaluation_premium"])),
                            "evaluation_premium": Decimal(str(position["evaluation_premium"])),
                            "capital_allocation": Decimal(str(position["capital_allocation"])),
                            "authorized_position_tuple_id": UUID(
                                str(position["position_tuple_id"])
                            ),
                            "target_portfolio_decision_id": UUID(str(position["decision_id"])),
                            "selected_portfolio_id": (
                                UUID(str(portfolio_decision["selected_portfolio_id"]))
                                if portfolio_decision.get("selected_portfolio_id")
                                else None
                            ),
                            "component_index": index,
                            "component_count": component_count,
                            "evaluated_objective_version": int(position["objective_version"]),
                            "evaluated_objective_fingerprint": position.get(
                                "evaluated_objective_fingerprint"
                            ),
                            "original_decision_snapshot_id": UUID(str(position["snapshot_id"])),
                            "evaluated_at_exchange_time": (
                                datetime.fromisoformat(str(position["evaluated_at_exchange_time"]))
                                if position.get("evaluated_at_exchange_time")
                                else None
                            ),
                            "decision_valid_until_exchange_time": (
                                datetime.fromisoformat(
                                    str(position["decision_valid_until_exchange_time"])
                                )
                                if position.get("decision_valid_until_exchange_time")
                                else None
                            ),
                            "maximum_decision_age_seconds": position.get(
                                "maximum_decision_age_seconds"
                            ),
                            "required_resolution_horizon_seconds": position.get(
                                "required_resolution_horizon_seconds"
                            ),
                            "sequence_order": index + 1,
                        }
                    )
                    for index, position in enumerate(authorized_positions)
                ]
            else:
                new_legs = [
                    template_leg.model_copy(
                        update={
                            "contract_id": str(ta_cid),
                            "quantity": ta_qty,
                        }
                    )
                ]
            proposal = proposal.model_copy(
                update={
                    "legs": tuple(new_legs),
                    "strategy_id": UUID(str(state.get("_target_attainment_strategy_id"))),
                }
            )
            if str(proposal.legs[0].contract_id) != str(ta_cid):
                return append_error(
                    state,
                    node_name="entry_tactician",
                    error_code="target_attainment_contract_mismatch",
                    message="failed to bind authoritative contract_id on proposal",
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
                now=(
                    deps.clock.now()
                    if deps.clock is not None and hasattr(deps.clock, "now")
                    else None
                ),
            )
            for leg in proposal.legs:
                validate_and_compile_proposal(
                    proposal.model_copy(update={"legs": (leg,)}),
                    truth=truth,
                    evidence_ids=tuple(e.evidence_id for e in state.get("evidence") or []),
                )
        except Exception as exc:
            if "stale" in str(exc).lower():
                return {
                    "stale_decision": True,
                    **append_error(
                        state,
                        node_name="validate_execution_proposal",
                        error_code="stale_proposal",
                        message=str(exc),
                    ),
                }
            return append_error(
                state,
                node_name="validate_execution_proposal",
                error_code="validation_failed",
                message=str(exc),
            )
        return trace_update(
            append_trace(state, node_name="validate_execution_proposal", status="completed")
        )

    async def submit_execution_command(state: CognitiveGraphState) -> dict[str, Any]:
        proposal = state.get("execution_proposal")
        if proposal is None:
            return {}
        if state.get("execution_command_id"):
            return trace_update(
                append_trace(state, node_name="submit_execution_command", status="skipped")
            )
        from dataclasses import replace

        from joker.runtime.order_action_gateway import (
            OrderActionKind,
            ensure_order_action_gateway,
            provenanced_to_action_request,
        )
        from joker.graph.observable_events import (
            publish_execution_observable_event,
        )

        gateway = ensure_order_action_gateway(deps)
        authorized_positions = list(state.get("_target_authorized_positions") or [])
        portfolio_decision = dict(state.get("_target_portfolio_decision") or {})
        command_ids: list[str] = []
        result_refs: list[str] = []
        submitted_by_tuple: dict[str, Any] = {}
        portfolio_execution_repo = None
        portfolio_owner = None
        portfolio_recovery = None
        if authorized_positions:
            if deps.provenance_registry is None and deps.db_path is not None:
                from joker.persistence.cognitive_execution_provenance import (
                    CognitiveExecutionProvenanceRegistry,
                )

                deps.provenance_registry = CognitiveExecutionProvenanceRegistry(deps.db_path)
                await deps.provenance_registry.initialize()
            if deps.provenance_registry is None:
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="portfolio_execution_store_unavailable",
                    message="durable portfolio execution store unavailable",
                )
            portfolio_execution_repo = deps.provenance_registry.portfolio_executions
            from joker.persistence.cognitive_execution_provenance import (
                PortfolioExecutionOwner,
            )

            runtime_session = (
                deps.execution_runtime.session_id
                if deps.execution_runtime is not None
                else deps.session_id
            )
            broker_account_id = (
                deps.execution_runtime.broker_account_identity
                if deps.execution_runtime is not None
                else deps.broker_account_identity
            )
            if not broker_account_id:
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="portfolio_execution_owner_unavailable",
                    message="broker account identity required for portfolio ownership",
                )
            if runtime_session != deps.session_id or (
                state.get("session_id") and str(state["session_id"]) != deps.session_id
            ):
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="portfolio_execution_owner_mismatch",
                    message="graph and execution runtime session ownership differ",
                )
            if deps.clock is None:
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="portfolio_execution_owner_unavailable",
                    message="exchange clock required for portfolio ownership",
                )
            from joker.runtime.cognitive_session import (
                stable_cognitive_session_trading_date,
            )

            stable_trading_date = stable_cognitive_session_trading_date(deps.session_id)
            portfolio_owner = PortfolioExecutionOwner(
                session_id=deps.session_id,
                broker_account_identity=broker_account_id,
                trading_date=(
                    stable_trading_date.isoformat()
                    if stable_trading_date is not None
                    else deps.clock.trading_date().isoformat()
                ),
            )
            portfolio_recovery = PortfolioRecoveryCoordinator(
                execution_runtime=deps.execution_runtime,
                provenance_registry=deps.provenance_registry,
                stable_owner=portfolio_owner,
                clock=deps.clock,
                objective_service=deps.objective_service,
                recovery_mode=recovery_mode_value(deps),
            )
        if deps.provenance_registry is not None and portfolio_decision.get("decision_id"):
            prior_components = await deps.provenance_registry.list_by_target_portfolio_decision_id(
                str(portfolio_decision["decision_id"])
            )
            submitted_by_tuple = {
                str((record.extra or {}).get("authorized_position_tuple_id")): record
                for record in prior_components
                if (record.extra or {}).get("authorized_position_tuple_id")
            }
            prior_indexes = sorted(
                int((record.extra or {}).get("component_index"))
                for record in prior_components
                if (record.extra or {}).get("component_index") is not None
            )
            if prior_indexes != list(range(len(prior_indexes))):
                return {
                    "_block_new_entries": True,
                    **append_error(
                        state,
                        node_name="submit_execution_command",
                        error_code="target_attainment_recalculation_required",
                        message="persisted portfolio components are non-contiguous",
                    ),
                }

        if authorized_positions and portfolio_execution_repo is not None:
            from joker.persistence.cognitive_execution_provenance import (
                PortfolioComponentStatus,
                PortfolioExecutionComponentRecord,
                stable_portfolio_client_order_id,
            )

            evaluated_timestamp = portfolio_decision.get("evaluated_at_exchange_time")
            if not evaluated_timestamp:
                return append_error(
                    state,
                    node_name="submit_execution_command",
                    error_code="target_attainment_recalculation_required",
                    message="portfolio evaluation timestamp is missing",
                )
            for component_index, position in enumerate(authorized_positions):
                tuple_id = str(position["position_tuple_id"])
                legacy = submitted_by_tuple.get(tuple_id)
                durable_existing = await portfolio_execution_repo.get(tuple_id)
                if durable_existing is not None and not portfolio_owner.matches(
                    durable_existing
                ):
                    return append_error(
                        state,
                        node_name="submit_execution_command",
                        error_code="portfolio_execution_owner_mismatch",
                        message="authorized tuple is owned by a different durable session",
                    )
                client_order_id = (
                    durable_existing.client_order_id
                    if durable_existing is not None
                    else legacy.client_order_id
                    if legacy is not None
                    else stable_portfolio_client_order_id(
                        str(portfolio_decision["decision_id"]), tuple_id
                    )
                )
                record = PortfolioExecutionComponentRecord(
                    session_id=portfolio_owner.session_id,
                    origin_run_id=(
                        durable_existing.origin_run_id
                        if durable_existing is not None
                        else deps.run_id
                    ),
                    broker_account_identity=portfolio_owner.broker_account_identity,
                    trading_date=portfolio_owner.trading_date,
                    target_portfolio_decision_id=str(portfolio_decision["decision_id"]),
                    selected_portfolio_id=(
                        str(portfolio_decision["selected_portfolio_id"])
                        if portfolio_decision.get("selected_portfolio_id")
                        else None
                    ),
                    authorized_position_tuple_id=tuple_id,
                    component_index=component_index,
                    component_count=len(authorized_positions),
                    strategy_id=str(position["strategy_id"]),
                    contract_id=str(position["contract_id"]),
                    authorized_quantity=int(position["quantity"]),
                    capital_allocation=Decimal(str(position["capital_allocation"])),
                    client_order_id=client_order_id,
                    status=PortfolioComponentStatus.AUTHORIZED,
                    submitted_quantity=0,
                    filled_quantity=0,
                    remaining_quantity=int(position["quantity"]),
                    original_decision_snapshot_id=str(position["snapshot_id"]),
                    evaluated_objective_version=int(position["objective_version"]),
                    evaluated_timestamp=str(evaluated_timestamp),
                    last_resumed_run_id=(
                        durable_existing.last_resumed_run_id
                        if durable_existing is not None
                        else None
                    ),
                    resume_count=(
                        durable_existing.resume_count
                        if durable_existing is not None
                        else 0
                    ),
                    last_resumed_at=(
                        durable_existing.last_resumed_at
                        if durable_existing is not None
                        else None
                    ),
                    extra={
                        "stable_owner": {
                            "session_id": portfolio_owner.session_id,
                            "broker_account_identity": (
                                portfolio_owner.broker_account_identity
                            ),
                            "trading_date": portfolio_owner.trading_date,
                        },
                        "origin_run_id": deps.run_id,
                        "portfolio_decision": portfolio_decision,
                        "authorized_positions": authorized_positions,
                        "execution_proposal": proposal.model_dump(mode="json"),
                    },
                )
                stored = await portfolio_execution_repo.authorize(record)
                if legacy is not None and stored.status == PortfolioComponentStatus.AUTHORIZED:
                    await portfolio_execution_repo.transition(
                        tuple_id,
                        owner=portfolio_owner,
                        status=PortfolioComponentStatus.SUBMITTED,
                        submitted_quantity=int(position["quantity"]),
                        latest_validation_snapshot_id=legacy.snapshot_id,
                        submission_objective_version=(
                            int((legacy.extra or {}).get("submission_objective_version"))
                            if (legacy.extra or {}).get("submission_objective_version") is not None
                            else None
                        ),
                        extra_update={"legacy_provenance_recovered": True},
                    )

        async def _mark_remaining_reoptimization(start_index: int, reason: str) -> None:
            if portfolio_recovery is None:
                return
            objective_status = None
            if state.get("_reconciliation_only_recovery") and deps.objective_service is not None:
                objective = await deps.objective_service.get_state()
                objective_status = str(getattr(objective, "status", "unknown") or "unknown")
            await portfolio_recovery.request_suffix_reoptimization(
                decision_id=str(portfolio_decision.get("decision_id") or ""),
                reason=reason,
                origin_run_id=deps.run_id,
                start_component_index=start_index,
                state=state,
                terminal_recovery=bool(state.get("_reconciliation_only_recovery")),
                latest_snapshot_id=str(
                    state.get("snapshot_id") or state.get("latest_known_snapshot_id") or ""
                )
                or None,
                objective_status=objective_status,
            )

        from joker.objectives.decision_fingerprint import (
            ObjectiveDecisionFingerprint,
            validate_decision_submission_timing,
        )
        from joker.runtime.order_action_gateway import (
            working_orders_from_projection,
        )

        evaluated_fingerprint_raw = portfolio_decision.get("evaluated_objective_fingerprint")
        expected_fingerprint = (
            ObjectiveDecisionFingerprint.from_json(evaluated_fingerprint_raw)
            if evaluated_fingerprint_raw
            else None
        )

        def _current_fingerprint(
            objective_state: Any,
            *,
            working_order_count: int,
        ) -> ObjectiveDecisionFingerprint:
            permission = getattr(deps, "entry_permission", None)
            broker_eligible = not bool(getattr(deps, "kill_switch", False))
            if permission is not None:
                broker_eligible = broker_eligible and bool(getattr(permission, "permitted", False))
            runtime = getattr(deps, "execution_runtime", None)
            reconciliation_eligible = (
                runtime is None or getattr(runtime, "unresolved_reconciliation", None) is None
            )
            broker = getattr(runtime, "_broker", None)
            broker_identity = type(broker).__qualname__ if broker is not None else "unconfigured"
            return ObjectiveDecisionFingerprint.from_state(
                objective_state,
                working_order_count=working_order_count,
                broker_identity=broker_identity,
                broker_eligible=broker_eligible,
                reconciliation_eligible=reconciliation_eligible,
            )

        async def _persist_filled_continuation(
            component: Any,
            *,
            filled_quantity: int,
            broker_order_id: str | None = None,
        ) -> Any:
            if portfolio_recovery is None:
                raise RuntimeError("deterministic portfolio recovery is unavailable")
            latest_snapshot = None
            if deps.snapshot_repo is not None:
                latest_snapshot = await deps.snapshot_repo.get_latest(deps.session_id)
            latest_snapshot_id = (
                str(getattr(latest_snapshot, "snapshot_id", "") or "")
                if latest_snapshot is not None
                else None
            )
            if not latest_snapshot_id:
                raise RuntimeError("post-fill snapshot is unavailable")
            return await portfolio_recovery.persist_filled_continuation(
                component,
                latest_snapshot_id=latest_snapshot_id,
                filled_quantity=filled_quantity,
                broker_order_id=broker_order_id,
            )

        if portfolio_execution_repo is not None:
            persisted = await portfolio_execution_repo.list_by_decision(
                str(portfolio_decision.get("decision_id") or ""),
                owner=portfolio_owner,
            )
            for component in persisted:
                if component.status != PortfolioComponentStatus.FILLED:
                    break
                if not component.continuation_ready:
                    await _mark_remaining_reoptimization(
                        component.component_index + 1,
                        "filled_component_missing_post_fill_checkpoint",
                    )
                    return {
                        "_block_new_entries": True,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="post_fill_continuation_checkpoint_missing",
                            message="filled component lacks durable continuation truth",
                        ),
                    }
                expected_fingerprint = ObjectiveDecisionFingerprint.from_json(
                    component.post_fill_objective_fingerprint
                )

        for index, leg in enumerate(proposal.legs):
            timing_evidence = None
            position = authorized_positions[index] if index < len(authorized_positions) else None
            tuple_id = str(position.get("position_tuple_id") or "") if position is not None else ""
            projection = (
                await deps.projection_loader() if deps.projection_loader is not None else None
            )
            component_record = (
                await portfolio_execution_repo.get(tuple_id)
                if portfolio_execution_repo is not None and tuple_id
                else None
            )
            if component_record is not None:
                if portfolio_owner is None or not portfolio_owner.matches(component_record):
                    return {
                        "_block_new_entries": True,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="portfolio_execution_owner_mismatch",
                            message="persisted component owner does not match runtime",
                        ),
                    }
                order = None
                orders = (
                    projection.get("orders")
                    if isinstance(projection, dict)
                    else getattr(projection, "orders", None)
                ) or {}
                if isinstance(orders, dict):
                    order = orders.get(component_record.client_order_id)
                else:
                    order = next(
                        (
                            item
                            for item in orders
                            if str(getattr(item, "client_order_id", ""))
                            == component_record.client_order_id
                        ),
                        None,
                    )
                if order is not None:
                    raw_status = (
                        order.get("status")
                        if isinstance(order, dict)
                        else getattr(order, "status", "")
                    )
                    status_value = str(getattr(raw_status, "value", raw_status) or "").lower()
                    filled_quantity = int(
                        (order.get("filled_qty") or order.get("filled_quantity") or 0)
                        if isinstance(order, dict)
                        else getattr(order, "filled_qty", 0) or getattr(order, "filled_quantity", 0)
                    )
                    status_map = {
                        "submitted": PortfolioComponentStatus.SUBMITTED,
                        "accepted": PortfolioComponentStatus.WORKING,
                        "open": PortfolioComponentStatus.WORKING,
                        "pending": PortfolioComponentStatus.WORKING,
                        "working": PortfolioComponentStatus.WORKING,
                        "partially_filled": (PortfolioComponentStatus.PARTIALLY_FILLED),
                        "filled": PortfolioComponentStatus.FILLED,
                        "rejected": PortfolioComponentStatus.REJECTED,
                        "cancelled": PortfolioComponentStatus.CANCELLED,
                    }
                    mapped_status = status_map.get(status_value)
                    terminal_statuses = {
                        PortfolioComponentStatus.FILLED,
                        PortfolioComponentStatus.REJECTED,
                        PortfolioComponentStatus.CANCELLED,
                        PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
                    }
                    if (
                        mapped_status is not None
                        and component_record.status not in terminal_statuses
                    ):
                        if mapped_status == PortfolioComponentStatus.FILLED:
                            component_record = await _persist_filled_continuation(
                                component_record,
                                filled_quantity=filled_quantity,
                                broker_order_id=component_record.broker_order_id,
                            )
                            expected_fingerprint = ObjectiveDecisionFingerprint.from_json(
                                component_record.post_fill_objective_fingerprint
                            )
                        else:
                            component_record = await portfolio_execution_repo.transition(
                                tuple_id,
                                owner=portfolio_owner,
                                status=mapped_status,
                                submitted_quantity=(component_record.authorized_quantity),
                                filled_quantity=filled_quantity,
                                last_reconciliation_timestamp=(
                                    deps.clock.now().isoformat()
                                    if deps.clock is not None
                                    else datetime.now(timezone.utc).isoformat()
                                ),
                            )
                if component_record.status in {
                    PortfolioComponentStatus.SUBMITTED,
                    PortfolioComponentStatus.WORKING,
                    PortfolioComponentStatus.PARTIALLY_FILLED,
                }:
                    command_ids.append(component_record.client_order_id)
                    result_refs.append(
                        component_record.broker_order_id or component_record.client_order_id
                    )
                    break
                if component_record.status == PortfolioComponentStatus.FILLED:
                    command_ids.append(component_record.client_order_id)
                    result_refs.append(
                        component_record.broker_order_id or component_record.client_order_id
                    )
                    continue
                if component_record.status in {
                    PortfolioComponentStatus.REJECTED,
                    PortfolioComponentStatus.CANCELLED,
                    PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
                }:
                    if component_record.status in {
                        PortfolioComponentStatus.REJECTED,
                        PortfolioComponentStatus.CANCELLED,
                    }:
                        await _mark_remaining_reoptimization(
                            index + 1,
                            "prior_component_" + component_record.status.value.lower(),
                        )
                    break
            if state.get("_reconciliation_only_recovery"):
                await _mark_remaining_reoptimization(
                    index,
                    "reconciliation_only_resume_no_new_entries",
                )
                return {
                    "_block_new_entries": True,
                    "_execution_command_ids": command_ids,
                    **append_error(
                        state,
                        node_name="submit_execution_command",
                        error_code="reconciliation_only_resume_blocks_new_component",
                        message=(
                            "terminal objective recovery may reconcile existing broker work "
                            "but cannot submit a new portfolio component"
                        ),
                    ),
                }
            if position is not None and deps.objective_service is not None:
                if deps.clock is not None and hasattr(
                    deps.objective_service, "recompute_from_truth"
                ):
                    await deps.objective_service.recompute_from_truth(now=deps.clock.now())
                current_objective = await deps.objective_service.get_state()
                objective_gate_reasons: list[str] = []
                if (
                    deps.clock is not None
                    and current_objective.deadline_exchange_time is not None
                    and deps.clock.now() >= current_objective.deadline_exchange_time
                ):
                    objective_gate_reasons.append("objective_deadline_reached")
                if (
                    current_objective.status == "target_reached"
                    or current_objective.required_profit_remaining_usd <= 0
                ):
                    objective_gate_reasons.append("target_already_reached")
                if current_objective.entries_paused:
                    objective_gate_reasons.append("entries_paused")
                if current_objective.truth_degraded:
                    objective_gate_reasons.append("truth_degraded")
                if objective_gate_reasons:
                    reason = ",".join(objective_gate_reasons)
                    await _mark_remaining_reoptimization(index, reason)
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message=("objective gate changed before component: " + reason),
                        ),
                    }
                if deps.clock is None or not hasattr(deps.clock, "now"):
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message="exchange clock unavailable for decision-age validation",
                        ),
                    }
                submission_exchange_time = deps.clock.now()
                evaluated_at_raw = position.get(
                    "evaluated_at_exchange_time"
                ) or portfolio_decision.get("evaluated_at_exchange_time")
                maximum_age_raw = position.get(
                    "maximum_decision_age_seconds"
                ) or portfolio_decision.get("maximum_decision_age_seconds")
                required_horizon = int(
                    position.get("required_resolution_horizon_seconds")
                    or portfolio_decision.get("required_resolution_horizon_seconds")
                    or 0
                )
                if (
                    not evaluated_at_raw
                    or maximum_age_raw is None
                    or current_objective.deadline_exchange_time is None
                ):
                    await _mark_remaining_reoptimization(
                        index, "decision_timing_provenance_incomplete"
                    )
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message="decision timing provenance is incomplete",
                        ),
                    }
                evaluated_at = (
                    evaluated_at_raw
                    if isinstance(evaluated_at_raw, datetime)
                    else datetime.fromisoformat(str(evaluated_at_raw))
                )
                timing_evidence = validate_decision_submission_timing(
                    evaluated_at_exchange_time=evaluated_at,
                    maximum_decision_age_seconds=int(maximum_age_raw),
                    submission_exchange_time=submission_exchange_time,
                    deadline_exchange_time=current_objective.deadline_exchange_time,
                    required_resolution_horizon_seconds=required_horizon,
                )
                portfolio_decision.update(timing_evidence.as_dict())
                if not timing_evidence.valid:
                    await _mark_remaining_reoptimization(
                        index, ",".join(timing_evidence.reason_codes)
                    )
                    await publish_execution_observable_event(
                        deps,
                        state,
                        reoptimization_required=True,
                        payload={
                            "component_index": index,
                            **timing_evidence.as_dict(),
                        },
                    )
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        "_target_portfolio_decision": portfolio_decision,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message=(
                                "decision timing invalid before component: "
                                + ",".join(timing_evidence.reason_codes)
                            ),
                        ),
                    }
                working_count = len(working_orders_from_projection(projection))
                submission_fingerprint = _current_fingerprint(
                    current_objective,
                    working_order_count=working_count,
                )
                eligibility_reasons: list[str] = []
                if not submission_fingerprint.broker_eligible:
                    eligibility_reasons.append("broker_ineligible")
                if not submission_fingerprint.reconciliation_eligible:
                    eligibility_reasons.append("reconciliation_ineligible")
                if eligibility_reasons:
                    reason = ",".join(eligibility_reasons)
                    await _mark_remaining_reoptimization(index, reason)
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message=("execution eligibility changed before component: " + reason),
                        ),
                    }
                material_differences = (
                    expected_fingerprint.material_differences(submission_fingerprint)
                    if expected_fingerprint is not None
                    else ("evaluated_objective_fingerprint_missing",)
                )
                if material_differences:
                    await _mark_remaining_reoptimization(
                        index,
                        "material_objective_truth_changed:" + ",".join(material_differences),
                    )
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message=(
                                "material objective truth changed before component: "
                                + ",".join(material_differences)
                            ),
                        ),
                    }
                remaining_components = len(authorized_positions) - index
                projected_positions = (
                    int(current_objective.open_position_count)
                    + working_count
                    + remaining_components
                )
                if projected_positions > int(current_objective.max_concurrent_positions):
                    await _mark_remaining_reoptimization(
                        index, "maximum_concurrent_positions_changed"
                    )
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message=(
                                "position limit changed before sequential "
                                "portfolio component submission"
                            ),
                        ),
                    }
                remaining_allocation = sum(
                    (
                        Decimal(str(item.get("capital_allocation") or "0"))
                        for item in authorized_positions[index:]
                    ),
                    Decimal("0"),
                )
                if remaining_allocation > Decimal(str(current_objective.available_capital_usd)):
                    await _mark_remaining_reoptimization(index, "available_capital_changed")
                    return {
                        "_block_new_entries": True,
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code="target_attainment_recalculation_required",
                            message=(
                                "available capital changed before sequential "
                                "portfolio component submission"
                            ),
                        ),
                    }
            else:
                current_objective = None
                submission_fingerprint = None
            child_proposal = proposal.model_copy(
                update={
                    "proposal_id": uuid5(
                        NAMESPACE_URL,
                        f"{proposal.proposal_id}:portfolio-component:{index}",
                    ),
                    "strategy_id": (
                        UUID(str(position["strategy_id"]))
                        if position is not None
                        else proposal.strategy_id
                    ),
                    "snapshot_id": (proposal.snapshot_id),
                    "legs": (leg,),
                }
            )
            try:
                # Refresh account/order projection and market truth before every
                # sequential component. Validation never mutates the tuple.
                if deps.snapshot_repo is None:
                    raise RuntimeError("latest snapshot repository unavailable")
                latest_snapshot = await deps.snapshot_repo.get_latest(deps.session_id)
                if latest_snapshot is None:
                    raise RuntimeError("latest submission snapshot unavailable")
                child_proposal = child_proposal.model_copy(
                    update={"snapshot_id": latest_snapshot.snapshot_id}
                )
                snapshot, data_quality, surface, _slice = await load_snapshot_truth(
                    deps, str(latest_snapshot.snapshot_id)
                )
                exchange_now = None
                if deps.clock is not None and hasattr(deps.clock, "now"):
                    exchange_now = deps.clock.now()
                    if authorized_positions:
                        if snapshot.trading_date != deps.clock.trading_date():
                            raise RuntimeError("latest snapshot is not from current trading date")
                        if surface is None or surface.trading_date != deps.clock.trading_date():
                            raise RuntimeError("latest option surface is not current trading date")
                elif authorized_positions:
                    raise RuntimeError("exchange clock unavailable for submission")
                truth = build_truth_from_deps(
                    snapshot=snapshot,
                    data_quality=data_quality,
                    option_surface=surface,
                    projection=projection,
                    already_submitted_proposal_ids=tuple(deps.submitted_proposal_ids),
                    now=exchange_now,
                )
                provenanced = validate_and_compile_proposal(
                    child_proposal,
                    truth=truth,
                    evidence_ids=tuple(e.evidence_id for e in state.get("evidence") or []),
                    client_order_id=(
                        component_record.client_order_id if component_record is not None else None
                    ),
                )
                if component_record is not None:
                    component_record = await portfolio_execution_repo.transition(
                        tuple_id,
                        owner=portfolio_owner,
                        status=PortfolioComponentStatus.READY,
                        latest_validation_snapshot_id=str(child_proposal.snapshot_id),
                        submission_objective_version=(
                            int(current_objective.version)
                            if current_objective is not None
                            else None
                        ),
                        last_validation_timestamp=(
                            timing_evidence.submission_exchange_time.isoformat()
                            if timing_evidence is not None
                            else datetime.now(timezone.utc).isoformat()
                        ),
                        extra_update={
                            **(timing_evidence.as_dict() if timing_evidence is not None else {}),
                            "latest_validation_snapshot_id": str(child_proposal.snapshot_id),
                        },
                    )
                await publish_execution_observable_event(
                    deps,
                    state,
                    reoptimization_required=False,
                    payload={
                        "component_index": index,
                        "contract_id": leg.contract_id,
                        "quantity": leg.quantity,
                        "quote_revalidation": "passed",
                        "capital_revalidation": "passed",
                        "objective_version_revalidation": "passed",
                        "data_quality_revalidation": "passed",
                        **(timing_evidence.as_dict() if timing_evidence is not None else {}),
                    },
                )
            except Exception as exc:
                await _mark_remaining_reoptimization(index, str(exc))
                await publish_execution_observable_event(
                    deps,
                    state,
                    reoptimization_required=True,
                    payload={
                        "component_index": index,
                        "reason": str(exc),
                        "submitted_component_count": len(command_ids),
                    },
                )
                return {
                    "_block_new_entries": bool(command_ids),
                    "_execution_command_ids": command_ids,
                    **append_error(
                        state,
                        node_name="submit_execution_command",
                        error_code=(
                            "target_attainment_recalculation_required"
                            if command_ids
                            else "submit_validation_failed"
                        ),
                        message=str(exc),
                    ),
                }
            if gateway is not None:
                action = (
                    OrderActionKind.PROBE
                    if getattr(proposal, "action", None) == "probe"
                    else OrderActionKind.ENTRY
                )
                action_request = provenanced_to_action_request(
                    provenanced,
                    action=action,
                    causation_event_id=_resolve_entry_causation_event_id(state),
                )
                action_request = replace(
                    action_request,
                    session_id=(portfolio_owner.session_id if portfolio_owner else None),
                    run_id=deps.run_id,
                    broker_account_id=(
                        portfolio_owner.broker_account_id
                        if portfolio_owner
                        else action_request.broker_account_id
                    ),
                    trading_date=(portfolio_owner.trading_date if portfolio_owner else None),
                    target_portfolio_decision_id=(
                        str(leg.target_portfolio_decision_id)
                        if leg.target_portfolio_decision_id is not None
                        else None
                    ),
                    selected_portfolio_id=(
                        str(leg.selected_portfolio_id)
                        if leg.selected_portfolio_id is not None
                        else None
                    ),
                    authorized_position_tuple_id=(
                        str(leg.authorized_position_tuple_id)
                        if leg.authorized_position_tuple_id is not None
                        else None
                    ),
                    component_index=index,
                    component_count=len(proposal.legs),
                    evaluated_objective_version=leg.evaluated_objective_version,
                    submission_objective_version=(
                        int(current_objective.version) if current_objective is not None else None
                    ),
                    evaluated_objective_fingerprint=(leg.evaluated_objective_fingerprint),
                    submission_objective_fingerprint=(
                        submission_fingerprint.canonical_json
                        if submission_fingerprint is not None
                        else None
                    ),
                    original_decision_snapshot_id=(
                        str(leg.original_decision_snapshot_id)
                        if leg.original_decision_snapshot_id is not None
                        else None
                    ),
                    submission_snapshot_id=str(child_proposal.snapshot_id),
                    evaluated_at_exchange_time=(
                        timing_evidence.evaluated_at_exchange_time.isoformat()
                        if timing_evidence is not None
                        else None
                    ),
                    decision_valid_until_exchange_time=(
                        timing_evidence.decision_valid_until_exchange_time.isoformat()
                        if timing_evidence is not None
                        else None
                    ),
                    maximum_decision_age_seconds=(
                        timing_evidence.maximum_decision_age_seconds
                        if timing_evidence is not None
                        else None
                    ),
                    submission_exchange_time=(
                        timing_evidence.submission_exchange_time.isoformat()
                        if timing_evidence is not None
                        else None
                    ),
                    decision_age_seconds=(
                        str(timing_evidence.decision_age_seconds)
                        if timing_evidence is not None
                        else None
                    ),
                    required_resolution_horizon_seconds=(
                        timing_evidence.required_resolution_horizon_seconds
                        if timing_evidence is not None
                        else None
                    ),
                    evaluation_premium=(
                        str(leg.evaluation_premium) if leg.evaluation_premium is not None else None
                    ),
                    capital_allocation=(
                        str(leg.capital_allocation) if leg.capital_allocation is not None else None
                    ),
                )
                sizing = state.get("_sizing_decision") or {}
                estimate_id = sizing.get("estimate_id")
                if estimate_id:
                    action_request = replace(action_request, estimate_id=str(estimate_id))
                gateway_result = await gateway.submit(action_request)
                if not gateway_result.submitted:
                    if component_record is not None:
                        component_record = await portfolio_execution_repo.transition(
                            tuple_id,
                            owner=portfolio_owner,
                            status=PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
                            failure_reoptimization_reason=(
                                gateway_result.blocked_reason or "order_action_blocked"
                            ),
                        )
                    await _mark_remaining_reoptimization(
                        index + 1,
                        gateway_result.blocked_reason or "order_action_blocked",
                    )
                    await publish_execution_observable_event(
                        deps,
                        state,
                        reoptimization_required=True,
                        payload={
                            "component_index": index,
                            "reason": (gateway_result.blocked_reason or "order action blocked"),
                            "submitted_component_count": len(command_ids),
                        },
                    )
                    return {
                        "_block_new_entries": bool(command_ids),
                        "_execution_command_ids": command_ids,
                        **append_error(
                            state,
                            node_name="submit_execution_command",
                            error_code=(
                                "target_attainment_recalculation_required"
                                if command_ids
                                else "gateway_blocked"
                            ),
                            message=(gateway_result.blocked_reason or "order action blocked"),
                        ),
                    }
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
                            extra={
                                "session_id": (
                                    portfolio_owner.session_id
                                    if portfolio_owner
                                    else deps.session_id
                                ),
                                "run_id": deps.run_id,
                                "origin_run_id": (
                                    component_record.origin_run_id
                                    if component_record is not None
                                    else deps.run_id
                                ),
                                "last_resumed_run_id": (
                                    component_record.last_resumed_run_id
                                    if component_record is not None
                                    else None
                                ),
                                "broker_account_identity": (
                                    portfolio_owner.broker_account_identity
                                    if portfolio_owner
                                    else "default"
                                ),
                                "trading_date": (
                                    portfolio_owner.trading_date if portfolio_owner else None
                                ),
                                "target_portfolio_decision_id": (
                                    str(leg.target_portfolio_decision_id)
                                    if leg.target_portfolio_decision_id
                                    else None
                                ),
                                "selected_portfolio_id": (
                                    str(leg.selected_portfolio_id)
                                    if leg.selected_portfolio_id
                                    else None
                                ),
                                "authorized_position_tuple_id": tuple_id or None,
                                "component_index": index,
                                "component_count": len(proposal.legs),
                                "evaluated_objective_version": (leg.evaluated_objective_version),
                                "submission_objective_version": (
                                    int(current_objective.version)
                                    if current_objective is not None
                                    else None
                                ),
                                "evaluated_objective_fingerprint": (
                                    leg.evaluated_objective_fingerprint
                                ),
                                "submission_objective_fingerprint": (
                                    submission_fingerprint.canonical_json
                                    if submission_fingerprint is not None
                                    else None
                                ),
                                "original_decision_snapshot_id": (
                                    str(leg.original_decision_snapshot_id)
                                    if leg.original_decision_snapshot_id
                                    else None
                                ),
                                "submission_snapshot_id": str(child_proposal.snapshot_id),
                                **(
                                    timing_evidence.as_dict() if timing_evidence is not None else {}
                                ),
                            },
                        )
                    )
                result = await deps.submit_callback(provenanced)
            if component_record is not None:
                broker_status = str(getattr(result, "status", "submitted") or "submitted").lower()
                broker_status_map = {
                    "accepted": PortfolioComponentStatus.WORKING,
                    "open": PortfolioComponentStatus.WORKING,
                    "pending": PortfolioComponentStatus.WORKING,
                    "working": PortfolioComponentStatus.WORKING,
                    "partially_filled": PortfolioComponentStatus.PARTIALLY_FILLED,
                    "filled": PortfolioComponentStatus.FILLED,
                    "rejected": PortfolioComponentStatus.REJECTED,
                    "cancelled": PortfolioComponentStatus.CANCELLED,
                }
                durable_status = broker_status_map.get(
                    broker_status, PortfolioComponentStatus.SUBMITTED
                )
                filled_quantity = int(
                    getattr(result, "filled_quantity", 0)
                    or (leg.quantity if durable_status == PortfolioComponentStatus.FILLED else 0)
                )
                result_broker_order_id = str(getattr(result, "order_id", "") or "") or None
                if durable_status == PortfolioComponentStatus.FILLED:
                    component_record = await _persist_filled_continuation(
                        component_record,
                        filled_quantity=filled_quantity,
                        broker_order_id=result_broker_order_id,
                    )
                    expected_fingerprint = ObjectiveDecisionFingerprint.from_json(
                        component_record.post_fill_objective_fingerprint
                    )
                else:
                    component_record = await portfolio_execution_repo.transition(
                        tuple_id,
                        owner=portfolio_owner,
                        status=durable_status,
                        broker_order_id=result_broker_order_id,
                        submitted_quantity=leg.quantity,
                        filled_quantity=filled_quantity,
                        last_reconciliation_timestamp=(
                            deps.clock.now().isoformat()
                            if deps.clock is not None
                            else datetime.now(timezone.utc).isoformat()
                        ),
                    )
                if component_record.status in {
                    PortfolioComponentStatus.REJECTED,
                    PortfolioComponentStatus.CANCELLED,
                }:
                    await _mark_remaining_reoptimization(
                        index + 1,
                        "prior_component_" + component_record.status.value.lower(),
                    )
            deps.submitted_proposal_ids.add(str(child_proposal.proposal_id))
            command_ids.append(command_id)
            result_refs.append(str(getattr(result, "order_id", command_id)))
            if (
                position is not None
                and deps.objective_service is not None
                and deps.clock is not None
            ):
                await deps.objective_service.recompute_from_truth(now=deps.clock.now())
                post_objective = await deps.objective_service.get_state()
                post_projection = (
                    await deps.projection_loader()
                    if deps.projection_loader is not None
                    else projection
                )
                post_fingerprint = _current_fingerprint(
                    post_objective,
                    working_order_count=len(working_orders_from_projection(post_projection)),
                )
                if (
                    component_record is None
                    or component_record.status != PortfolioComponentStatus.FILLED
                ):
                    expected_fingerprint = post_fingerprint
                if deps.provenance_registry is not None:
                    recorded = await deps.provenance_registry.get_by_client_order_id(command_id)
                    if recorded is not None:
                        await deps.provenance_registry.record(
                            replace(
                                recorded,
                                extra={
                                    **(recorded.extra or {}),
                                    "post_submission_objective_version": int(
                                        post_objective.version
                                    ),
                                    "post_submission_objective_fingerprint": (
                                        post_fingerprint.canonical_json
                                    ),
                                },
                            )
                        )
            if component_record is not None and component_record.status in {
                PortfolioComponentStatus.SUBMITTED,
                PortfolioComponentStatus.WORKING,
                PortfolioComponentStatus.PARTIALLY_FILLED,
                PortfolioComponentStatus.REJECTED,
                PortfolioComponentStatus.CANCELLED,
            }:
                break
        execution_state = (
            [
                record.as_dict()
                for record in await portfolio_execution_repo.list_by_decision(
                    str(portfolio_decision.get("decision_id") or ""),
                    owner=portfolio_owner,
                )
            ]
            if portfolio_execution_repo is not None and portfolio_decision.get("decision_id")
            else []
        )
        return {
            "execution_command_id": command_ids[0] if command_ids else None,
            "_execution_command_ids": command_ids,
            "execution_result_ref": ",".join(result_refs),
            "_target_portfolio_decision": portfolio_decision,
            "_portfolio_execution_state": execution_state,
            **trace_update(
                append_trace(state, node_name="submit_execution_command", status="completed")
            ),
        }

    async def persist_cycle(state: CognitiveGraphState) -> dict[str, Any]:
        await _publish_cycle_completed(deps, state, outcome="completed")
        return trace_update(append_trace(state, node_name="persist_cycle", status="completed"))

    async def persist_pending_cycle(state: CognitiveGraphState) -> dict[str, Any]:
        await _publish_cycle_completed(deps, state, outcome="delayed")
        return trace_update(
            append_trace(state, node_name="persist_pending_cycle", status="completed")
        )

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
    graph.add_node("score_strategies_against_objective", score_strategies_against_objective)
    graph.add_node("select_debate_candidates", select_debate_candidates)
    graph.add_node("debate", debate)
    graph.add_node("finalize_portfolio_review", finalize_portfolio_review)
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
    graph.add_edge("debate", "finalize_portfolio_review")
    graph.add_edge("finalize_portfolio_review", "meta_decision")
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
        {
            "debate": "debate",
            "meta_decision": "finalize_portfolio_review",
        },
    )

    _SIZING_STOP_CODES = frozenset(
        {
            "validation_failed",
            "sizing_rejected",
            "entries_blocked",
            "missing_proposal",
            "estimate_invalid",
            "target_attainment_recalculation_required",
            "target_attainment_contract_mismatch",
            "target_attainment_strategy_mismatch",
            "target_attainment_tuple_incomplete",
            "target_attainment_missing_legs",
        }
    )

    def after_validate_route(state: CognitiveGraphState) -> str:
        if state.get("stale_decision"):
            return "persist_stale"
        errors = state.get("errors") or []
        if any(e.error_code in _SIZING_STOP_CODES for e in errors):
            return "persist_cycle"
        return "submit_execution_command"

    def after_apply_sizing(state: CognitiveGraphState) -> str:
        errors = state.get("errors") or []
        if any(e.error_code in _SIZING_STOP_CODES for e in errors):
            return "persist_cycle"
        if state.get("_block_new_entries"):
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
    from joker.graph.observable_events import publish_cycle_observable_events

    await publish_cycle_observable_events(deps, state, outcome=outcome)


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

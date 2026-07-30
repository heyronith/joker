"""Cognitive graph nodes for goal feasibility, scoring, and sizing."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from joker.cognition.schemas import MetaDecisionAction
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_error, append_trace, trace_update
from joker.objectives.feasibility import FeasibilityInputs
from joker.objectives.scoring import StrategyScoreInput
from joker.objectives.schemas import state_to_context

_HARD_BLOCK_CODES = frozenset(
    {
        "objective_unconfirmed",
        "objective_unavailable",
        "deadline_reached",
        "entries_paused",
        "objective_infeasible",
    }
)


async def load_objective_context(deps: CognitiveGraphDeps) -> dict[str, Any] | None:
    if deps.objective_state_loader is None:
        return None
    state = await deps.objective_state_loader()
    return state_to_context(state).model_dump_for_hash()


def entry_blocked_by_objective(state: CognitiveGraphState) -> bool:
    """True when new ENTRY/PROBE must not proceed."""
    if state.get("_block_new_entries"):
        return True
    errors = state.get("errors") or []
    return any(getattr(e, "error_code", None) in _HARD_BLOCK_CODES for e in errors)


async def gate_objective_confirmed(
    deps: CognitiveGraphDeps, state: CognitiveGraphState
) -> dict[str, Any] | None:
    """Return an error update if objective is required but not entry-ready."""
    if deps.objective_service is None:
        return None
    try:
        obj_state = await deps.objective_service.get_state()
    except Exception as exc:
        return {
            "_block_new_entries": True,
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="objective_unavailable",
                message=str(exc),
                recoverable=False,
            ),
        }
    if obj_state.status == "pending_confirmation":
        return {
            "_block_new_entries": True,
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="objective_unconfirmed",
                message="entry graph requires a confirmed objective",
                recoverable=False,
            ),
        }
    if obj_state.status == "deadline_reached" or obj_state.time_remaining_seconds <= 0:
        return {
            "_block_new_entries": True,
            "_meta_decision_override": "abandon",
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="deadline_reached",
                message="deadline reached — no new entries",
            ),
        }
    if obj_state.entries_paused or obj_state.status == "target_reached":
        return {
            "_block_new_entries": True,
            "_meta_decision_override": "abandon",
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="entries_paused",
                message="entries paused (target reached or paused)",
            ),
        }
    if obj_state.feasibility_classification == "infeasible":
        return {
            "_block_new_entries": True,
            "_meta_decision_override": "abandon",
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="objective_infeasible",
                message="feasibility infeasible — no new entries",
            ),
        }
    return None


async def assess_goal_feasibility_node(
    deps: CognitiveGraphDeps, state: CognitiveGraphState
) -> dict[str, Any]:
    if deps.feasibility_engine is None or deps.objective_service is None:
        return trace_update(
            append_trace(state, node_name="assess_goal_feasibility", status="skipped")
        )
    obj_state = await deps.objective_service.get_state()
    snapshot_id = UUID(str(state.get("snapshot_id")))
    assessment = deps.feasibility_engine.assess(
        obj_state,
        FeasibilityInputs(snapshot_id=snapshot_id),
    )
    if deps.objective_service._repo is not None:  # noqa: SLF001
        deps.objective_service._repo.save_feasibility(assessment)
    await deps.objective_service.update_feasibility(
        classification=assessment.classification,
        estimated_success_probability=assessment.estimated_success_probability,
    )
    block = assessment.classification == "infeasible"
    update: dict[str, Any] = {
        "_feasibility_assessment": assessment.model_dump(mode="json"),
        **trace_update(
            append_trace(state, node_name="assess_goal_feasibility", status="completed")
        ),
    }
    if block:
        update["_block_new_entries"] = True
        update["_meta_decision_override"] = "abandon"
        update.update(
            append_error(
                state,
                node_name="assess_goal_feasibility",
                error_code="objective_infeasible",
                message="feasibility infeasible — no new entries",
            )
        )
    return update


async def score_strategies_against_objective_node(
    deps: CognitiveGraphDeps, state: CognitiveGraphState
) -> dict[str, Any]:
    if deps.objective_strategy_scorer is None or deps.objective_service is None:
        return trace_update(
            append_trace(
                state, node_name="score_strategies_against_objective", status="skipped"
            )
        )
    obj_state = await deps.objective_service.get_state()
    snapshot_id = UUID(str(state.get("snapshot_id")))
    p_before = obj_state.estimated_success_probability
    candidates: list[StrategyScoreInput] = []
    for strategy in state.get("strategies") or []:
        ev = getattr(strategy, "expected_value_usd", None)
        win_p = getattr(strategy, "estimated_win_probability", None)
        extras = getattr(strategy, "model_extra", None) or {}
        if ev is None and isinstance(extras, dict):
            ev = extras.get("expected_value_usd")
        if win_p is None and isinstance(extras, dict):
            win_p = extras.get("estimated_win_probability")
        capital = obj_state.available_capital_usd
        max_loss = obj_state.available_capital_usd
        # Prefer explicit capital/max-loss when present on the strategy payload.
        for attr, target in (
            ("capital_required_usd", "capital"),
            ("maximum_loss_usd", "max_loss"),
        ):
            raw = getattr(strategy, attr, None)
            if raw is None and isinstance(extras, dict):
                raw = extras.get(attr)
            if raw is not None:
                if target == "capital":
                    capital = Decimal(str(raw))
                else:
                    max_loss = Decimal(str(raw))
        candidates.append(
            StrategyScoreInput(
                strategy_id=strategy.strategy_id,
                snapshot_id=snapshot_id,
                expected_value_usd=Decimal(str(ev)) if ev is not None else None,
                estimated_win_probability=(
                    Decimal(str(win_p)) if win_p is not None else None
                ),
                maximum_loss_usd=max_loss,
                capital_required_usd=capital,
                calculation_inputs={"name": getattr(strategy, "name", None)},
            )
        )
    scores = deps.objective_strategy_scorer.score_all(
        obj_state,
        candidates,
        snapshot_id=snapshot_id,
        target_probability_before=p_before,
    )
    for score in scores:
        deps.objective_service._repo.save_strategy_score(score)  # noqa: SLF001
    valid_trade = [s for s in scores if s.valid and not s.is_no_trade]
    return {
        "_strategy_scores": [s.model_dump(mode="json") for s in scores],
        "_no_valid_strategy": len(valid_trade) == 0,
        **trace_update(
            append_trace(
                state,
                node_name="score_strategies_against_objective",
                status="completed",
            )
        ),
    }


async def deterministic_sizing_node(
    deps: CognitiveGraphDeps, state: CognitiveGraphState
) -> dict[str, Any]:
    if deps.capital_sizer is None or deps.objective_service is None:
        return trace_update(
            append_trace(state, node_name="deterministic_sizing", status="skipped")
        )
    meta = state.get("meta_decision")
    if meta is None or meta.action not in {
        MetaDecisionAction.EXECUTE,
        MetaDecisionAction.PROBE,
    }:
        return trace_update(
            append_trace(state, node_name="deterministic_sizing", status="skipped")
        )
    if entry_blocked_by_objective(state):
        return {
            "_sizing_decision": {"approved": False, "reason_codes": ["entries_blocked"]},
            **append_error(
                state,
                node_name="deterministic_sizing",
                error_code="entries_blocked",
                message="objective gate blocks sizing",
            ),
        }
    proposal = state.get("execution_proposal")
    obj_state = await deps.objective_service.get_state()
    if obj_state.feasibility_classification == "infeasible":
        return {
            "_sizing_decision": {"approved": False, "reason_codes": ["infeasible"]},
            "_block_new_entries": True,
            **append_error(
                state,
                node_name="deterministic_sizing",
                error_code="infeasible",
                message="infeasible objective blocks sizing",
            ),
        }
    requested = None
    premium = Decimal("0.10")
    if proposal is not None and getattr(proposal, "legs", None):
        leg = proposal.legs[0]
        requested = int(getattr(leg, "quantity", 1) or 1)
        px = getattr(leg, "limit_price", None)
        if px is not None:
            premium = Decimal(str(px))
    is_probe = meta.action == MetaDecisionAction.PROBE
    decision = deps.capital_sizer.size(
        obj_state,
        strategy_id=meta.selected_strategy_id,
        premium_per_contract_usd=premium,
        requested_quantity=requested,
        is_probe=is_probe,
    )
    update: dict[str, Any] = {
        "_sizing_decision": decision.model_dump(mode="json"),
        **trace_update(
            append_trace(state, node_name="deterministic_sizing", status="completed")
        ),
    }
    if not decision.approved:
        update.update(
            append_error(
                state,
                node_name="deterministic_sizing",
                error_code="sizing_rejected",
                message="deterministic sizer rejected quantity",
            )
        )
    return update


async def apply_objective_sizing_to_proposal(
    deps: CognitiveGraphDeps, state: CognitiveGraphState
) -> dict[str, Any]:
    """Re-size with proposal premium and clamp legs; fail closed if rejected."""
    proposal = state.get("execution_proposal")
    if proposal is None:
        return append_error(
            state,
            node_name="apply_objective_sizing",
            error_code="missing_proposal",
            message="no execution proposal to size",
        )
    if deps.capital_sizer is None or deps.objective_service is None:
        return trace_update(
            append_trace(state, node_name="apply_objective_sizing", status="skipped")
        )
    if entry_blocked_by_objective(state):
        return {
            "_block_new_entries": True,
            **append_error(
                state,
                node_name="apply_objective_sizing",
                error_code="entries_blocked",
                message="objective gate blocks sized entry",
            ),
        }
    meta = state.get("meta_decision")
    obj_state = await deps.objective_service.get_state()
    leg = proposal.legs[0]
    premium = Decimal(str(leg.limit_price or "0.10"))
    requested = int(leg.quantity)
    is_probe = bool(
        meta is not None and meta.action == MetaDecisionAction.PROBE
    ) or getattr(proposal, "action", None) == "probe"
    decision = deps.capital_sizer.size(
        obj_state,
        strategy_id=getattr(meta, "selected_strategy_id", None)
        or getattr(proposal, "strategy_id", None),
        premium_per_contract_usd=premium,
        requested_quantity=requested,
        is_probe=is_probe,
    )
    if not decision.approved or decision.approved_quantity <= 0:
        return {
            "_sizing_decision": decision.model_dump(mode="json"),
            **append_error(
                state,
                node_name="apply_objective_sizing",
                error_code="sizing_rejected",
                message="deterministic sizer rejected proposal quantity",
            ),
        }
    qty = int(decision.approved_quantity)
    new_legs = tuple(
        leg.model_copy(update={"quantity": qty}) for leg in proposal.legs
    )
    sized = proposal.model_copy(update={"legs": new_legs})
    return {
        "execution_proposal": sized,
        "_sizing_decision": decision.model_dump(mode="json"),
        **trace_update(
            append_trace(state, node_name="apply_objective_sizing", status="completed")
        ),
    }

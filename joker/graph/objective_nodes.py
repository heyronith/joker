"""Cognitive graph nodes for goal feasibility, scoring, and sizing."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

from joker.cognition.schemas import MetaDecisionAction
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.context_hydrate import load_snapshot_truth
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.node_helpers import append_error, append_trace, trace_update
from joker.objectives.estimate import StrategyEstimateBuilder
from joker.objectives.feasibility_inputs import build_feasibility_inputs_from_truth
from joker.objectives.scoring import StrategyScoreInput
from joker.objectives.schemas import state_to_context

_HARD_BLOCK_CODES = frozenset(
    {
        "objective_unconfirmed",
        "objective_unavailable",
        "deadline_reached",
        "entries_paused",
        "objective_infeasible",
        "truth_degraded",
        "insufficient_historical_evidence",
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
    if getattr(obj_state, "truth_degraded", False) or obj_state.status == "truth_degraded":
        return {
            "_block_new_entries": True,
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="truth_degraded",
                message="objective truth degraded — no new entries",
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
    if obj_state.status == "insufficient_historical_evidence":
        return {
            "_block_new_entries": True,
            "_meta_decision_override": "abandon",
            **append_error(
                state,
                node_name="validate_trigger",
                error_code="insufficient_historical_evidence",
                message="cold-start: insufficient factual historical EV samples",
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
    snapshot, data_quality, _surface, surface_slice = await load_snapshot_truth(
        deps, snapshot_id
    )
    projection = None
    if deps.projection_loader is not None:
        projection = await deps.projection_loader()
    evidence_ids = tuple(e.evidence_id for e in (state.get("evidence") or []))
    inputs = build_feasibility_inputs_from_truth(
        snapshot_id=snapshot_id,
        snapshot=snapshot,
        data_quality=data_quality,
        option_surface_slice=surface_slice,
        projection=projection,
        available_capital_usd=obj_state.available_capital_usd,
        evidence_ids=evidence_ids,
    )
    assessment = deps.feasibility_engine.assess(obj_state, inputs)
    deps.objective_service.save_feasibility(assessment)
    await deps.objective_service.update_feasibility(
        classification=assessment.classification,
        estimated_success_probability=assessment.estimated_success_probability,
    )
    block = assessment.classification == "infeasible"
    update: dict[str, Any] = {
        "_feasibility_assessment": assessment.model_dump(mode="json"),
        "_feasibility_inputs": {
            "session_phase": inputs.session_phase,
            "median_premium_usd": (
                str(inputs.median_premium_usd)
                if inputs.median_premium_usd is not None
                else None
            ),
            "typical_spread_pct": inputs.typical_spread_pct,
            "quote_age_seconds": inputs.quote_age_seconds,
            "valid_contract_count": inputs.valid_contract_count,
            "comparable_outcome_samples": inputs.comparable_outcome_samples,
        },
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
    snapshot, _dq, _surface, surface_slice = await load_snapshot_truth(deps, snapshot_id)
    # Prefer mid of first usable contract as default premium for estimates.
    default_premium: Decimal | None = None
    option_type: str | None = None
    for contract in surface_slice:
        bid = getattr(contract, "bid", None)
        ask = getattr(contract, "ask", None)
        if bid is None or ask is None:
            continue
        try:
            default_premium = (
                (Decimal(str(bid)) + Decimal(str(ask))) / Decimal("2")
            ).quantize(Decimal("0.01"))
            option_type = getattr(contract, "option_type", None)
            break
        except Exception:
            continue

    hist_settings = getattr(deps, "historical_outcome_settings", None)
    min_samples = 20
    require_lcb = True
    ttl = 300
    if hist_settings is not None:
        min_samples = int(hist_settings.minimum_samples_for_ev)
        require_lcb = bool(hist_settings.require_lower_confidence_bound_positive)
        ttl = int(hist_settings.estimate_ttl_seconds)

    builder = StrategyEstimateBuilder(
        minimum_samples_for_calibrated_ev=min_samples,
        require_positive_expected_value=bool(
            getattr(deps.objective_service, "require_positive_expected_value", True)
        ),
        require_lower_confidence_bound_positive=require_lcb,
        estimate_ttl_seconds=ttl,
    )
    as_of = datetime.now(timezone.utc)
    if snapshot is not None:
        as_of = getattr(snapshot, "exchange_time", None) or getattr(
            snapshot, "exchange_timestamp", None
        ) or as_of

    candidates: list[StrategyScoreInput] = []
    estimates: list[dict[str, Any]] = []
    historical_summaries: list[dict[str, Any]] = []
    max_sample_seen = 0
    for strategy in state.get("strategies") or []:
        summary = None
        if deps.historical_outcome_service is not None:
            direction = str(getattr(strategy.direction, "value", strategy.direction))
            summary = await deps.historical_outcome_service.summarize_for_strategy(
                objective_id=obj_state.objective_id,
                strategy_id=strategy.strategy_id,
                snapshot_id=snapshot_id,
                as_of_timestamp=as_of,
                direction=direction,
                strategy_family=direction,
                pattern_ids=tuple(
                    getattr(strategy, "source_hypothesis_ids", ()) or ()
                ),
                option_type=option_type,
                premium_per_contract_usd=default_premium,
                expected_horizon_seconds=int(strategy.expected_horizon_seconds),
                current_episode_id=None,
            )
            historical_summaries.append(summary.model_dump(mode="json"))
            max_sample_seen = max(max_sample_seen, int(summary.sample_count))
        estimate = builder.build(
            strategy=strategy,
            objective_state=obj_state,
            snapshot_id=snapshot_id,
            premium_per_contract_usd=default_premium,
            historical_summary=summary,
            evidence_ids=tuple(getattr(strategy, "supporting_evidence_ids", ()) or ()),
        )
        deps.objective_service.save_strategy_estimate(estimate)
        estimates.append(estimate.model_dump(mode="json"))
        candidates.append(
            StrategyScoreInput(
                strategy_id=strategy.strategy_id,
                snapshot_id=snapshot_id,
                expected_value_usd=estimate.expected_value_usd,
                estimated_win_probability=estimate.estimated_win_probability,
                estimated_payoff_ratio=estimate.estimated_payoff_ratio,
                estimated_resolution_seconds=estimate.estimated_resolution_seconds,
                maximum_loss_usd=estimate.maximum_loss_usd,
                capital_required_usd=estimate.capital_required_usd,
                evidence_ids=estimate.evidence_ids,
                assumptions=estimate.assumptions,
                calculation_inputs={
                    "estimate_id": str(estimate.estimate_id),
                    "calculation_method": estimate.calculation_method,
                    "uncertainty_reasons": list(estimate.uncertainty_reasons),
                    "historical_summary_id": (
                        str(estimate.historical_summary_id)
                        if estimate.historical_summary_id
                        else None
                    ),
                    "sample_count": estimate.sample_count,
                    "lower_confidence_bound_ev_usd": (
                        str(estimate.lower_confidence_bound_ev_usd)
                        if estimate.lower_confidence_bound_ev_usd is not None
                        else None
                    ),
                },
            )
        )
    scores = deps.objective_strategy_scorer.score_all(
        obj_state,
        candidates,
        snapshot_id=snapshot_id,
        target_probability_before=p_before,
    )
    # Attach estimate_id onto scores via calculation_inputs already present.
    for score, cand in zip(scores, candidates, strict=False):
        if score.is_no_trade:
            continue
        est_id = (cand.calculation_inputs or {}).get("estimate_id")
        if est_id:
            score.estimate_id = UUID(str(est_id))
            score.uncertainty_reasons = tuple(
                (cand.calculation_inputs or {}).get("uncertainty_reasons") or ()
            )
        deps.objective_service.save_strategy_score(score)
    for score in scores:
        if score.is_no_trade:
            deps.objective_service.save_strategy_score(score)
    valid_trade = [s for s in scores if s.valid and not s.is_no_trade]
    if (
        not valid_trade
        and deps.historical_outcome_service is not None
        and max_sample_seen < min_samples
    ):
        await deps.objective_service.mark_insufficient_historical_evidence(
            sample_count=max_sample_seen,
            minimum_required=min_samples,
        )
    return {
        "_strategy_scores": [s.model_dump(mode="json") for s in scores],
        "_strategy_estimates": estimates,
        "_historical_summaries": historical_summaries,
        "_no_valid_strategy": len(valid_trade) == 0,
        "_historical_sample_count": max_sample_seen,
        "_historical_minimum_required": min_samples,
        **trace_update(
            append_trace(
                state,
                node_name="score_strategies_against_objective",
                status="completed",
            )
        ),
    }


def _estimate_for_strategy(
    state: CognitiveGraphState, strategy_id: UUID | None
) -> dict[str, Any] | None:
    if strategy_id is None:
        return None
    for est in state.get("_strategy_estimates") or []:
        if str(est.get("strategy_id")) == str(strategy_id):
            return est
    return None


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
    estimate = _estimate_for_strategy(state, meta.selected_strategy_id)
    if estimate is None or not estimate.get("valid"):
        return {
            "_sizing_decision": {
                "approved": False,
                "reason_codes": ["estimate_missing_or_invalid"],
            },
            **append_error(
                state,
                node_name="deterministic_sizing",
                error_code="estimate_invalid",
                message="selected strategy lacks a valid objective estimate",
            ),
        }
    ev = estimate.get("expected_value_usd")
    win_p = estimate.get("estimated_win_probability")
    payoff = estimate.get("estimated_payoff_ratio")
    requested = None
    premium = Decimal(str(estimate.get("quote_inputs", {}).get("premium_per_contract") or "0.10"))
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
        expected_value_usd=ev,
        estimated_win_probability=win_p,
        expected_r=payoff,
        is_probe=is_probe,
    )
    dump = decision.model_dump(mode="json")
    dump["estimate_id"] = estimate.get("estimate_id")
    update: dict[str, Any] = {
        "_sizing_decision": dump,
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
    strategy_id = getattr(meta, "selected_strategy_id", None) or getattr(
        proposal, "strategy_id", None
    )
    estimate = _estimate_for_strategy(state, strategy_id)
    if estimate is None or not estimate.get("valid"):
        return append_error(
            state,
            node_name="apply_objective_sizing",
            error_code="estimate_invalid",
            message="cannot size without a valid positive-EV estimate",
        )
    leg = proposal.legs[0]
    premium = Decimal(str(leg.limit_price or "0.10"))
    requested = int(leg.quantity)
    is_probe = bool(
        meta is not None and meta.action == MetaDecisionAction.PROBE
    ) or getattr(proposal, "action", None) == "probe"
    decision = deps.capital_sizer.size(
        obj_state,
        strategy_id=strategy_id,
        premium_per_contract_usd=premium,
        requested_quantity=requested,
        expected_value_usd=estimate.get("expected_value_usd"),
        estimated_win_probability=estimate.get("estimated_win_probability"),
        expected_r=estimate.get("estimated_payoff_ratio"),
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
    new_legs = tuple(leg.model_copy(update={"quantity": qty}) for leg in proposal.legs)
    sized = proposal.model_copy(update={"legs": new_legs})
    dump = decision.model_dump(mode="json")
    dump["estimate_id"] = estimate.get("estimate_id")
    return {
        "execution_proposal": sized,
        "_sizing_decision": dump,
        **trace_update(
            append_trace(state, node_name="apply_objective_sizing", status="completed")
        ),
    }

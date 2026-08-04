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
        require_lower_confidence_bound_positive=(
            require_lcb
            if getattr(deps, "objective_policy", "positive_ev_baseline")
            != "target_attainment"
            else False
        ),
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
    # Regime / liquidity context from world-model assessments when present.
    regime_labels: list[str] = []
    volatility_bucket: str | None = None
    liquidity_bucket: str | None = None
    similarity_bucket = "unknown"
    wm = state.get("world_model")
    if wm is not None:
        ms = getattr(wm, "market_structure", None)
        if ms is not None:
            regime_labels.append(str(getattr(ms, "primary_direction", "")))
            if getattr(ms, "range_bound", False):
                regime_labels.append("range_bound")
            else:
                regime_labels.append("trend")
        vol = getattr(wm, "volatility_state", None)
        if vol is not None:
            volatility_bucket = str(getattr(vol, "state", None) or "") or None
        opt = getattr(wm, "options_state", None)
        if opt is not None:
            spread = str(getattr(opt, "spread_conditions", "") or "").lower()
            if "tight" in spread:
                liquidity_bucket = "tight"
            elif "wide" in spread:
                liquidity_bucket = "wide"
            else:
                liquidity_bucket = "normal"
        temporal = getattr(wm, "temporal_state", None)
        if temporal is not None:
            similarity_bucket = str(
                getattr(temporal, "session_phase", "unknown") or "unknown"
            )
    # Historical similarity bucket may be open/midday/close; never use it for
    # physical eligibility. Fall back to clock-derived similarity for history.
    if similarity_bucket in {"", "unknown"} and deps.clock is not None:
        from joker.objectives.historical_outcomes import session_phase_from_exchange_ts

        similarity_bucket = session_phase_from_exchange_ts(deps.clock.now())
    session_phase = similarity_bucket  # historical summarizer still expects bucket labels
    from joker.objectives.session_eligibility import resolve_objective_session_state

    objective_session = resolve_objective_session_state(
        clock=deps.clock,
        similarity_bucket=similarity_bucket,
    )

    current_episode_id = state.get("current_episode_id")
    if current_episode_id is not None:
        current_episode_id = UUID(str(current_episode_id))

    # Resolve active configuration → training/challenger dataset IDs for leakage.
    configuration_version_id = None
    blocked_training_dataset_ids: tuple[UUID, ...] = ()
    challenger_dataset_ids: tuple[UUID, ...] = ()
    configuration_dataset_provenance_resolved = True
    evo = getattr(deps, "evolution_runtime", None)
    cfg_repo = getattr(deps, "configuration_repo", None)
    if evo is not None:
        try:
            champ_id = await evo.current_champion_id()
        except Exception:
            champ_id = None
        if champ_id is not None:
            configuration_version_id = champ_id
            cfg = None
            if cfg_repo is not None:
                cfg = await cfg_repo.get_by_id(champ_id)
            elif hasattr(evo, "repositories"):
                cfg_repo_rt = evo.repositories.get("configurations")
                if cfg_repo_rt is not None:
                    cfg = await cfg_repo_rt.get_by_id(champ_id)
            if cfg is None:
                configuration_dataset_provenance_resolved = False
            else:
                provenance = str(
                    getattr(cfg, "dataset_provenance_status", "unknown") or "unknown"
                )
                if provenance == "unknown":
                    configuration_dataset_provenance_resolved = False
                    blocked_training_dataset_ids = ()
                    challenger_dataset_ids = ()
                elif provenance == "not_applicable":
                    configuration_dataset_provenance_resolved = True
                    blocked_training_dataset_ids = ()
                    challenger_dataset_ids = ()
                else:
                    # resolved — only configuration-linked datasets are blocked
                    blocked_training_dataset_ids = tuple(
                        getattr(cfg, "training_dataset_ids", ()) or ()
                    )
                    challenger_dataset_ids = tuple(
                        getattr(cfg, "challenger_dataset_ids", ()) or ()
                    )
                    configuration_dataset_provenance_resolved = True

    for strategy in state.get("strategies") or []:
        summary = None
        if deps.historical_outcome_service is not None:
            direction = str(getattr(strategy.direction, "value", strategy.direction))
            # Never invent strategy_family from agent role.
            family = getattr(strategy, "strategy_family", None)
            if family:
                summary = await deps.historical_outcome_service.summarize_for_strategy(
                    objective_id=obj_state.objective_id,
                    strategy_id=strategy.strategy_id,
                    snapshot_id=snapshot_id,
                    as_of_timestamp=as_of,
                    direction=direction,
                    strategy_family=str(family),
                    pattern_ids=tuple(
                        getattr(strategy, "source_hypothesis_ids", ()) or ()
                    ),
                    regime_labels=tuple(r for r in regime_labels if r),
                    session_phase=session_phase,
                    option_type=option_type,
                    volatility_bucket=volatility_bucket,
                    liquidity_bucket=liquidity_bucket,
                    premium_per_contract_usd=default_premium,
                    expected_horizon_seconds=int(strategy.expected_horizon_seconds),
                    current_episode_id=current_episode_id,
                    configuration_version_id=configuration_version_id,
                    blocked_training_dataset_ids=blocked_training_dataset_ids,
                    challenger_dataset_ids=challenger_dataset_ids,
                    configuration_dataset_provenance_resolved=(
                        configuration_dataset_provenance_resolved
                    ),
                )
                historical_summaries.append(summary.model_dump(mode="json"))
                max_sample_seen = max(max_sample_seen, int(summary.sample_count))
            else:
                # Explicit family required — never invent from agent role.
                historical_summaries.append(
                    {
                        "strategy_id": str(strategy.strategy_id),
                        "valid_for_ev": False,
                        "sample_count": 0,
                        "invalidation_reasons": ["historical_strategy_family_missing"],
                    }
                )
        estimate = builder.build(
            strategy=strategy,
            objective_state=obj_state,
            snapshot_id=snapshot_id,
            premium_per_contract_usd=default_premium,
            historical_summary=summary,
            evidence_ids=tuple(getattr(strategy, "supporting_evidence_ids", ()) or ()),
        )
        if not getattr(strategy, "strategy_family", None):
            estimate = estimate.model_copy(
                update={
                    "expected_value_usd": None,
                    "calculation_method": "ev_unavailable",
                    "uncertainty_reasons": tuple(
                        dict.fromkeys(
                            (
                                *estimate.uncertainty_reasons,
                                "historical_strategy_family_missing",
                            )
                        )
                    ),
                }
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

    policy = str(
        getattr(deps, "objective_policy", None)
        or getattr(deps.objective_service, "objective_policy", "positive_ev_baseline")
    )
    shadow_enabled = bool(
        getattr(deps, "shadow_baseline_enabled", False)
        or getattr(deps.objective_service, "shadow_baseline_enabled", False)
    )
    ta_decision_dump: dict[str, Any] | None = None
    ta_no_valid = False
    if policy == "target_attainment" and deps.target_attainment_policy is not None:
        from joker.objectives.contract_candidates import (
            build_contract_candidates_for_strategies,
        )
        from joker.objectives.target_attainment import (
            TargetAttainmentAction,
            TargetAttainmentContext,
            candidate_from_score_input,
            run_positive_ev_baseline_shadow,
        )

        baseline_shadow = None
        if shadow_enabled:
            baseline_shadow = run_positive_ev_baseline_shadow(
                obj_state,
                candidates,
                snapshot_id=snapshot_id,
                require_positive_expected_value=True,
                minimum_win_probability=0.45,
            )
        ta_settings = getattr(deps, "target_attainment_settings", None)
        allow_full = True
        max_frac = 1.0
        min_cal = 20
        if ta_settings is not None:
            allow_full = bool(getattr(ta_settings, "allow_full_remaining_capital", True))
            max_frac = float(getattr(ta_settings, "maximum_capital_fraction", 1.0))
            min_cal = int(getattr(ta_settings, "minimum_calibrated_samples", 20))
        max_contracts = int(
            getattr(deps.capital_sizer, "maximum_authorised_contracts", 20)
            if deps.capital_sizer is not None
            else 20
        )
        # Durable original duration — never substitute remaining time.
        duration_s = obj_state.objective_duration_seconds
        exchange_phase = (
            objective_session.exchange_phase.value
            if objective_session.exchange_phase is not None
            else None
        )
        dq_codes: list[str] = []
        if _dq is not None:
            for finding in getattr(_dq, "findings", ()) or ():
                code = getattr(finding, "code", None) or getattr(finding, "finding_code", None)
                if code:
                    dq_codes.append(str(code))
            for code in getattr(_dq, "codes", ()) or ():
                dq_codes.append(str(code))
        market_usable = True
        if _dq is not None and hasattr(_dq, "usable_for_execution"):
            market_usable = bool(getattr(_dq, "usable_for_execution"))
        ctx = TargetAttainmentContext.from_state(
            obj_state,
            snapshot_id=snapshot_id,
            objective_duration_seconds=duration_s,
            maximum_authorised_contracts=max_contracts,
            allow_full_remaining_capital=allow_full,
            maximum_capital_fraction=max_frac,
            minimum_calibrated_samples=min_cal,
            exchange_session_phase=exchange_phase,
            session_similarity_bucket=similarity_bucket,
            session_phase=exchange_phase or "unknown",
            market_usable_for_execution=market_usable,
            option_surface_usable=bool(surface_slice),
            data_quality_codes=tuple(dq_codes),
        )
        estimates_by_strategy = {
            str(e.get("strategy_id")): e for e in estimates if e.get("strategy_id")
        }
        trading_date = None
        if deps.clock is not None:
            trading_date = deps.clock.trading_date()
        max_quote_age = getattr(deps, "max_quote_age_seconds", 30)
        contract_cands = build_contract_candidates_for_strategies(
            strategies=list(state.get("strategies") or []),
            surface_slice=list(surface_slice or []),
            trading_date=trading_date,
            estimates_by_strategy=estimates_by_strategy,
            max_quote_age_seconds=(
                float(max_quote_age) if max_quote_age is not None else None
            ),
            now=deps.clock.now() if deps.clock is not None else as_of,
        )
        if contract_cands:
            ta_cands = [c.as_candidate() for c in contract_cands]
        else:
            # Fail closed to score-input bridge only when strategies lack legs;
            # still attach strategy-specific premiums when present.
            ta_cands = []
            for c in candidates:
                prem = default_premium
                ta_cands.append(
                    candidate_from_score_input(c, premium_per_contract_usd=prem)
                )
        from dataclasses import replace as dc_replace

        for i, summary in enumerate(historical_summaries):
            if i >= len(ta_cands):
                break
            rate = summary.get("hit_rate") or summary.get("win_rate")
            if rate is None:
                continue
            c0 = ta_cands[i]
            ta_cands[i] = dc_replace(
                c0,
                historical_hit_rate=Decimal(str(rate)),
                sample_count=int(summary.get("sample_count") or c0.sample_count),
            )
        decision = deps.target_attainment_policy.decide(
            ctx,
            ta_cands,
            baseline_shadow=baseline_shadow,
            session_state=objective_session,
        )
        ta_decision_dump = decision.as_dict()
        if decision.action == TargetAttainmentAction.ENTER:
            ta_no_valid = False
            for idx, score in enumerate(scores):
                if score.is_no_trade:
                    continue
                if (
                    decision.selected_strategy_id is not None
                    and score.strategy_id == decision.selected_strategy_id
                ):
                    scores[idx] = score.model_copy(
                        update={
                            "valid": True,
                            "invalidation_codes": tuple(
                                c
                                for c in (score.invalidation_codes or ())
                                if c
                                not in {
                                    "non_positive_expected_value",
                                    "expected_value_unavailable",
                                    "win_probability_below_minimum",
                                }
                            ),
                        }
                    )
        else:
            ta_no_valid = True

    valid_trade = [s for s in scores if s.valid and not s.is_no_trade]
    if policy == "target_attainment":
        no_valid = ta_no_valid or (
            ta_decision_dump is not None
            and ta_decision_dump.get("action") != "enter"
        )
    else:
        no_valid = len(valid_trade) == 0
        if (
            not valid_trade
            and deps.historical_outcome_service is not None
            and max_sample_seen < min_samples
        ):
            await deps.objective_service.mark_insufficient_historical_evidence(
                sample_count=max_sample_seen,
                minimum_required=min_samples,
            )
    result: dict[str, Any] = {
        "_strategy_scores": [s.model_dump(mode="json") for s in scores],
        "_strategy_estimates": estimates,
        "_historical_summaries": historical_summaries,
        "_no_valid_strategy": no_valid,
        "_historical_sample_count": max_sample_seen,
        "_historical_minimum_required": min_samples,
        "_objective_policy": policy,
        "_objective_session": objective_session.as_dict(),
        **trace_update(
            append_trace(
                state,
                node_name="score_strategies_against_objective",
                status="completed",
            )
        ),
    }
    if ta_decision_dump is not None:
        result["_target_attainment_decision"] = ta_decision_dump
        result["_target_attainment_action"] = ta_decision_dump.get("action")
        result["_target_attainment_quantity"] = int(
            ta_decision_dump.get("selected_quantity") or 0
        )
        result["_target_attainment_strategy_id"] = ta_decision_dump.get(
            "selected_strategy_id"
        )
        result["_target_attainment_contract_id"] = ta_decision_dump.get(
            "selected_contract_id"
        )
        result["_target_attainment_objective_version"] = ta_decision_dump.get(
            "objective_version"
        )
        result["_target_attainment_snapshot_id"] = ta_decision_dump.get("snapshot_id")
        result["_target_attainment_authoritative"] = True
    else:
        # Explicitly clear so a reused graph state cannot leak prior authority.
        result["_target_attainment_authoritative"] = False
        result["_target_attainment_action"] = None
        result["_target_attainment_decision"] = None
        result["_target_attainment_strategy_id"] = None
        result["_target_attainment_contract_id"] = None
        result["_target_attainment_quantity"] = None
    return result


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
    target_mode = (
        str(getattr(deps, "objective_policy", "positive_ev_baseline"))
        == "target_attainment"
    )
    if estimate is None or (not estimate.get("valid") and not target_mode):
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
    if estimate is None:
        return {
            "_sizing_decision": {
                "approved": False,
                "reason_codes": ["estimate_missing_or_invalid"],
            },
            **append_error(
                state,
                node_name="deterministic_sizing",
                error_code="estimate_invalid",
                message="selected strategy lacks an objective estimate",
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
    ta_qty = state.get("_target_attainment_quantity")
    if (
        target_mode
        and bool(state.get("_target_attainment_authoritative"))
        and ta_qty
        and int(ta_qty) > 0
    ):
        requested = int(ta_qty)
    is_probe = meta.action == MetaDecisionAction.PROBE
    decision = deps.capital_sizer.size(
        obj_state,
        strategy_id=meta.selected_strategy_id,
        premium_per_contract_usd=premium,
        requested_quantity=requested,
        expected_value_usd=None if target_mode else ev,
        estimated_win_probability=None if target_mode else win_p,
        expected_r=payoff if not target_mode else None,
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
    target_mode = (
        str(getattr(deps, "objective_policy", "positive_ev_baseline"))
        == "target_attainment"
    )
    if estimate is None or (not estimate.get("valid") and not target_mode):
        return append_error(
            state,
            node_name="apply_objective_sizing",
            error_code="estimate_invalid",
            message="cannot size without a valid objective estimate",
        )
    if estimate is None:
        return append_error(
            state,
            node_name="apply_objective_sizing",
            error_code="estimate_invalid",
            message="cannot size without an objective estimate",
        )
    leg = proposal.legs[0]
    premium = Decimal(str(leg.limit_price or "0.10"))
    requested = int(leg.quantity)
    ta_qty = state.get("_target_attainment_quantity")
    ta_cid = state.get("_target_attainment_contract_id")
    if (
        target_mode
        and bool(state.get("_target_attainment_authoritative"))
        and ta_qty
        and int(ta_qty) > 0
    ):
        requested = int(ta_qty)
    if (
        target_mode
        and bool(state.get("_target_attainment_authoritative"))
        and ta_cid
        and str(leg.contract_id) != str(ta_cid)
    ):
        return append_error(
            state,
            node_name="apply_objective_sizing",
            error_code="target_attainment_contract_mismatch",
            message="proposal contract differs from authoritative target-attainment tuple",
        )
    is_probe = bool(
        meta is not None and meta.action == MetaDecisionAction.PROBE
    ) or getattr(proposal, "action", None) == "probe"
    decision = deps.capital_sizer.size(
        obj_state,
        strategy_id=strategy_id,
        premium_per_contract_usd=premium,
        requested_quantity=requested,
        expected_value_usd=(
            None if target_mode else estimate.get("expected_value_usd")
        ),
        estimated_win_probability=(
            None if target_mode else estimate.get("estimated_win_probability")
        ),
        expected_r=(
            None if target_mode else estimate.get("estimated_payoff_ratio")
        ),
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

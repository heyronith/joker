"""Structured, redacted graph evidence events for terminal and JSONL consumers."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from joker.cli.graph_view import sanitize_graph_evidence
from joker.events.schemas import EventType, make_event
from joker.graph.graph_deps import CognitiveGraphDeps


async def publish_optimizer_scoring_events(
    deps: CognitiveGraphDeps,
    state: Mapping[str, Any],
) -> None:
    if deps.event_bus is None:
        return
    correlation = _correlation_id(state)
    timestamp = _now(deps)
    common = _cycle_panels(state)
    for strategy in state.get("strategies") or []:
        await _publish(
            deps,
            EventType.STRATEGY_THESIS_GENERATED,
            {
                "cycle_id": state.get("cycle_id"),
                "strategy_id": str(strategy.strategy_id),
                "agent_role": str(
                    getattr(getattr(strategy, "agent_role", None), "value", "")
                ),
                "strategy_name": strategy.name,
                "strategy_family": strategy.strategy_family,
                "direction": str(
                    getattr(getattr(strategy, "direction", None), "value", "")
                ),
                "confidence": strategy.confidence,
                "thesis_summary": strategy.market_thesis,
                "expected_horizon_seconds": strategy.expected_horizon_seconds,
                "key_evidence": [
                    str(x) for x in strategy.supporting_evidence_ids[:5]
                ],
            },
            correlation=correlation,
            timestamp=timestamp,
        )
    universe = state.get("_full_chain_universe")
    if isinstance(universe, dict):
        await _publish(
            deps,
            EventType.CHAIN_UNIVERSE_BUILT,
            {**universe, **common},
            correlation=correlation,
            timestamp=timestamp,
        )
    outcomes = list(state.get("_contract_outcomes") or [])
    if outcomes:
        await _publish(
            deps,
            EventType.CONTRACT_OUTCOME_ESTIMATED,
            {
                "cycle_id": state.get("cycle_id"),
                "estimate_count": len(outcomes),
                "estimate_types": sorted(
                    {str(row.get("estimate_type")) for row in outcomes}
                ),
                "unknown_count": sum(
                    1 for row in outcomes if not row.get("usable_for_ranking")
                ),
            },
            correlation=correlation,
            timestamp=timestamp,
        )
    contracts = _rank_contract_rows(state)
    for row in contracts:
        row["provisional_selected"] = bool(row.get("selected"))
        row["selected"] = False
    if contracts or universe is not None:
        review_limit = int(
            getattr(
                deps.full_chain_optimizer_settings,
                "top_candidates_for_agent_review",
                10,
            )
            if deps.full_chain_optimizer_settings is not None
            else 10
        )
        await _publish(
            deps,
            EventType.CONTRACT_GRID_SCORED,
            {
                **common,
                "row_count": len(state.get("_quantity_grid") or []),
                "best_probability_goal": (
                    contracts[0].get("probability_goal") if contracts else None
                ),
                "wait_probability_goal": (
                    contracts[0].get("probability_wait") if contracts else None
                ),
                "contracts": contracts[:review_limit],
            },
            correlation=correlation,
            timestamp=timestamp,
        )
    portfolios = _rank_portfolios(state)
    for row in portfolios:
        row["provisional_selected"] = bool(row.get("selected"))
        row["selected"] = False
    if portfolios or state.get("_target_portfolio_decision") is not None:
        portfolio_limit = int(
            getattr(
                deps.full_chain_optimizer_settings,
                "cli_top_portfolio_rows",
                10,
            )
            if deps.full_chain_optimizer_settings is not None
            else 10
        )
        await _publish(
            deps,
            EventType.PORTFOLIO_GRID_SCORED,
            {
                **common,
                "row_count": len(portfolios),
                "best_probability_goal": (
                    portfolios[0].get("probability_goal") if portfolios else None
                ),
                "portfolios": portfolios[:portfolio_limit],
            },
            correlation=correlation,
            timestamp=timestamp,
        )


async def publish_cycle_observable_events(
    deps: CognitiveGraphDeps,
    state: Mapping[str, Any],
    *,
    outcome: str,
) -> None:
    if deps.event_bus is None:
        return
    correlation = _correlation_id(state)
    timestamp = _now(deps)
    reviews: list[dict[str, Any]] = []
    for review in state.get("reviews") or []:
        payload = {
            "cycle_id": state.get("cycle_id"),
            "reviewer_role": str(
                getattr(getattr(review, "reviewer_role", None), "value", "")
            ),
            "reviewed_id": str(review.strategy_id),
            "verdict": str(getattr(review.verdict, "value", review.verdict)),
            "confidence": review.confidence,
            "claims_summary": list(review.claims),
            "failure_modes": list(review.identified_failure_modes),
            "required_revisions": list(review.required_revisions),
            "evidence_ids": [
                str(x)
                for x in (
                    *review.supporting_evidence_ids,
                    *review.contradicting_evidence_ids,
                )
            ][:10],
        }
        reviews.append(payload)
        await _publish(
            deps,
            EventType.DEBATE_REVIEW_COMPLETED,
            payload,
            correlation=correlation,
            timestamp=timestamp,
        )
    portfolio_decision = state.get("_target_portfolio_decision")
    if isinstance(portfolio_decision, dict):
        action = str(portfolio_decision.get("action") or "wait")
        payload = {
            **_cycle_panels(state),
            "reviews": reviews,
            "decision": {
                "action": action,
                "authorized_positions": portfolio_decision.get(
                    "authorized_positions"
                )
                or [],
                "selected_probability_goal": portfolio_decision.get(
                    "selected_probability_goal"
                ),
                "wait_probability_goal": portfolio_decision.get(
                    "wait_probability_goal"
                ),
                "probability_delta": portfolio_decision.get("probability_delta"),
                "reason_codes": portfolio_decision.get("reason_codes") or [],
                "objective_version": portfolio_decision.get("objective_version"),
                "snapshot_id": portfolio_decision.get("snapshot_id"),
                "selected_portfolio_id": portfolio_decision.get(
                    "selected_portfolio_id"
                ),
                "selected_strategy_id": portfolio_decision.get(
                    "selected_strategy_id"
                ),
                "selected_contract_id": portfolio_decision.get(
                    "selected_contract_id"
                ),
                "selected_quantity": portfolio_decision.get(
                    "selected_quantity", 0
                ),
                "selected_capital": portfolio_decision.get(
                    "selected_capital", "0"
                ),
                "evaluated_at_exchange_time": portfolio_decision.get(
                    "evaluated_at_exchange_time"
                ),
                "decision_valid_until_exchange_time": portfolio_decision.get(
                    "decision_valid_until_exchange_time"
                ),
                "maximum_decision_age_seconds": portfolio_decision.get(
                    "maximum_decision_age_seconds"
                ),
                "submission_exchange_time": portfolio_decision.get(
                    "submission_exchange_time"
                ),
                "decision_age_seconds": portfolio_decision.get(
                    "decision_age_seconds"
                ),
            },
            **{
                key: portfolio_decision.get(key)
                for key in (
                    "action",
                    "selected_probability_goal",
                    "wait_probability_goal",
                    "probability_delta",
                    "reason_codes",
                    "objective_version",
                    "snapshot_id",
                )
            },
        }
        event_type = (
            EventType.TARGET_PORTFOLIO_SELECTED
            if action == "enter"
            else EventType.TARGET_WAIT_SELECTED
        )
        await _publish(
            deps,
            event_type,
            payload,
            correlation=correlation,
            timestamp=timestamp,
        )
    await _publish(
        deps,
        EventType.GRAPH_CYCLE_COMPLETED,
        {
            "cycle_id": state.get("cycle_id"),
            "outcome": outcome,
            "decision_action": (
                portfolio_decision.get("action")
                if isinstance(portfolio_decision, dict)
                else state.get("_target_attainment_action")
            ),
            "execution_command_ids": list(
                state.get("_execution_command_ids") or []
            ),
            "error_codes": [
                getattr(error, "error_code", None)
                for error in state.get("errors") or []
            ],
        },
        correlation=correlation,
        timestamp=timestamp,
    )


async def publish_graph_cycle_started(
    deps: CognitiveGraphDeps,
    state: Mapping[str, Any],
) -> None:
    if deps.event_bus is None:
        return
    await _publish(
        deps,
        EventType.GRAPH_CYCLE_STARTED,
        {
            "cycle_id": state.get("cycle_id"),
            "snapshot_id": state.get("snapshot_id"),
            "objective_version": (
                (state.get("_objective_context") or {}).get("version")
                if isinstance(state.get("_objective_context"), dict)
                else None
            ),
        },
        correlation=_correlation_id(state),
        timestamp=_now(deps),
    )


async def publish_execution_observable_event(
    deps: CognitiveGraphDeps,
    state: Mapping[str, Any],
    *,
    reoptimization_required: bool,
    payload: dict[str, Any],
) -> None:
    """Publish safe execution revalidation/reoptimization evidence."""
    if deps.event_bus is None:
        return
    await _publish(
        deps,
        (
            EventType.EXECUTION_REOPTIMIZATION_REQUIRED
            if reoptimization_required
            else EventType.EXECUTION_REVALIDATION
        ),
        {
            "cycle_id": state.get("cycle_id"),
            "snapshot_id": state.get("snapshot_id"),
            **payload,
        },
        correlation=_correlation_id(state),
        timestamp=_now(deps),
    )


def _cycle_panels(state: Mapping[str, Any]) -> dict[str, Any]:
    objective = state.get("_objective_context") or {}
    if not isinstance(objective, dict):
        objective = {}
    universe = state.get("_full_chain_universe") or {}
    if not isinstance(universe, dict):
        universe = {}
    world_model = state.get("world_model")
    direction = None
    volatility = None
    if world_model is not None:
        structure = getattr(world_model, "market_structure", None)
        direction = getattr(structure, "primary_direction", None)
        volatility_state = getattr(world_model, "volatility_state", None)
        volatility = getattr(volatility_state, "state", None)
    return {
        "goal": {
            "authorized_capital": _pick(
                objective, "authorised_capital_usd", "authorized_capital_usd"
            ),
            "available_capital": objective.get("available_capital_usd"),
            "realized_pnl": _pick(
                objective, "realised_pnl_usd", "realized_pnl_usd"
            ),
            "remaining_goal_gap": _pick(
                objective,
                "required_profit_remaining_usd",
                "remaining_goal_gap_usd",
            ),
            "target": objective.get("target_profit_usd"),
            "deadline": objective.get("deadline_exchange_time"),
            "time_remaining_seconds": objective.get("time_remaining_seconds"),
            "maximum_positions": objective.get("max_concurrent_positions"),
        },
        "market": {
            "spy_price": universe.get("underlying_price"),
            "market_direction": _enum_value(direction),
            "volatility_regime": _enum_value(volatility),
            "session_phase": (
                (state.get("_objective_session") or {}).get("exchange_phase")
                if isinstance(state.get("_objective_session"), dict)
                else None
            ),
            "option_surface_size": universe.get("source_contract_count"),
            "eligible_contract_count": universe.get("eligible_contract_count"),
            "data_quality_state": (
                getattr(state.get("_data_quality"), "severity", None)
                if state.get("_data_quality") is not None
                else "unknown"
            ),
        },
    }


def _rank_contract_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    outcomes = {
        (str(row.get("strategy_id")), str(row.get("contract_id"))): row
        for row in state.get("_contract_outcomes") or []
    }
    strategies = {
        str(strategy.strategy_id): strategy.name
        for strategy in state.get("strategies") or []
    }
    rows: list[dict[str, Any]] = []
    for row in state.get("_quantity_grid") or []:
        merged = dict(row)
        outcome = outcomes.get(
            (str(row.get("strategy_id")), str(row.get("contract_id"))), {}
        )
        for key in (
            "bid",
            "ask",
            "delta",
            "distance_from_spot",
            "relative_spread",
        ):
            merged[key] = outcome.get(key)
        merged["strategy"] = strategies.get(
            str(row.get("strategy_id")), str(row.get("strategy_id"))
        )
        rows.append(merged)
    rows.sort(
        key=lambda row: (
            -_decimal(row.get("probability_goal"), default="-1"),
            -_decimal(row.get("lower_probability_bound"), default="-1"),
            _decimal(row.get("capital_required"), default="999999999"),
            str(row.get("strategy_id")),
            str(row.get("contract_id")),
            int(row.get("quantity") or 0),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _rank_portfolios(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in state.get("_portfolio_grid") or []]
    rows.sort(
        key=lambda row: (
            -_decimal(row.get("probability_goal"), default="-1"),
            -_decimal(row.get("lower_probability_bound"), default="-1"),
            _decimal(row.get("capital_deployed"), default="999999999"),
            tuple(row.get("component_contract_ids") or []),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


async def _publish(
    deps: CognitiveGraphDeps,
    event_type: EventType,
    payload: dict[str, Any],
    *,
    correlation: UUID,
    timestamp: datetime,
) -> None:
    assert deps.event_bus is not None
    await deps.event_bus.publish(
        make_event(
            event_type,
            session_id=deps.session_id,
            source="cognitive_graph_observability",
            exchange_timestamp=timestamp,
            correlation_id=correlation,
            payload=sanitize_graph_evidence(payload),
        )
    )


def _correlation_id(state: Mapping[str, Any]) -> UUID:
    return uuid5(
        NAMESPACE_URL,
        f"joker-graph-cycle:{state.get('session_id')}:{state.get('cycle_id')}",
    )


def _now(deps: CognitiveGraphDeps) -> datetime:
    if deps.clock is not None:
        return deps.clock.now()
    return datetime.now(timezone.utc)


def _pick(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if payload.get(key) is not None:
            return payload[key]
    return None


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _decimal(value: Any, *, default: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)

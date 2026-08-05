"""Goal-conditioned full-chain optimizer orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from joker.market.option_surface import OptionSurfaceSnapshot
from joker.objectives.config import FullChainOptimizerSettings
from joker.objectives.contract_outcomes import (
    ContractOutcomeEstimate,
    estimate_contract_outcome,
)
from joker.objectives.full_chain_universe import (
    ContractSelectionSpec,
    FullChainUniverse,
    FullChainUniverseSettings,
    build_full_chain_universe,
    contracts_for_spec,
    stratify_contracts,
)
from joker.objectives.portfolio_search import (
    PortfolioSearchSettings,
    TargetPortfolioDecision,
    expand_quantity_grid,
    search_target_portfolios,
)
from joker.objectives.shared_scenarios import (
    SharedUnderlyingScenarioGrid,
    build_shared_underlying_scenario_grid,
)
from joker.objectives.target_attainment import (
    TargetAttainmentContext,
    estimate_target_hit_probability,
)


@dataclass(frozen=True)
class FullChainOptimizationResult:
    universe: FullChainUniverse
    selection_specs: tuple[ContractSelectionSpec, ...]
    shared_scenario_grid: SharedUnderlyingScenarioGrid | None
    contract_outcomes: tuple[ContractOutcomeEstimate, ...]
    decision: TargetPortfolioDecision

    def state_payload(self) -> dict[str, Any]:
        return {
            "_full_chain_universe": self.universe.as_dict(),
            "_contract_selection_specs": [
                {
                    "option_types": list(spec.option_types),
                    "direction": spec.direction,
                    "strategy_family": spec.strategy_family,
                    "resolution_horizon_seconds": spec.resolution_horizon_seconds,
                    "minimum_moneyness_pct": str(spec.minimum_moneyness_pct),
                    "maximum_moneyness_pct": str(spec.maximum_moneyness_pct),
                    "preferred_delta_min": (
                        str(spec.preferred_delta_min)
                        if spec.preferred_delta_min is not None
                        else None
                    ),
                    "preferred_delta_max": (
                        str(spec.preferred_delta_max)
                        if spec.preferred_delta_max is not None
                        else None
                    ),
                    "maximum_relative_spread": str(spec.maximum_relative_spread),
                    "maximum_quote_age_seconds": spec.maximum_quote_age_seconds,
                }
                for spec in self.selection_specs
            ],
            "_contract_outcomes": [
                outcome.as_dict(include_scenarios=False)
                for outcome in self.contract_outcomes
            ],
            "_shared_underlying_scenario_grid": (
                self.shared_scenario_grid.as_dict()
                if self.shared_scenario_grid is not None
                else None
            ),
            "_quantity_grid": [
                row.as_dict() for row in self.decision.quantity_grid
            ],
            "_portfolio_grid": [
                portfolio.as_dict()
                for portfolio in self.decision.portfolio_evaluations
            ],
            "_target_portfolio_decision": self.decision.as_dict(),
            "_target_authorized_positions": [
                position.as_dict()
                for position in self.decision.authorized_positions
            ],
        }


def optimize_full_chain(
    *,
    strategies: Sequence[Any],
    surface: OptionSurfaceSnapshot,
    ctx: TargetAttainmentContext,
    settings: FullChainOptimizerSettings,
    maximum_authorised_contracts: int,
    current_exchange_time: datetime,
    current_trading_date: date,
    evaluated_objective_fingerprint: str | None = None,
) -> FullChainOptimizationResult:
    """Run truth-bound discovery through authoritative WAIT/portfolio selection."""
    universe_settings = FullChainUniverseSettings(
        maximum_quote_age_seconds=settings.maximum_quote_age_seconds,
        maximum_surface_age_seconds=settings.maximum_surface_age_seconds,
        maximum_future_timestamp_seconds=(
            settings.maximum_future_timestamp_seconds
        ),
        maximum_relative_spread=Decimal(str(settings.maximum_relative_spread)),
        maximum_contracts_evaluated=settings.maximum_contracts_evaluated,
        moneyness_buckets=tuple(
            Decimal(str(x)) for x in settings.moneyness_buckets
        ),
        premium_buckets=tuple(Decimal(str(x)) for x in settings.premium_buckets),
        delta_buckets=tuple(Decimal(str(x)) for x in settings.delta_buckets),
    )
    universe = build_full_chain_universe(
        snapshot_id=ctx.snapshot_id,
        surface=surface,
        current_exchange_time=current_exchange_time,
        current_trading_date=current_trading_date,
        available_capital_usd=ctx.available_capital_usd,
        settings=universe_settings,
    )
    wait_estimate = estimate_target_hit_probability(
        ctx=ctx,
        win_p=None,
        useful_upside_usd=Decimal("0"),
        capital_required_usd=Decimal("0"),
        sample_count=0,
        historical_hit_rate=None,
        resolution_seconds=None,
        is_no_trade=True,
    )
    common_horizon = min(
        max(1, int(ctx.time_remaining_seconds)),
        max(
            (
                max(1, int(getattr(strategy, "expected_horizon_seconds", 600) or 600))
                for strategy in strategies
            ),
            default=max(1, int(ctx.time_remaining_seconds)),
        ),
    )
    shared_scenario_grid = (
        build_shared_underlying_scenario_grid(
            strategies=strategies,
            reference_underlying_price=universe.underlying_price,
            evaluation_time=current_exchange_time,
            horizon_seconds=common_horizon,
        )
        if universe.underlying_price > 0 and universe.contracts
        else None
    )

    specs: list[ContractSelectionSpec] = []
    outcomes: list[ContractOutcomeEstimate] = []
    for strategy in sorted(strategies, key=lambda item: str(item.strategy_id)):
        spec = ContractSelectionSpec.from_strategy(
            strategy,
            maximum_relative_spread=settings.maximum_relative_spread,
            maximum_quote_age_seconds=settings.maximum_quote_age_seconds,
            maximum_premium_usd=ctx.available_capital_usd,
        )
        specs.append(spec)
        compatible = stratify_contracts(
            contracts_for_spec(universe, spec),
            maximum=settings.top_contracts_per_strategy,
        )
        for contract in compatible:
            evidence_ids = tuple(
                getattr(strategy, "supporting_evidence_ids", ()) or ()
            )
            outcomes.append(
                estimate_contract_outcome(
                    strategy=strategy,
                    contract=contract,
                    remaining_goal_gap_usd=ctx.remaining_goal_gap_usd,
                    time_remaining_seconds=ctx.time_remaining_seconds,
                    evidence_ids=evidence_ids,
                    shared_scenario_grid=shared_scenario_grid,
                )
            )

    quantity_grid = expand_quantity_grid(
        outcomes=outcomes,
        available_capital_usd=ctx.available_capital_usd,
        remaining_goal_gap_usd=ctx.remaining_goal_gap_usd,
        wait_probability_goal=wait_estimate.p_goal,
        maximum_authorised_contracts=maximum_authorised_contracts,
    )
    search_settings = PortfolioSearchSettings(
        enabled=settings.portfolio_search_enabled,
        beam_width=settings.portfolio_beam_width,
        maximum_portfolio_candidates=settings.maximum_portfolio_candidates,
        allow_duplicate_contracts=settings.allow_duplicate_contracts,
        minimum_probability_improvement_over_wait=Decimal(
            str(settings.minimum_probability_improvement_over_wait)
        ),
    )
    decision = search_target_portfolios(
        quantity_grid=quantity_grid,
        snapshot_id=ctx.snapshot_id,
        objective_version=ctx.objective_version,
        time_remaining_seconds=ctx.time_remaining_seconds,
        remaining_goal_gap_usd=ctx.remaining_goal_gap_usd,
        available_capital_usd=ctx.available_capital_usd,
        open_position_count=ctx.open_position_count,
        working_order_count=ctx.working_order_count,
        max_concurrent_positions=(
            ctx.max_concurrent_positions
            if settings.portfolio_search_enabled
            else min(ctx.max_concurrent_positions, 1)
        ),
        wait_probability_goal=wait_estimate.p_goal,
        evaluated_objective_fingerprint=evaluated_objective_fingerprint,
        evaluated_at_exchange_time=current_exchange_time,
        deadline_exchange_time=ctx.deadline_exchange_time,
        maximum_decision_age_seconds=settings.maximum_decision_age_seconds,
        settings=search_settings,
    )
    return FullChainOptimizationResult(
        universe=universe,
        selection_specs=tuple(specs),
        shared_scenario_grid=shared_scenario_grid,
        contract_outcomes=tuple(outcomes),
        decision=decision,
    )


def portfolio_decision_as_legacy_target_dict(
    decision: TargetPortfolioDecision,
) -> dict[str, Any]:
    """Compatibility view for the existing single-tuple graph authority channels."""
    first = decision.authorized_positions[0] if decision.authorized_positions else None
    selected_p = (
        {
            "p_goal": str(decision.selected_probability_goal),
            "estimate_type": "ordinal",
            "sample_count": 0,
            "lower_bound": None,
            "upper_bound": None,
            "uncertainty_reasons": [
                "scenario_based_contract_portfolio_estimate_not_calibrated"
            ],
            "assumptions": ["shared_underlying_scenario_evaluation"],
        }
        if decision.selected_probability_goal is not None
        else None
    )
    wait_p = (
        {
            "p_goal": str(decision.wait_probability_goal),
            "estimate_type": "ordinal",
            "sample_count": 0,
            "lower_bound": None,
            "upper_bound": None,
            "uncertainty_reasons": ["no_trade_opportunity_cost_ordinal"],
            "assumptions": ["wait_value_decays_with_urgency_and_gap"],
        }
        if decision.wait_probability_goal is not None
        else None
    )
    return {
        "decision_id": str(decision.decision_id),
        "action": decision.action.value,
        "feasibility": (
            "attainable"
            if decision.action.value == "enter"
            else "low_probability"
        ),
        "selected_strategy_id": str(first.strategy_id) if first else None,
        "selected_contract_id": first.contract_id if first else None,
        "selected_quantity": first.quantity if first else 0,
        "selected_capital_usd": (
            str(sum((p.capital_allocation for p in decision.authorized_positions), Decimal("0")))
            if first
            else "0"
        ),
        "selected_evaluation_premium_usd": (
            str(first.evaluation_premium) if first else None
        ),
        "selected_p_goal": selected_p,
        "no_trade_p_goal": wait_p,
        "probability_delta": (
            str(decision.probability_delta)
            if decision.probability_delta is not None
            else None
        ),
        "snapshot_id": str(decision.snapshot_id),
        "objective_version": decision.objective_version,
        "authoritative": True,
        "reason_codes": list(decision.reason_codes),
        "quantity_evaluations": [
            row.as_dict() for row in decision.quantity_grid
        ],
        "portfolio_evaluations": [
            portfolio.as_dict()
            for portfolio in decision.portfolio_evaluations
        ],
        "authorized_positions": [
            position.as_dict() for position in decision.authorized_positions
        ],
        "time_remaining_seconds": decision.time_remaining_seconds,
        "evaluated_at_exchange_time": (
            decision.evaluated_at_exchange_time.isoformat()
            if decision.evaluated_at_exchange_time is not None
            else None
        ),
        "decision_valid_until_exchange_time": (
            decision.decision_valid_until_exchange_time.isoformat()
            if decision.decision_valid_until_exchange_time is not None
            else None
        ),
        "maximum_decision_age_seconds": decision.maximum_decision_age_seconds,
        "required_resolution_horizon_seconds": (
            decision.required_resolution_horizon_seconds
        ),
    }

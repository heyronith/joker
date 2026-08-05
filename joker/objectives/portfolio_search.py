"""Deterministic quantity expansion and bounded correlated portfolio search."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from itertools import combinations
from typing import Any, Sequence
from uuid import UUID, uuid4

from joker.objectives.contract_outcomes import ContractOutcomeEstimate
from joker.objectives.shared_scenarios import SharedUnderlyingScenario


class PortfolioAction(StrEnum):
    ENTER = "enter"
    WAIT = "wait"
    BLOCK = "block"


@dataclass(frozen=True)
class QuantityScenarioPnl:
    scenario_id: str
    shared_scenario_grid_hash: str
    probability: Decimal
    underlying_price: Decimal
    horizon_seconds: int
    pnl_usd: Decimal

    @property
    def identity(self) -> tuple[str, str, Decimal, Decimal, int]:
        return (
            self.shared_scenario_grid_hash,
            self.scenario_id,
            self.underlying_price,
            self.probability,
            self.horizon_seconds,
        )


@dataclass(frozen=True)
class QuantityGridRow:
    evaluation_id: UUID
    strategy_id: UUID
    strategy_family: str
    contract_id: str
    option_type: str
    strike: Decimal
    evaluation_premium: Decimal
    quantity: int
    capital_required: Decimal
    maximum_loss: Decimal
    useful_upside: Decimal
    expected_pnl: Decimal
    estimated_resolution_seconds: int
    probability_goal: Decimal | None
    probability_wait: Decimal | None
    probability_delta: Decimal | None
    lower_probability_bound: Decimal | None
    estimate_type: str
    reason_codes: tuple[str, ...]
    physically_feasible: bool
    selected: bool = False
    outcome_estimate_id: UUID | None = None
    scenario_pnl: tuple[QuantityScenarioPnl, ...] = ()
    relative_spread: Decimal = Decimal("1")
    liquidity_score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": str(self.evaluation_id),
            "strategy_id": str(self.strategy_id),
            "strategy_family": self.strategy_family,
            "contract_id": self.contract_id,
            "option_type": self.option_type,
            "strike": str(self.strike),
            "evaluation_premium": str(self.evaluation_premium),
            "quantity": self.quantity,
            "capital_required": str(self.capital_required),
            "maximum_loss": str(self.maximum_loss),
            "useful_upside": str(self.useful_upside),
            "expected_pnl": str(self.expected_pnl),
            "estimated_resolution_seconds": self.estimated_resolution_seconds,
            "probability_goal": (
                str(self.probability_goal)
                if self.probability_goal is not None
                else None
            ),
            "probability_wait": (
                str(self.probability_wait)
                if self.probability_wait is not None
                else None
            ),
            "probability_delta": (
                str(self.probability_delta)
                if self.probability_delta is not None
                else None
            ),
            "lower_probability_bound": (
                str(self.lower_probability_bound)
                if self.lower_probability_bound is not None
                else None
            ),
            "estimate_type": self.estimate_type,
            "relative_spread": str(self.relative_spread),
            "liquidity_score": self.liquidity_score,
            "reason_codes": list(self.reason_codes),
            "physically_feasible": self.physically_feasible,
            "selected": self.selected,
            "outcome_estimate_id": (
                str(self.outcome_estimate_id) if self.outcome_estimate_id else None
            ),
            "shared_scenario_grid_hash": (
                self.scenario_pnl[0].shared_scenario_grid_hash
                if self.scenario_pnl
                else None
            ),
        }


@dataclass(frozen=True)
class AuthorizedPositionTuple:
    position_tuple_id: UUID
    strategy_id: UUID
    contract_id: str
    quantity: int
    evaluation_premium: Decimal
    capital_allocation: Decimal
    maximum_loss: Decimal
    snapshot_id: UUID
    objective_version: int
    decision_id: UUID
    evaluated_objective_fingerprint: str | None = None
    submission_objective_version: int | None = None
    submission_objective_fingerprint: str | None = None
    submission_snapshot_id: UUID | None = None
    evaluated_at_exchange_time: datetime | None = None
    decision_valid_until_exchange_time: datetime | None = None
    maximum_decision_age_seconds: int | None = None
    required_resolution_horizon_seconds: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "position_tuple_id": str(self.position_tuple_id),
            "strategy_id": str(self.strategy_id),
            "contract_id": self.contract_id,
            "quantity": self.quantity,
            "evaluation_premium": str(self.evaluation_premium),
            "capital_allocation": str(self.capital_allocation),
            "maximum_loss": str(self.maximum_loss),
            "snapshot_id": str(self.snapshot_id),
            "objective_version": self.objective_version,
            "evaluated_objective_version": self.objective_version,
            "evaluated_objective_fingerprint": self.evaluated_objective_fingerprint,
            "submission_objective_version": self.submission_objective_version,
            "submission_objective_fingerprint": self.submission_objective_fingerprint,
            "original_decision_snapshot_id": str(self.snapshot_id),
            "submission_snapshot_id": (
                str(self.submission_snapshot_id)
                if self.submission_snapshot_id is not None
                else None
            ),
            "decision_id": str(self.decision_id),
            "evaluated_at_exchange_time": (
                self.evaluated_at_exchange_time.isoformat()
                if self.evaluated_at_exchange_time is not None
                else None
            ),
            "decision_valid_until_exchange_time": (
                self.decision_valid_until_exchange_time.isoformat()
                if self.decision_valid_until_exchange_time is not None
                else None
            ),
            "maximum_decision_age_seconds": self.maximum_decision_age_seconds,
            "required_resolution_horizon_seconds": (
                self.required_resolution_horizon_seconds
            ),
        }


@dataclass(frozen=True)
class PortfolioAttainmentEvaluation:
    portfolio_id: UUID
    component_evaluation_ids: tuple[UUID, ...]
    component_contract_ids: tuple[str, ...]
    component_quantities: tuple[int, ...]
    capital_deployed: Decimal
    maximum_loss: Decimal
    expected_pnl: Decimal
    probability_goal: Decimal | None
    probability_wait: Decimal | None
    probability_delta: Decimal | None
    lower_probability_bound: Decimal | None
    maximum_resolution_seconds: int
    concentration_penalty: Decimal
    liquidity_penalty: Decimal
    reason_codes: tuple[str, ...]
    physically_feasible: bool
    shared_scenario_grid_hash: str | None = None
    shared_scenarios: tuple[SharedUnderlyingScenario, ...] = ()
    selected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "portfolio_id": str(self.portfolio_id),
            "component_evaluation_ids": [
                str(x) for x in self.component_evaluation_ids
            ],
            "component_contract_ids": list(self.component_contract_ids),
            "component_quantities": list(self.component_quantities),
            "capital_deployed": str(self.capital_deployed),
            "maximum_loss": str(self.maximum_loss),
            "expected_pnl": str(self.expected_pnl),
            "probability_goal": (
                str(self.probability_goal)
                if self.probability_goal is not None
                else None
            ),
            "probability_wait": (
                str(self.probability_wait)
                if self.probability_wait is not None
                else None
            ),
            "probability_delta": (
                str(self.probability_delta)
                if self.probability_delta is not None
                else None
            ),
            "lower_probability_bound": (
                str(self.lower_probability_bound)
                if self.lower_probability_bound is not None
                else None
            ),
            "maximum_resolution_seconds": self.maximum_resolution_seconds,
            "concentration_penalty": str(self.concentration_penalty),
            "liquidity_penalty": str(self.liquidity_penalty),
            "reason_codes": list(self.reason_codes),
            "physically_feasible": self.physically_feasible,
            "shared_scenario_grid_hash": self.shared_scenario_grid_hash,
            "shared_scenarios": [
                scenario.as_dict() for scenario in self.shared_scenarios
            ],
            "selected": self.selected,
        }


@dataclass(frozen=True)
class TargetPortfolioDecision:
    decision_id: UUID
    action: PortfolioAction
    authorized_positions: tuple[AuthorizedPositionTuple, ...]
    selected_portfolio_id: UUID | None
    selected_probability_goal: Decimal | None
    wait_probability_goal: Decimal | None
    probability_delta: Decimal | None
    snapshot_id: UUID
    objective_version: int
    time_remaining_seconds: int
    reason_codes: tuple[str, ...]
    quantity_grid: tuple[QuantityGridRow, ...]
    portfolio_evaluations: tuple[PortfolioAttainmentEvaluation, ...]
    evaluated_objective_fingerprint: str | None = None
    submission_objective_version: int | None = None
    submission_objective_fingerprint: str | None = None
    submission_snapshot_id: UUID | None = None
    authoritative: bool = True
    evaluated_at_exchange_time: datetime | None = None
    decision_valid_until_exchange_time: datetime | None = None
    maximum_decision_age_seconds: int | None = None
    required_resolution_horizon_seconds: int | None = None
    submission_exchange_time: datetime | None = None
    decision_age_seconds: Decimal | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision_id": str(self.decision_id),
            "action": self.action.value,
            "authorized_positions": [p.as_dict() for p in self.authorized_positions],
            "selected_portfolio_id": (
                str(self.selected_portfolio_id)
                if self.selected_portfolio_id is not None
                else None
            ),
            "selected_probability_goal": (
                str(self.selected_probability_goal)
                if self.selected_probability_goal is not None
                else None
            ),
            "wait_probability_goal": (
                str(self.wait_probability_goal)
                if self.wait_probability_goal is not None
                else None
            ),
            "probability_delta": (
                str(self.probability_delta)
                if self.probability_delta is not None
                else None
            ),
            "snapshot_id": str(self.snapshot_id),
            "objective_version": self.objective_version,
            "evaluated_objective_version": self.objective_version,
            "evaluated_objective_fingerprint": self.evaluated_objective_fingerprint,
            "submission_objective_version": self.submission_objective_version,
            "submission_objective_fingerprint": self.submission_objective_fingerprint,
            "original_decision_snapshot_id": str(self.snapshot_id),
            "submission_snapshot_id": (
                str(self.submission_snapshot_id)
                if self.submission_snapshot_id is not None
                else None
            ),
            "time_remaining_seconds": self.time_remaining_seconds,
            "reason_codes": list(self.reason_codes),
            "quantity_grid": [row.as_dict() for row in self.quantity_grid],
            "portfolio_evaluations": [
                portfolio.as_dict() for portfolio in self.portfolio_evaluations
            ],
            "authoritative": self.authoritative,
            "evaluated_at_exchange_time": (
                self.evaluated_at_exchange_time.isoformat()
                if self.evaluated_at_exchange_time is not None
                else None
            ),
            "decision_valid_until_exchange_time": (
                self.decision_valid_until_exchange_time.isoformat()
                if self.decision_valid_until_exchange_time is not None
                else None
            ),
            "maximum_decision_age_seconds": self.maximum_decision_age_seconds,
            "required_resolution_horizon_seconds": (
                self.required_resolution_horizon_seconds
            ),
            "submission_exchange_time": (
                self.submission_exchange_time.isoformat()
                if self.submission_exchange_time is not None
                else None
            ),
            "decision_age_seconds": (
                str(self.decision_age_seconds)
                if self.decision_age_seconds is not None
                else None
            ),
        }


@dataclass(frozen=True)
class PortfolioSearchSettings:
    enabled: bool = True
    beam_width: int = 50
    maximum_portfolio_candidates: int = 500
    allow_duplicate_contracts: bool = False
    minimum_probability_improvement_over_wait: Decimal = Decimal("0.01")
    concentration_penalty_weight: Decimal = Decimal("0.02")
    liquidity_penalty_weight: Decimal = Decimal("0.02")


def expand_quantity_grid(
    *,
    outcomes: Sequence[ContractOutcomeEstimate],
    available_capital_usd: Decimal,
    remaining_goal_gap_usd: Decimal,
    wait_probability_goal: Decimal | None,
    maximum_authorised_contracts: int,
) -> tuple[QuantityGridRow, ...]:
    """Evaluate every affordable quantity for every defensible contract estimate."""
    rows: list[QuantityGridRow] = []
    for outcome in sorted(
        outcomes, key=lambda o: (str(o.strategy_id), o.contract_id)
    ):
        per_contract = outcome.maximum_loss_usd_per_contract
        if per_contract <= 0:
            continue
        max_q = min(
            int(maximum_authorised_contracts),
            int(available_capital_usd // per_contract),
        )
        if max_q <= 0:
            continue
        for quantity in range(1, max_q + 1):
            capital = (per_contract * Decimal(quantity)).quantize(Decimal("0.01"))
            scenario_pnl = tuple(
                QuantityScenarioPnl(
                    scenario_id=scenario.scenario_id,
                    shared_scenario_grid_hash=scenario.shared_scenario_grid_hash,
                    probability=scenario.probability,
                    underlying_price=scenario.underlying_price,
                    horizon_seconds=scenario.horizon_seconds,
                    pnl_usd=scenario.pnl_per_contract_usd * Decimal(quantity),
                )
                for scenario in outcome.scenarios
            )
            p_goal: Decimal | None = None
            lower: Decimal | None = None
            if outcome.usable_for_ranking and scenario_pnl:
                p_goal = sum(
                    (
                        scenario.probability
                        for scenario in scenario_pnl
                        if scenario.pnl_usd >= remaining_goal_gap_usd
                    ),
                    Decimal("0"),
                ).quantize(Decimal("0.0001"))
                uncertainty_width = (
                    (
                        outcome.probability_closes_goal_gap
                        - outcome.lower_probability_bound
                    )
                    if outcome.probability_closes_goal_gap is not None
                    and outcome.lower_probability_bound is not None
                    else Decimal("0.20")
                )
                lower = max(Decimal("0"), p_goal - uncertainty_width).quantize(
                    Decimal("0.0001")
                )
            delta = (
                (p_goal - wait_probability_goal).quantize(Decimal("0.0001"))
                if p_goal is not None and wait_probability_goal is not None
                else None
            )
            reasons: list[str] = []
            feasible = outcome.usable_for_ranking
            if not outcome.usable_for_ranking:
                reasons.append("contract_estimate_not_defensible")
            if outcome.estimated_resolution_seconds <= 0:
                feasible = False
                reasons.append("resolution_time_unavailable")
            rows.append(
                QuantityGridRow(
                    evaluation_id=uuid4(),
                    strategy_id=outcome.strategy_id,
                    strategy_family=outcome.strategy_family,
                    contract_id=outcome.contract_id,
                    option_type=outcome.option_type,
                    strike=outcome.strike,
                    evaluation_premium=outcome.evaluation_premium,
                    quantity=quantity,
                    capital_required=capital,
                    maximum_loss=capital,
                    useful_upside=(
                        outcome.estimated_useful_upside_usd * Decimal(quantity)
                    ).quantize(Decimal("0.01")),
                    expected_pnl=(outcome.expected_pnl_usd * Decimal(quantity)).quantize(
                        Decimal("0.01")
                    ),
                    estimated_resolution_seconds=outcome.estimated_resolution_seconds,
                    probability_goal=p_goal,
                    probability_wait=wait_probability_goal,
                    probability_delta=delta,
                    lower_probability_bound=lower,
                    estimate_type=outcome.estimate_type,
                    reason_codes=tuple(reasons),
                    physically_feasible=feasible,
                    outcome_estimate_id=outcome.estimate_id,
                    scenario_pnl=scenario_pnl,
                    relative_spread=outcome.relative_spread,
                    liquidity_score=max(
                        0.0, min(1.0, float(outcome.liquidity_score))
                    ),
                )
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                str(row.strategy_id),
                row.contract_id,
                row.quantity,
            ),
        )
    )


def search_target_portfolios(
    *,
    quantity_grid: Sequence[QuantityGridRow],
    snapshot_id: UUID,
    objective_version: int,
    time_remaining_seconds: int,
    remaining_goal_gap_usd: Decimal,
    available_capital_usd: Decimal,
    open_position_count: int,
    working_order_count: int,
    max_concurrent_positions: int,
    wait_probability_goal: Decimal | None,
    evaluated_objective_fingerprint: str | None = None,
    evaluated_at_exchange_time: datetime | None = None,
    deadline_exchange_time: datetime | None = None,
    maximum_decision_age_seconds: int | None = None,
    settings: PortfolioSearchSettings | None = None,
) -> TargetPortfolioDecision:
    """Bounded deterministic search using shared-scenario portfolio P&L."""
    cfg = settings or PortfolioSearchSettings()
    if (
        maximum_decision_age_seconds is not None
        and maximum_decision_age_seconds < 1
    ):
        raise ValueError("maximum_decision_age_seconds must be >= 1")
    if (
        evaluated_at_exchange_time is not None
        and evaluated_at_exchange_time.tzinfo is None
    ):
        raise ValueError("evaluated_at_exchange_time must be timezone-aware")
    if (
        deadline_exchange_time is not None
        and deadline_exchange_time.tzinfo is None
    ):
        raise ValueError("deadline_exchange_time must be timezone-aware")
    validity_deadline = None
    if (
        evaluated_at_exchange_time is not None
        and maximum_decision_age_seconds is not None
    ):
        validity_deadline = evaluated_at_exchange_time + timedelta(
            seconds=maximum_decision_age_seconds
        )
        if deadline_exchange_time is not None:
            validity_deadline = min(validity_deadline, deadline_exchange_time)
    decision_id = uuid4()
    slots = max(
        0,
        int(max_concurrent_positions)
        - int(open_position_count)
        - int(working_order_count),
    )
    feasible = [
        row
        for row in quantity_grid
        if row.physically_feasible
        and row.capital_required <= available_capital_usd
        and row.probability_goal is not None
    ]
    feasible.sort(key=_row_search_key)
    if slots <= 0:
        return _wait_decision(
            decision_id=decision_id,
            snapshot_id=snapshot_id,
            objective_version=objective_version,
            time_remaining_seconds=time_remaining_seconds,
            wait_probability_goal=wait_probability_goal,
            evaluated_objective_fingerprint=evaluated_objective_fingerprint,
            evaluated_at_exchange_time=evaluated_at_exchange_time,
            decision_valid_until_exchange_time=validity_deadline,
            maximum_decision_age_seconds=maximum_decision_age_seconds,
            quantity_grid=quantity_grid,
            portfolios=(),
            reason="maximum_concurrent_positions_reached",
        )
    if not feasible:
        return _wait_decision(
            decision_id=decision_id,
            snapshot_id=snapshot_id,
            objective_version=objective_version,
            time_remaining_seconds=time_remaining_seconds,
            wait_probability_goal=wait_probability_goal,
            evaluated_objective_fingerprint=evaluated_objective_fingerprint,
            evaluated_at_exchange_time=evaluated_at_exchange_time,
            decision_valid_until_exchange_time=validity_deadline,
            maximum_decision_age_seconds=maximum_decision_age_seconds,
            quantity_grid=quantity_grid,
            portfolios=(),
            reason="no_valid_contract_candidates",
        )

    # Keep the strongest quantity per contract in the combination frontier.
    # Every quantity remains auditable in quantity_grid.
    by_contract: dict[str, list[QuantityGridRow]] = {}
    for row in feasible:
        by_contract.setdefault(row.contract_id, []).append(row)
    frontier: list[QuantityGridRow] = []
    per_contract_limit = max(1, cfg.beam_width // max(1, len(by_contract)))
    for contract_id in sorted(by_contract):
        ranked = sorted(by_contract[contract_id], key=_row_rank_key, reverse=True)
        frontier.extend(ranked[:per_contract_limit])
    frontier = sorted(frontier, key=_row_rank_key, reverse=True)[
        : max(cfg.beam_width, slots)
    ]

    evaluations: list[PortfolioAttainmentEvaluation] = []
    maximum_components = min(slots, len(frontier))
    for component_count in range(1, maximum_components + 1):
        for combo in combinations(frontier, component_count):
            if len(evaluations) >= cfg.maximum_portfolio_candidates:
                break
            contract_ids = tuple(row.contract_id for row in combo)
            if not cfg.allow_duplicate_contracts and len(set(contract_ids)) != len(
                contract_ids
            ):
                continue
            capital = sum((row.capital_required for row in combo), Decimal("0"))
            if capital > available_capital_usd:
                continue
            evaluations.append(
                _evaluate_portfolio(
                    combo=combo,
                    remaining_goal_gap_usd=remaining_goal_gap_usd,
                    available_capital_usd=available_capital_usd,
                    wait_probability_goal=wait_probability_goal,
                    cfg=cfg,
                )
            )
        if len(evaluations) >= cfg.maximum_portfolio_candidates:
            break

    rankable = [
        portfolio
        for portfolio in evaluations
        if portfolio.physically_feasible and portfolio.probability_goal is not None
    ]
    if not rankable:
        return _wait_decision(
            decision_id=decision_id,
            snapshot_id=snapshot_id,
            objective_version=objective_version,
            time_remaining_seconds=time_remaining_seconds,
            wait_probability_goal=wait_probability_goal,
            evaluated_objective_fingerprint=evaluated_objective_fingerprint,
            evaluated_at_exchange_time=evaluated_at_exchange_time,
            decision_valid_until_exchange_time=validity_deadline,
            maximum_decision_age_seconds=maximum_decision_age_seconds,
            quantity_grid=quantity_grid,
            portfolios=evaluations,
            reason="no_rankable_portfolio",
        )
    best = max(rankable, key=_portfolio_rank_key)
    delta = (
        best.probability_goal - wait_probability_goal
        if best.probability_goal is not None and wait_probability_goal is not None
        else None
    )
    if (
        delta is None
        or delta < cfg.minimum_probability_improvement_over_wait
    ):
        return _wait_decision(
            decision_id=decision_id,
            snapshot_id=snapshot_id,
            objective_version=objective_version,
            time_remaining_seconds=time_remaining_seconds,
            wait_probability_goal=wait_probability_goal,
            evaluated_objective_fingerprint=evaluated_objective_fingerprint,
            evaluated_at_exchange_time=evaluated_at_exchange_time,
            decision_valid_until_exchange_time=validity_deadline,
            maximum_decision_age_seconds=maximum_decision_age_seconds,
            quantity_grid=quantity_grid,
            portfolios=evaluations,
            reason="wait_has_higher_or_insufficient_probability_improvement",
        )

    selected_eval_ids = set(best.component_evaluation_ids)
    selected_rows = [
        row for row in quantity_grid if row.evaluation_id in selected_eval_ids
    ]
    authorized = tuple(
        AuthorizedPositionTuple(
            position_tuple_id=uuid4(),
            strategy_id=row.strategy_id,
            contract_id=row.contract_id,
            quantity=row.quantity,
            evaluation_premium=row.evaluation_premium,
            capital_allocation=row.capital_required,
            maximum_loss=row.maximum_loss,
            snapshot_id=snapshot_id,
            objective_version=objective_version,
            decision_id=decision_id,
            evaluated_objective_fingerprint=evaluated_objective_fingerprint,
            evaluated_at_exchange_time=evaluated_at_exchange_time,
            decision_valid_until_exchange_time=validity_deadline,
            maximum_decision_age_seconds=maximum_decision_age_seconds,
            required_resolution_horizon_seconds=best.maximum_resolution_seconds,
        )
        for row in sorted(selected_rows, key=_row_search_key)
    )
    marked_rows = tuple(
        _replace_row_selected(row, row.evaluation_id in selected_eval_ids)
        for row in quantity_grid
    )
    marked_portfolios = tuple(
        _replace_portfolio_selected(p, p.portfolio_id == best.portfolio_id)
        for p in evaluations
    )
    return TargetPortfolioDecision(
        decision_id=decision_id,
        action=PortfolioAction.ENTER,
        authorized_positions=authorized,
        selected_portfolio_id=best.portfolio_id,
        selected_probability_goal=best.probability_goal,
        wait_probability_goal=wait_probability_goal,
        probability_delta=delta.quantize(Decimal("0.0001")),
        snapshot_id=snapshot_id,
        objective_version=objective_version,
        time_remaining_seconds=time_remaining_seconds,
        reason_codes=(
            "portfolio_improves_probability_over_wait",
            "shared_underlying_scenario_evaluation",
        ),
        quantity_grid=marked_rows,
        portfolio_evaluations=marked_portfolios,
        evaluated_objective_fingerprint=evaluated_objective_fingerprint,
        evaluated_at_exchange_time=evaluated_at_exchange_time,
        decision_valid_until_exchange_time=validity_deadline,
        maximum_decision_age_seconds=maximum_decision_age_seconds,
        required_resolution_horizon_seconds=best.maximum_resolution_seconds,
    )


def _evaluate_portfolio(
    *,
    combo: Sequence[QuantityGridRow],
    remaining_goal_gap_usd: Decimal,
    available_capital_usd: Decimal,
    wait_probability_goal: Decimal | None,
    cfg: PortfolioSearchSettings,
) -> PortfolioAttainmentEvaluation:
    reference_scenarios = combo[0].scenario_pnl if combo else ()
    reference_identity = tuple(
        scenario.identity for scenario in reference_scenarios
    )
    scenarios_compatible = bool(reference_scenarios) and all(
        tuple(scenario.identity for scenario in row.scenario_pnl)
        == reference_identity
        for row in combo
    )
    p_goal: Decimal | None = None
    lower: Decimal | None = None
    aggregate: list[tuple[Decimal, Decimal]] = []
    if scenarios_compatible:
        for index, scenario in enumerate(reference_scenarios):
            aggregate.append(
                (
                    scenario.probability,
                    sum(
                        (row.scenario_pnl[index].pnl_usd for row in combo),
                        Decimal("0"),
                    ),
                )
            )
        total_probability = sum((probability for probability, _ in aggregate), Decimal("0"))
        if total_probability == Decimal("1"):
            p_goal = sum(
                (
                    probability
                    for probability, pnl in aggregate
                    if pnl >= remaining_goal_gap_usd
                ),
                Decimal("0"),
            ).quantize(Decimal("0.0001"))
            component_lowers = [
                row.lower_probability_bound
                for row in combo
                if row.lower_probability_bound is not None
            ]
            lower = (
                min(p_goal, min(component_lowers))
                if component_lowers
                else max(Decimal("0"), p_goal - Decimal("0.20"))
            )
    capital = sum((row.capital_required for row in combo), Decimal("0"))
    maximum_loss = sum((row.maximum_loss for row in combo), Decimal("0"))
    expected = sum((row.expected_pnl for row in combo), Decimal("0"))
    concentration = (
        max((row.capital_required for row in combo), default=Decimal("0"))
        / max(available_capital_usd, Decimal("0.01"))
    ).quantize(Decimal("0.0001"))
    average_liquidity = (
        sum((Decimal(str(row.liquidity_score)) for row in combo), Decimal("0"))
        / Decimal(len(combo))
    )
    liquidity_penalty = (Decimal("1") - average_liquidity).quantize(
        Decimal("0.0001")
    )
    delta = (
        (p_goal - wait_probability_goal).quantize(Decimal("0.0001"))
        if p_goal is not None and wait_probability_goal is not None
        else None
    )
    return PortfolioAttainmentEvaluation(
        portfolio_id=uuid4(),
        component_evaluation_ids=tuple(row.evaluation_id for row in combo),
        component_contract_ids=tuple(row.contract_id for row in combo),
        component_quantities=tuple(row.quantity for row in combo),
        capital_deployed=capital.quantize(Decimal("0.01")),
        maximum_loss=maximum_loss.quantize(Decimal("0.01")),
        expected_pnl=expected.quantize(Decimal("0.01")),
        probability_goal=p_goal,
        probability_wait=wait_probability_goal,
        probability_delta=delta,
        lower_probability_bound=lower,
        maximum_resolution_seconds=max(
            (row.estimated_resolution_seconds for row in combo), default=0
        ),
        concentration_penalty=concentration,
        liquidity_penalty=liquidity_penalty,
        reason_codes=(
            ("shared_underlying_scenarios",)
            if scenarios_compatible
            else ("incompatible_shared_scenario_grid",)
        ),
        physically_feasible=(
            capital <= available_capital_usd
            and scenarios_compatible
            and p_goal is not None
        ),
        shared_scenario_grid_hash=(
            reference_scenarios[0].shared_scenario_grid_hash
            if scenarios_compatible
            else None
        ),
        shared_scenarios=(
            tuple(
                SharedUnderlyingScenario(
                    scenario_id=scenario.scenario_id,
                    probability=scenario.probability,
                    underlying_price=scenario.underlying_price,
                    horizon_seconds=scenario.horizon_seconds,
                    generation_method="referenced_shared_grid",
                    assumptions=("portfolio_components_priced_on_identical_spots",),
                )
                for scenario in reference_scenarios
            )
            if scenarios_compatible
            else ()
        ),
    )


def _row_rank_key(row: QuantityGridRow) -> tuple[Any, ...]:
    return (
        row.probability_goal if row.probability_goal is not None else Decimal("-1"),
        row.lower_probability_bound
        if row.lower_probability_bound is not None
        else Decimal("-1"),
        row.expected_pnl,
        -row.relative_spread,
        -row.capital_required,
        row.contract_id,
        -row.quantity,
        str(row.strategy_id),
    )


def _row_search_key(row: QuantityGridRow) -> tuple[Any, ...]:
    return (str(row.strategy_id), row.contract_id, row.quantity)


def _portfolio_rank_key(portfolio: PortfolioAttainmentEvaluation) -> tuple[Any, ...]:
    return (
        portfolio.probability_goal
        if portfolio.probability_goal is not None
        else Decimal("-1"),
        portfolio.lower_probability_bound
        if portfolio.lower_probability_bound is not None
        else Decimal("-1"),
        -portfolio.concentration_penalty,
        -portfolio.liquidity_penalty,
        -portfolio.maximum_resolution_seconds,
        -portfolio.capital_deployed,
        tuple(portfolio.component_contract_ids),
        tuple(-q for q in portfolio.component_quantities),
    )


def _replace_row_selected(
    row: QuantityGridRow, selected: bool
) -> QuantityGridRow:
    from dataclasses import replace

    return replace(row, selected=selected)


def _replace_portfolio_selected(
    portfolio: PortfolioAttainmentEvaluation, selected: bool
) -> PortfolioAttainmentEvaluation:
    from dataclasses import replace

    return replace(portfolio, selected=selected)


def _wait_decision(
    *,
    decision_id: UUID,
    snapshot_id: UUID,
    objective_version: int,
    time_remaining_seconds: int,
    wait_probability_goal: Decimal | None,
    evaluated_objective_fingerprint: str | None,
    quantity_grid: Sequence[QuantityGridRow],
    portfolios: Sequence[PortfolioAttainmentEvaluation],
    reason: str,
    evaluated_at_exchange_time: datetime | None = None,
    decision_valid_until_exchange_time: datetime | None = None,
    maximum_decision_age_seconds: int | None = None,
) -> TargetPortfolioDecision:
    return TargetPortfolioDecision(
        decision_id=decision_id,
        action=PortfolioAction.WAIT,
        authorized_positions=(),
        selected_portfolio_id=None,
        selected_probability_goal=wait_probability_goal,
        wait_probability_goal=wait_probability_goal,
        probability_delta=Decimal("0") if wait_probability_goal is not None else None,
        snapshot_id=snapshot_id,
        objective_version=objective_version,
        time_remaining_seconds=time_remaining_seconds,
        reason_codes=(reason,),
        quantity_grid=tuple(quantity_grid),
        portfolio_evaluations=tuple(portfolios),
        evaluated_objective_fingerprint=evaluated_objective_fingerprint,
        evaluated_at_exchange_time=evaluated_at_exchange_time,
        decision_valid_until_exchange_time=decision_valid_until_exchange_time,
        maximum_decision_age_seconds=maximum_decision_age_seconds,
        required_resolution_horizon_seconds=0,
    )

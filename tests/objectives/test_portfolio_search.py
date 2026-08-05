from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

from joker.objectives.contract_outcomes import (
    ContractOutcomeEstimate,
    ContractScenarioOutcome,
)
from joker.objectives.portfolio_search import (
    PortfolioAction,
    PortfolioSearchSettings,
    expand_quantity_grid,
    search_target_portfolios,
)


def _outcome(
    contract_id: str,
    *,
    cost: str,
    up_pnl: str,
    down_pnl: str | None = None,
    up_probability: str = "0.40",
) -> ContractOutcomeEstimate:
    cost_d = Decimal(cost)
    up = Decimal(up_pnl)
    down = Decimal(down_pnl or f"-{cost_d}")
    up_p = Decimal(up_probability)
    return ContractOutcomeEstimate(
        estimate_id=uuid4(),
        strategy_id=uuid4(),
        strategy_family="directional",
        contract_id=contract_id,
        strike=Decimal("500"),
        option_type="call",
        distance_from_spot=Decimal("0"),
        bid=cost_d - Decimal("0.01"),
        ask=cost_d,
        midpoint=cost_d - Decimal("0.005"),
        relative_spread=Decimal("0.02"),
        liquidity_score=0.8,
        delta=Decimal("0.5"),
        gamma=Decimal("0.02"),
        theta=Decimal("-0.1"),
        implied_volatility=Decimal("0.3"),
        evaluation_premium=cost_d,
        maximum_loss_usd_per_contract=cost_d * Decimal("100"),
        estimated_useful_upside_usd=up,
        expected_pnl_usd=up_p * up + (Decimal("1") - up_p) * down,
        estimated_resolution_seconds=300,
        required_contract_price=cost_d + Decimal("1"),
        probability_reaches_required_price=up_p,
        probability_closes_goal_gap=up_p,
        lower_probability_bound=max(Decimal("0"), up_p - Decimal("0.1")),
        upper_probability_bound=min(Decimal("1"), up_p + Decimal("0.1")),
        estimate_type="market_greeks_scenario",
        assumptions=("test_shared_scenarios",),
        uncertainty_reasons=(),
        evidence_ids=(),
        historical_sample_count=0,
        scenarios=(
            ContractScenarioOutcome(
                scenario_id="up",
                probability=up_p,
                underlying_price=Decimal("505"),
                estimated_option_price=cost_d + up / Decimal("100"),
                pnl_per_contract_usd=up,
            ),
            ContractScenarioOutcome(
                scenario_id="down",
                probability=Decimal("1") - up_p,
                underlying_price=Decimal("495"),
                estimated_option_price=Decimal("0"),
                pnl_per_contract_usd=down,
            ),
        ),
        usable_for_ranking=True,
    )


def _decision(
    rows,
    *,
    gap: str = "100",
    budget: str = "200",
    max_positions: int = 2,
    wait: str = "0.10",
):
    return search_target_portfolios(
        quantity_grid=rows,
        snapshot_id=uuid4(),
        objective_version=3,
        time_remaining_seconds=900,
        remaining_goal_gap_usd=Decimal(gap),
        available_capital_usd=Decimal(budget),
        open_position_count=0,
        working_order_count=0,
        max_concurrent_positions=max_positions,
        wait_probability_goal=Decimal(wait),
        settings=PortfolioSearchSettings(
            beam_width=50,
            maximum_portfolio_candidates=500,
            minimum_probability_improvement_over_wait=Decimal("0.01"),
        ),
    )


def test_every_affordable_quantity_is_evaluated_without_exceeding_capital() -> None:
    rows = expand_quantity_grid(
        outcomes=[_outcome("A", cost="0.50", up_pnl="80")],
        available_capital_usd=Decimal("200"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.10"),
        maximum_authorised_contracts=20,
    )
    assert [row.quantity for row in rows] == [1, 2, 3, 4]
    assert all(row.capital_required <= Decimal("200") for row in rows)


def test_two_smaller_positions_can_beat_every_single_position() -> None:
    outcomes = [
        _outcome("A", cost="0.50", up_pnl="60"),
        _outcome("B", cost="0.50", up_pnl="60"),
    ]
    rows = expand_quantity_grid(
        outcomes=outcomes,
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.10"),
        maximum_authorised_contracts=1,
    )
    decision = _decision(rows, budget="100", max_positions=2)
    assert decision.action == PortfolioAction.ENTER
    assert len(decision.authorized_positions) == 2
    assert sum(
        (position.capital_allocation for position in decision.authorized_positions),
        Decimal("0"),
    ) <= Decimal("100")


def test_one_concentrated_position_wins_when_superior() -> None:
    outcomes = [
        _outcome("A", cost="0.50", up_pnl="70", up_probability="0.60"),
        _outcome("B", cost="0.50", up_pnl="20", up_probability="0.30"),
    ]
    rows = expand_quantity_grid(
        outcomes=outcomes,
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.05"),
        maximum_authorised_contracts=2,
    )
    decision = _decision(rows, budget="100", max_positions=2, wait="0.05")
    assert decision.action == PortfolioAction.ENTER
    assert len(decision.authorized_positions) == 1
    assert decision.authorized_positions[0].contract_id == "A"
    assert decision.authorized_positions[0].quantity == 2


def test_wait_wins_when_no_portfolio_improves_probability() -> None:
    rows = expand_quantity_grid(
        outcomes=[_outcome("A", cost="0.50", up_pnl="20", up_probability="0.10")],
        available_capital_usd=Decimal("50"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.30"),
        maximum_authorised_contracts=1,
    )
    decision = _decision(rows, budget="50", max_positions=1, wait="0.30")
    assert decision.action == PortfolioAction.WAIT
    assert decision.authorized_positions == ()


def test_candidate_order_does_not_change_authoritative_selection() -> None:
    outcomes = [
        _outcome("A", cost="0.50", up_pnl="60"),
        _outcome("B", cost="0.50", up_pnl="60"),
        _outcome("C", cost="0.25", up_pnl="10"),
    ]
    rows = expand_quantity_grid(
        outcomes=outcomes,
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.10"),
        maximum_authorised_contracts=1,
    )
    forward = _decision(rows, budget="100", max_positions=2)
    reverse = _decision(tuple(reversed(rows)), budget="100", max_positions=2)
    forward_tuples = [
        (position.contract_id, position.quantity)
        for position in forward.authorized_positions
    ]
    reverse_tuples = [
        (position.contract_id, position.quantity)
        for position in reverse.authorized_positions
    ]
    assert forward.action == reverse.action
    assert forward_tuples == reverse_tuples


def test_position_limit_and_no_duplicate_contracts_are_respected() -> None:
    rows = expand_quantity_grid(
        outcomes=[
            _outcome("A", cost="0.25", up_pnl="60"),
            _outcome("B", cost="0.25", up_pnl="60"),
        ],
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.10"),
        maximum_authorised_contracts=2,
    )
    decision = _decision(rows, budget="100", max_positions=1)
    assert len(decision.authorized_positions) <= 1
    assert len(
        {position.contract_id for position in decision.authorized_positions}
    ) == len(decision.authorized_positions)
    assert all(row.option_type in {"call", "put"} for row in rows)


def _outcome_with_scenarios(
    contract_id: str,
    *,
    cost: str,
    scenarios: tuple[ContractScenarioOutcome, ...],
) -> ContractOutcomeEstimate:
    cost_d = Decimal(cost)
    expected = sum(
        (s.probability * s.pnl_per_contract_usd for s in scenarios), Decimal("0")
    )
    return ContractOutcomeEstimate(
        estimate_id=uuid4(),
        strategy_id=uuid4(),
        strategy_family="directional",
        contract_id=contract_id,
        strike=Decimal("500"),
        option_type="call",
        distance_from_spot=Decimal("0"),
        bid=cost_d - Decimal("0.01"),
        ask=cost_d,
        midpoint=cost_d - Decimal("0.005"),
        relative_spread=Decimal("0.02"),
        liquidity_score=0.8,
        delta=Decimal("0.5"),
        gamma=Decimal("0.02"),
        theta=Decimal("-0.1"),
        implied_volatility=Decimal("0.3"),
        evaluation_premium=cost_d,
        maximum_loss_usd_per_contract=cost_d * Decimal("100"),
        estimated_useful_upside_usd=max(
            (s.pnl_per_contract_usd for s in scenarios), default=Decimal("0")
        ),
        expected_pnl_usd=expected,
        estimated_resolution_seconds=300,
        required_contract_price=cost_d + Decimal("1"),
        probability_reaches_required_price=Decimal("0.4"),
        probability_closes_goal_gap=Decimal("0.4"),
        lower_probability_bound=Decimal("0.3"),
        upper_probability_bound=Decimal("0.5"),
        estimate_type="market_greeks_scenario",
        assumptions=("test_shared_scenarios",),
        uncertainty_reasons=(),
        evidence_ids=(),
        historical_sample_count=0,
        scenarios=scenarios,
        usable_for_ranking=True,
        shared_scenario_grid_hash=scenarios[0].shared_scenario_grid_hash
        if scenarios
        else None,
    )


def test_incompatible_strategy_distributions_fail_closed() -> None:
    from joker.objectives.portfolio_search import _evaluate_portfolio

    left = expand_quantity_grid(
        outcomes=[
            _outcome_with_scenarios(
                "A",
                cost="0.50",
                scenarios=(
                    ContractScenarioOutcome(
                        scenario_id="up",
                        probability=Decimal("0.40"),
                        underlying_price=Decimal("505"),
                        estimated_option_price=Decimal("1.10"),
                        pnl_per_contract_usd=Decimal("60"),
                        shared_scenario_grid_hash="grid-a",
                    ),
                    ContractScenarioOutcome(
                        scenario_id="down",
                        probability=Decimal("0.60"),
                        underlying_price=Decimal("495"),
                        estimated_option_price=Decimal("0"),
                        pnl_per_contract_usd=Decimal("-50"),
                        shared_scenario_grid_hash="grid-a",
                    ),
                ),
            )
        ],
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.10"),
        maximum_authorised_contracts=1,
    )
    right = expand_quantity_grid(
        outcomes=[
            _outcome_with_scenarios(
                "B",
                cost="0.50",
                scenarios=(
                    ContractScenarioOutcome(
                        scenario_id="up",
                        probability=Decimal("0.70"),
                        underlying_price=Decimal("510"),
                        estimated_option_price=Decimal("1.50"),
                        pnl_per_contract_usd=Decimal("80"),
                        shared_scenario_grid_hash="grid-b",
                    ),
                    ContractScenarioOutcome(
                        scenario_id="down",
                        probability=Decimal("0.30"),
                        underlying_price=Decimal("490"),
                        estimated_option_price=Decimal("0"),
                        pnl_per_contract_usd=Decimal("-50"),
                        shared_scenario_grid_hash="grid-b",
                    ),
                ),
            )
        ],
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.10"),
        maximum_authorised_contracts=1,
    )
    portfolio = _evaluate_portfolio(
        combo=(left[0], right[0]),
        remaining_goal_gap_usd=Decimal("100"),
        available_capital_usd=Decimal("200"),
        wait_probability_goal=Decimal("0.10"),
        cfg=PortfolioSearchSettings(),
    )
    assert portfolio.physically_feasible is False
    assert "incompatible_shared_scenario_grid" in portfolio.reason_codes
    assert portfolio.probability_goal is None


def test_portfolio_p_goal_uses_aggregate_pnl_on_common_spots() -> None:
    from joker.objectives.portfolio_search import _evaluate_portfolio

    shared_hash = "shared-grid"
    # Individually neither closes a $100 gap on the up spot; together they do.
    a = expand_quantity_grid(
        outcomes=[
            _outcome_with_scenarios(
                "A",
                cost="0.50",
                scenarios=(
                    ContractScenarioOutcome(
                        scenario_id="up",
                        probability=Decimal("0.40"),
                        underlying_price=Decimal("505"),
                        estimated_option_price=Decimal("1.10"),
                        pnl_per_contract_usd=Decimal("60"),
                        shared_scenario_grid_hash=shared_hash,
                    ),
                    ContractScenarioOutcome(
                        scenario_id="down",
                        probability=Decimal("0.60"),
                        underlying_price=Decimal("495"),
                        estimated_option_price=Decimal("0"),
                        pnl_per_contract_usd=Decimal("-50"),
                        shared_scenario_grid_hash=shared_hash,
                    ),
                ),
            )
        ],
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.05"),
        maximum_authorised_contracts=1,
    )
    b = expand_quantity_grid(
        outcomes=[
            _outcome_with_scenarios(
                "B",
                cost="0.50",
                scenarios=(
                    ContractScenarioOutcome(
                        scenario_id="up",
                        probability=Decimal("0.40"),
                        underlying_price=Decimal("505"),
                        estimated_option_price=Decimal("1.10"),
                        pnl_per_contract_usd=Decimal("60"),
                        shared_scenario_grid_hash=shared_hash,
                    ),
                    ContractScenarioOutcome(
                        scenario_id="down",
                        probability=Decimal("0.60"),
                        underlying_price=Decimal("495"),
                        estimated_option_price=Decimal("0"),
                        pnl_per_contract_usd=Decimal("-50"),
                        shared_scenario_grid_hash=shared_hash,
                    ),
                ),
            )
        ],
        available_capital_usd=Decimal("100"),
        remaining_goal_gap_usd=Decimal("100"),
        wait_probability_goal=Decimal("0.05"),
        maximum_authorised_contracts=1,
    )
    assert a[0].probability_goal == Decimal("0.0000")
    assert b[0].probability_goal == Decimal("0.0000")
    portfolio = _evaluate_portfolio(
        combo=(a[0], b[0]),
        remaining_goal_gap_usd=Decimal("100"),
        available_capital_usd=Decimal("200"),
        wait_probability_goal=Decimal("0.05"),
        cfg=PortfolioSearchSettings(),
    )
    assert portfolio.physically_feasible is True
    assert portfolio.probability_goal == Decimal("0.4000")

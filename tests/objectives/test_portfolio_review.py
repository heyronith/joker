"""Non-authoritative portfolio review context and finalizer behavior."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from joker.graph.cognitive_graph import build_cognitive_graph
from joker.objectives.config import FullChainOptimizerSettings
from joker.objectives.portfolio_review import (
    build_portfolio_review_context,
    portfolio_review_from_debate,
)
from joker.objectives.target_attainment import TargetAttainmentPolicy
from tests.integration.test_goal_driven_full_graph import (
    _prepare_stack,
    _teardown_stack,
)


def _review_state(*, limit_candidates: int = 5) -> dict:
    decision_id = uuid4()
    snapshot_id = uuid4()
    portfolio_id = uuid4()
    contracts = []
    portfolios = []
    for index in range(limit_candidates + 3):
        contracts.append(
            {
                "evaluation_id": str(uuid4()),
                "strategy_id": str(uuid4()),
                "contract_id": f"SPY:C{index}",
                "quantity": index + 1,
                "capital_required": str(Decimal(index + 1) * Decimal("10")),
                "probability_goal": str(Decimal("0.50") - Decimal(index) * Decimal("0.01")),
                "lower_probability_bound": str(
                    Decimal("0.40") - Decimal(index) * Decimal("0.01")
                ),
                "estimate_type": "market_greeks_scenario",
                "relative_spread": "0.05",
                "liquidity_score": 0.8,
                "outcome_estimate_id": str(uuid4()),
            }
        )
        portfolios.append(
            {
                "portfolio_id": str(portfolio_id if index == 0 else uuid4()),
                "component_contract_ids": [f"SPY:C{index}"],
                "component_quantities": [index + 1],
                "capital_deployed": "50",
                "probability_goal": str(
                    Decimal("0.55") - Decimal(index) * Decimal("0.01")
                ),
                "lower_probability_bound": "0.45",
                "concentration_penalty": "0.10",
                "liquidity_penalty": "0.05",
                "shared_scenario_grid_hash": "grid",
                "reason_codes": ["shared_underlying_scenarios"],
                "selected": index == 0,
            }
        )
    return {
        "_target_portfolio_decision": {
            "decision_id": str(decision_id),
            "snapshot_id": str(snapshot_id),
            "evaluated_objective_version": 4,
            "selected_portfolio_id": str(portfolio_id),
            "wait_probability_goal": "0.12",
            "time_remaining_seconds": 900,
        },
        "_quantity_grid": contracts,
        "_portfolio_grid": portfolios,
        "_contract_outcomes": [],
        "_objective_context": {
            "required_profit_remaining_usd": "25",
            "deadline_exchange_time": "2026-08-05T16:00:00+00:00",
        },
        "_shared_underlying_scenario_grid": {
            "grid_hash": "grid",
            "horizon_seconds": 300,
            "generation_method": "deterministic_symmetric_grid",
            "scenarios": [{"scenario_id": "z0"}, {"scenario_id": "z1"}],
        },
    }


def test_portfolio_review_receives_ranked_contracts_and_portfolios() -> None:
    context = build_portfolio_review_context(state=_review_state(), limit=5)
    assert context is not None
    assert len(context.ranked_contract_candidates) == 5
    assert len(context.ranked_portfolio_candidates) == 5
    assert context.selected_candidate is not None
    assert context.ranked_contract_candidates[0].rank == 1
    assert context.ranked_portfolio_candidates[0].rank == 1


def test_portfolio_review_includes_wait_candidate() -> None:
    context = build_portfolio_review_context(state=_review_state(), limit=3)
    assert context is not None
    assert context.wait_candidate["action"] == "wait"
    assert context.wait_candidate["probability_goal"] == "0.12"
    assert "explicit_wait_candidate" in context.wait_candidate["reason_codes"]


def test_top_candidates_for_agent_review_controls_agent_context() -> None:
    state = _review_state(limit_candidates=10)
    narrow = build_portfolio_review_context(state=state, limit=2)
    wide = build_portfolio_review_context(state=state, limit=8)
    assert narrow is not None and wide is not None
    assert len(narrow.ranked_contract_candidates) == 2
    assert len(narrow.ranked_portfolio_candidates) == 2
    assert len(wide.ranked_contract_candidates) == 8
    assert len(wide.ranked_portfolio_candidates) == 8


def test_portfolio_reviewer_cannot_mutate_authorized_tuple() -> None:
    context = build_portfolio_review_context(state=_review_state(), limit=3)
    assert context is not None
    original_selected = context.selected_candidate
    assert original_selected is not None
    review = portfolio_review_from_debate(
        SimpleNamespace(
            review_id=uuid4(),
            reviewer_role="risk",
            verdict="support",
            confidence=0.9,
            claims=("ok",),
            identified_failure_modes=(),
            required_revisions=(),
        ),
        context,
    )
    assert review.selected_portfolio_id == original_selected.portfolio_id
    assert review.finalizer_recommendation == "preserve"
    # Review payload cannot rewrite ranked quantities/contracts.
    assert context.ranked_portfolio_candidates[0].component_quantities == (
        original_selected.component_quantities
    )


def test_portfolio_reviewer_can_force_wait_or_reoptimization() -> None:
    context = build_portfolio_review_context(state=_review_state(), limit=2)
    assert context is not None
    wait_review = portfolio_review_from_debate(
        SimpleNamespace(
            review_id=uuid4(),
            reviewer_role="evidence",
            verdict="request_more_evidence",
            confidence=0.4,
            claims=(),
            identified_failure_modes=("thin_evidence",),
            required_revisions=(),
        ),
        context,
    )
    reopt_review = portfolio_review_from_debate(
        SimpleNamespace(
            review_id=uuid4(),
            reviewer_role="risk",
            verdict="oppose",
            confidence=0.8,
            claims=("concentration",),
            identified_failure_modes=("crowded",),
            required_revisions=("rebuild",),
        ),
        context,
    )
    assert wait_review.finalizer_recommendation == "wait"
    assert reopt_review.finalizer_recommendation == "reoptimize"


@pytest.mark.asyncio
async def test_final_authority_occurs_after_portfolio_review(tmp_path) -> None:
    stack = await _prepare_stack(
        tmp_path,
        pnl=Decimal("15"),
        n=20,
        objective_duration=timedelta(minutes=4),
    )
    try:
        deps = stack["deps"]
        deps.objective_policy = "target_attainment"
        deps.objective_service.objective_policy = "target_attainment"
        deps.target_attainment_policy = TargetAttainmentPolicy()
        deps.full_chain_optimizer_settings = FullChainOptimizerSettings(
            enabled=True,
            maximum_quote_age_seconds=3600,
            maximum_surface_age_seconds=3600,
            minimum_probability_improvement_over_wait=0,
        )
        result = await build_cognitive_graph(deps).ainvoke(
            stack["state"], config=stack["config"]
        )
        trace_names = [trace.node_name for trace in result.get("node_trace") or ()]
        assert "debate_panel" in trace_names
        assert "finalize_portfolio_review" in trace_names
        assert "meta_decision" in trace_names
        debate_idx = trace_names.index("debate_panel")
        finalize_idx = next(
            i
            for i, name in enumerate(trace_names)
            if name == "finalize_portfolio_review"
        )
        meta_idx = next(
            i for i, name in enumerate(trace_names) if name == "meta_decision"
        )
        assert debate_idx < finalize_idx < meta_idx
        assert result.get("_portfolio_review_finalization") is not None
    finally:
        await _teardown_stack(stack)

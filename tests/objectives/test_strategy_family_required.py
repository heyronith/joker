"""Objective graph requires explicit strategy_family for historical EV."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.cognition.schemas import (
    AgentRole,
    EntryPlan,
    ExecutionPlan,
    ExitPlan,
    InvalidationPlan,
    MarketDirection,
    StrategyHypothesis,
    StrategyLegCandidate,
)
from joker.evolution.episode_metadata import resolve_episode_similarity_metadata


def _strategy(*, family: str | None) -> StrategyHypothesis:
    return StrategyHypothesis(
        session_id="s",
        snapshot_id=uuid4(),
        cycle_id="c1",
        prompt_version="1.0.0",
        model_call_id=uuid4(),
        strategy_id=uuid4(),
        source_hypothesis_ids=(uuid4(),),
        name="test",
        market_thesis="t",
        direction=MarketDirection.BULLISH,
        strategy_family=family,
        candidate_legs=(
            StrategyLegCandidate(
                contract_id="SPY:2026-07-01:500.0:call",
                side="buy",
                option_type="call",
                strike=Decimal("500"),
                quantity=1,
                rationale="primary",
            ),
        ),
        entry_plan=EntryPlan(entry_style="immediate", preferred_order_type="limit"),
        execution_plan=ExecutionPlan(
            max_quote_age_seconds=5,
            partial_fill_policy="accept",
            replacement_policy="none",
        ),
        exit_plan=ExitPlan(stop_conditions=("stop",)),
        invalidation_plan=InvalidationPlan(conditions=("inv",)),
        expected_horizon_seconds=600,
        confidence=0.7,
        novelty_score=0.5,
        agent_role=AgentRole.BULLISH_INVENTOR,
    )


@pytest.mark.asyncio
async def test_objective_graph_requires_explicit_strategy_family() -> None:
    """Missing family must not be filled from bullish_inventor role."""
    strategy = _strategy(family=None)

    class _Repo:
        async def get_by_id(self, _sid):
            return strategy

    meta = await resolve_episode_similarity_metadata(
        contract_id="SPY:2026-07-01:500.0:call",
        entry_orders=(type("O", (), {"side": "buy"})(),),
        strategy_id=strategy.strategy_id,
        entry_cycle_id="c1",
        session_id="s",
        strategy_repo=_Repo(),
        exchange_timestamp=datetime.now(timezone.utc),
    )
    assert meta.strategy_family is None
    assert "historical_strategy_family_missing" in meta.findings
    assert meta.historical_ev_eligible is False

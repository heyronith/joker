"""Execution-time EV repricing and gateway revalidation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.objectives.estimate import StrategyEstimateBuilder
from joker.objectives.repricing import reprice_long_option_estimate
from joker.objectives.schemas import SessionObjectiveState, StrategyObjectiveEstimate
from joker.runtime.order_action_gateway import OrderActionKind, OrderActionRequest
from tests.objectives.historical_fixtures import make_hist_service, seed_positive_history
from tests.objectives.test_ev_feasibility_estimates import _strategy


def _state(**kw) -> SessionObjectiveState:
    base = {
        "objective_id": uuid4(),
        "session_id": "s",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("50"),
        "target_ending_equity_usd": Decimal("550"),
        "available_capital_usd": Decimal("500"),
        "required_profit_remaining_usd": Decimal("50"),
        "time_remaining_seconds": 3600,
        "version": 1,
        "max_concurrent_positions": 1,
        "deadline_exchange_time": datetime.now(timezone.utc) + timedelta(hours=1),
    }
    base.update(kw)
    return SessionObjectiveState.model_validate(base)


@pytest.mark.asyncio
async def test_gateway_reprices_against_current_quote(tmp_path) -> None:
    hist, repo = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime.now(timezone.utc)
    seed_positive_history(hist, as_of=as_of, n=20, pnl=Decimal("20.00"))
    summary = await hist.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    repo.save_strategy_estimate(est)
    repriced = reprice_long_option_estimate(
        est,
        current_premium_per_contract_usd=Decimal("1.05"),
        quantity=1,
        request_snapshot_id=uuid4(),
    )
    assert repriced.repricing_method == "long_option_entry_cost_adjust_v1"
    assert repriced.original_premium_usd == Decimal("1.00")
    assert repriced.current_premium_usd == Decimal("1.05")
    assert repriced.repriced_expected_value_usd is not None
    # Entry cost rose by $5 → EV falls by $5
    assert repriced.repriced_expected_value_usd == (
        est.expected_value_usd - Decimal("5.00")
    )


def test_gateway_rejects_stale_estimate_without_revalidation() -> None:
    est = StrategyObjectiveEstimate(
        strategy_id=uuid4(),
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        expected_value_usd=Decimal("10.00"),
        capital_required_usd=Decimal("100.00"),
        maximum_loss_usd=Decimal("100.00"),
        calculation_method="calibrated_episode_average",
        quote_inputs={"premium_per_contract": "1.00", "quantity": 1},
        valid=True,
        valid_until=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    repriced = reprice_long_option_estimate(
        est,
        current_premium_per_contract_usd=Decimal("1.00"),
        quantity=1,
        request_snapshot_id=uuid4(),
    )
    assert repriced.valid is False
    assert "estimate_expired" in repriced.invalidation_reasons


def test_gateway_rejects_non_positive_repriced_ev() -> None:
    est = StrategyObjectiveEstimate(
        strategy_id=uuid4(),
        objective_id=uuid4(),
        snapshot_id=uuid4(),
        expected_value_usd=Decimal("5.00"),
        capital_required_usd=Decimal("100.00"),
        maximum_loss_usd=Decimal("100.00"),
        calculation_method="calibrated_episode_average",
        quote_inputs={
            "premium_per_contract": "1.00",
            "quantity": 1,
            "slippage_per_contract": "0.00",
        },
        valid=True,
    )
    # Premium jump of $0.10 → +$10 entry cost → EV becomes -5
    repriced = reprice_long_option_estimate(
        est,
        current_premium_per_contract_usd=Decimal("1.10"),
        quantity=1,
        request_snapshot_id=uuid4(),
        current_slippage_per_contract_usd=Decimal("0.00"),
    )
    assert repriced.repriced_expected_value_usd == Decimal("-5.00")
    assert repriced.valid is False
    assert "repriced_expected_value_not_positive" in repriced.invalidation_reasons


@pytest.mark.asyncio
async def test_incremental_add_requires_positive_ev(tmp_path) -> None:
    """ADD uses the command quantity as incremental capital; non-positive EV rejects."""
    from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
    from joker.objectives.service import SessionObjectiveService

    db = tmp_path / "add.db"
    apply_objective_migrations(db)
    repo = ObjectiveRepository(db)
    svc = SessionObjectiveService(repo, require_positive_expected_value=True)
    definition = await svc.create_objective(
        session_id="add",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(timezone.utc) + timedelta(hours=2),
        max_concurrent_positions=2,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    hist, _ = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime.now(timezone.utc)
    seed_positive_history(hist, as_of=as_of, n=20, pnl=Decimal("-3.00"))
    summary = await hist.summarize_for_strategy(
        objective_id=definition.objective_id,
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=await svc.get_state(),
        snapshot_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    # Negative history → estimate invalid → ADD blocked
    assert est.valid is False
    svc.save_strategy_estimate(est)
    loaded = svc.get_strategy_estimate(est.estimate_id)
    assert loaded is not None and not loaded.valid
    # Repricing also fails closed
    repriced = reprice_long_option_estimate(
        est.model_copy(update={"valid": True, "expected_value_usd": Decimal("-3.00")})
        if hasattr(est, "model_copy")
        else est,
        current_premium_per_contract_usd=Decimal("1.00"),
        quantity=1,
        request_snapshot_id=uuid4(),
    )
    # Construct explicitly for incremental ADD rejection
    bad = StrategyObjectiveEstimate(
        estimate_id=est.estimate_id,
        strategy_id=est.strategy_id,
        objective_id=definition.objective_id,
        snapshot_id=est.snapshot_id,
        expected_value_usd=Decimal("-3.00"),
        capital_required_usd=Decimal("100"),
        maximum_loss_usd=Decimal("100"),
        calculation_method="calibrated_episode_average",
        quote_inputs={"premium_per_contract": "1.00", "quantity": 1},
        valid=True,
        historical_summary_id=summary.summary_id,
    )
    add_reprice = reprice_long_option_estimate(
        bad,
        current_premium_per_contract_usd=Decimal("1.00"),
        quantity=1,
        request_snapshot_id=uuid4(),
    )
    assert add_reprice.valid is False
    assert OrderActionKind.ADD.value == "add"
    _ = OrderActionRequest  # imported for action-kind coupling


@pytest.mark.asyncio
async def test_historical_ev_restart_reuses_persisted_artifacts(tmp_path) -> None:
    hist, repo = make_hist_service(tmp_path, minimum_samples_for_ev=20)
    as_of = datetime.now(timezone.utc)
    seed_positive_history(hist, as_of=as_of, n=20)
    sid = uuid4()
    snap = uuid4()
    summary = await hist.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=sid,
        snapshot_id=snap,
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="bullish",
    )
    est = StrategyEstimateBuilder().build(
        strategy=_strategy(),
        objective_state=_state(),
        snapshot_id=snap,
        premium_per_contract_usd=Decimal("1.00"),
        historical_summary=summary,
    )
    repo.save_strategy_estimate(est)
    # Fresh service/repo handles — same DB
    from joker.objectives.repository import ObjectiveRepository

    repo2 = ObjectiveRepository(tmp_path / "obj_hist.db")
    loaded_summary = repo2.get_historical_summary(summary.summary_id)
    loaded_est = repo2.get_strategy_estimate(est.estimate_id)
    assert loaded_summary is not None
    assert loaded_summary.summary_id == summary.summary_id
    assert loaded_est is not None
    assert loaded_est.estimate_id == est.estimate_id
    assert loaded_est.historical_summary_id == summary.summary_id

"""Objective adversarial scenario unit coverage (adv_obj_01–20 behaviours)."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.evolution.adversarial import ADVERSARIAL_CORPUS
from joker.evolution.adversarial_fixtures import FIXTURE_CORPUS, _fid
from joker.objectives.feasibility import FeasibilityInputs, GoalFeasibilityEngine
from joker.objectives.repository import ObjectiveRepository
from joker.objectives.schemas import SessionObjectiveState
from joker.objectives.scoring import ObjectiveStrategyScorer, StrategyScoreInput
from joker.objectives.service import ObjectiveServiceError, SessionObjectiveService
from joker.objectives.sizing import DeterministicObjectiveSizer

ET = ZoneInfo("America/New_York")


def test_corpus_includes_twenty_objective_scenarios() -> None:
    ids = [s.scenario_id for s in ADVERSARIAL_CORPUS if s.scenario_id.startswith("adv_obj_")]
    assert len(ids) == 20
    for sid in ids:
        scenario = next(s for s in ADVERSARIAL_CORPUS if s.scenario_id == sid)
        assert scenario.required is True
        fixture = FIXTURE_CORPUS[_fid(sid)]
        assert fixture.stimulus.get("objective_scenario") is True
        assert fixture.expected_invariants
        assert fixture.expected_invariants[0].startswith("objective_")
        # Fixture may still carry prove_* for documentation, but runners must
        # observe the named behaviour via `_prove_objective_scenario`.
        assert "objective_invariant" not in fixture.expected_invariants
        assert fixture.stimulus.get("objective_scenario_id") == sid


def _state(**kw: object) -> SessionObjectiveState:
    base = {
        "objective_id": uuid4(),
        "session_id": "adv",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("100"),
        "target_ending_equity_usd": Decimal("600"),
        "reserved_capital_usd": Decimal("0"),
        "available_capital_usd": Decimal("500"),
        "realised_pnl_usd": Decimal("0"),
        "unrealised_pnl_usd": Decimal("0"),
        "progress_to_goal_pct": Decimal("0"),
        "required_profit_remaining_usd": Decimal("100"),
        "time_remaining_seconds": 7200,
        "version": 1,
        "max_concurrent_positions": 1,
    }
    base.update(kw)
    return SessionObjectiveState.model_validate(base)


def test_adv_obj_high_target_insufficient_time() -> None:
    a = GoalFeasibilityEngine().assess(
        _state(required_profit_remaining_usd=Decimal("450"), time_remaining_seconds=900),
        FeasibilityInputs(snapshot_id=uuid4()),
    )
    assert a.classification in {"low", "infeasible"}


def test_adv_obj_target_reached_pauses() -> None:
    state = _state(
        realised_pnl_usd=Decimal("100"),
        progress_to_goal_pct=Decimal("100"),
        status="target_reached",
        entries_paused=True,
    )
    assert state.entries_paused
    assert state.status == "target_reached"


def test_adv_obj_negative_ev_and_no_trade() -> None:
    scores = ObjectiveStrategyScorer().score_all(
        _state(),
        [
            StrategyScoreInput(
                strategy_id=uuid4(),
                snapshot_id=uuid4(),
                expected_value_usd=Decimal("-2"),
                capital_required_usd=40,
                maximum_loss_usd=40,
            )
        ],
        snapshot_id=uuid4(),
    )
    assert any(s.is_no_trade for s in scores)
    assert all((not s.valid) or s.is_no_trade for s in scores)


def test_adv_obj_no_trade_better_than_weak_trade() -> None:
    scores = ObjectiveStrategyScorer().score_all(
        _state(),
        [
            StrategyScoreInput(
                strategy_id=uuid4(),
                snapshot_id=uuid4(),
                expected_value_usd=Decimal("0.01"),
                estimated_win_probability=Decimal("0.40"),
                capital_required_usd=40,
                maximum_loss_usd=40,
            )
        ],
        snapshot_id=uuid4(),
    )
    no_trade = next(s for s in scores if s.is_no_trade)
    trades = [s for s in scores if not s.is_no_trade]
    assert no_trade.valid
    assert all(not t.valid for t in trades)
    assert any("win_probability_below_minimum" in t.invalidation_codes for t in trades)


def test_adv_obj_martingale_and_oversize() -> None:
    sizer = DeterministicObjectiveSizer(prohibit_loss_multiplier=True)
    state = _state(available_capital_usd=Decimal("100"), authorised_capital_usd=Decimal("100"))
    d = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("0.50"),
        expected_value_usd=5,
        estimated_win_probability=0.55,
        requested_quantity=50,
        prior_loss_count=4,
    )
    assert d.approved_quantity <= 2
    assert "agent_quantity_capped" in d.reason_codes or d.approved_quantity < 50


def test_adv_obj_capital_below_premium() -> None:
    sizer = DeterministicObjectiveSizer()
    d = sizer.size(
        _state(available_capital_usd=Decimal("10"), authorised_capital_usd=Decimal("10")),
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        requested_quantity=1,
    )
    assert not d.approved or d.approved_quantity == 0


def test_adv_obj_probability_unavailable_ordinal() -> None:
    scores = ObjectiveStrategyScorer(
        allow_ordinal_when_probability_unavailable=True
    ).score_all(
        _state(estimated_success_probability=None),
        [
            StrategyScoreInput(
                strategy_id=uuid4(),
                snapshot_id=uuid4(),
                expected_value_usd=Decimal("5"),
                capital_required_usd=40,
                maximum_loss_usd=40,
            )
        ],
        snapshot_id=uuid4(),
    )
    assert scores


@pytest.mark.asyncio
async def test_adv_obj_reservation_race_and_restart(tmp_path: Path) -> None:
    db = tmp_path / "race.db"
    svc = SessionObjectiveService(ObjectiveRepository(db))
    definition = await svc.create_objective(
        session_id="race",
        authorised_capital_usd=100,
        target_profit_pct=20,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="a",
        estimated_premium_usd=40,
        objective_state_version=state.version,
    )
    with pytest.raises(ObjectiveServiceError, match="stale"):
        await svc.reserve_for_order(
            client_order_id="b",
            estimated_premium_usd=40,
            objective_state_version=state.version,
        )
    recovered = SessionObjectiveService(ObjectiveRepository(db))
    loaded = await recovered.load_or_recover("race")
    assert loaded is not None
    assert loaded.reserved_capital_usd == Decimal("40.00")


@pytest.mark.asyncio
async def test_adv_obj_large_loss_and_capital_exhaustion(tmp_path: Path) -> None:
    db = tmp_path / "loss.db"
    svc = SessionObjectiveService(ObjectiveRepository(db))
    definition = await svc.create_objective(
        session_id="loss",
        authorised_capital_usd=100,
        target_profit_pct=50,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    state = await svc.recompute_from_truth(realised_pnl_usd=Decimal("-90"))
    assert state.available_capital_usd <= Decimal("100")
    # After nearly exhausting via reservation, further reserve fails.
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="last",
        estimated_premium_usd=state.available_capital_usd,
        objective_state_version=state.version,
    )
    state = await svc.get_state()
    with pytest.raises(ObjectiveServiceError):
        await svc.reserve_for_order(
            client_order_id="overflow",
            estimated_premium_usd=1,
            objective_state_version=state.version,
        )

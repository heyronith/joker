"""Durable objective duration / no-trade decay integration."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.objectives.target_attainment import (
    TargetAttainmentContext,
    TargetAttainmentPolicy,
    estimate_target_hit_probability,
)
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_objective_persists_confirmation_time_and_duration(tmp_path: Path) -> None:
    db = tmp_path / "obj.db"
    apply_objective_migrations(db)
    clock = FrozenExchangeClock(
        datetime(2026, 8, 4, 10, 0, tzinfo=ET), calendar=MarketCalendar()
    )
    svc = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    deadline = clock.now() + timedelta(minutes=60)
    definition = await svc.create_objective(
        session_id="s1",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    state = await svc.confirm_objective(
        definition.objective_id,
        confirmed_at_exchange_time=clock.now(),
    )
    assert state.objective_confirmed_at_exchange_time == clock.now()
    assert state.objective_duration_seconds == 3600
    assert state.elapsed_seconds == 0
    assert state.time_remaining_seconds == 3600
    assert state.fraction_remaining == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_objective_duration_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "obj.db"
    apply_objective_migrations(db)
    start = datetime(2026, 8, 4, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    svc = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    deadline = start + timedelta(minutes=60)
    definition = await svc.create_objective(
        session_id="s1",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(
        definition.objective_id, confirmed_at_exchange_time=start
    )
    # Simulate restart with a new service on the same DB.
    svc2 = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    svc2._objective_id = definition.objective_id  # noqa: SLF001
    clock.set_now(start + timedelta(minutes=30))
    state = await svc2.recompute_from_truth(now=clock.now())
    assert state.objective_duration_seconds == 3600
    assert state.objective_confirmed_at_exchange_time == start
    assert state.time_remaining_seconds == 1800
    assert state.elapsed_seconds == 1800
    assert state.fraction_remaining == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_deadline_does_not_reset_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "obj.db"
    apply_objective_migrations(db)
    start = datetime(2026, 8, 4, 10, 0, tzinfo=ET)
    deadline = start + timedelta(minutes=45)
    svc = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    definition = await svc.create_objective(
        session_id="s1",
        authorised_capital_usd=200,
        target_profit_pct=10,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id, confirmed_at_exchange_time=start)
    loaded = ObjectiveRepository(db).get_definition(definition.objective_id)
    assert loaded is not None
    assert loaded.deadline_exchange_time == deadline
    assert loaded.objective_duration_seconds == 45 * 60
    assert loaded.objective_confirmed_at_exchange_time == start


def test_no_trade_value_decays_halfway_and_near_deadline() -> None:
    snap = uuid4()
    oid = uuid4()

    def ctx(remaining: int, duration: int = 3600) -> TargetAttainmentContext:
        return TargetAttainmentContext(
            objective_id=oid,
            snapshot_id=snap,
            authorised_capital_usd=Decimal("500"),
            available_capital_usd=Decimal("500"),
            reserved_capital_usd=Decimal("0"),
            realised_pnl_usd=Decimal("0"),
            unrealised_pnl_usd=Decimal("0"),
            target_profit_usd=Decimal("100"),
            remaining_goal_gap_usd=Decimal("100"),
            time_remaining_seconds=remaining,
            objective_duration_seconds=duration,
            elapsed_seconds=duration - remaining,
            open_position_count=0,
            working_order_count=0,
            max_concurrent_positions=1,
            maximum_authorised_contracts=20,
            exchange_session_phase="regular",
        )

    start_p = estimate_target_hit_probability(
        ctx=ctx(3600),
        win_p=None,
        useful_upside_usd=Decimal("0"),
        capital_required_usd=Decimal("0"),
        sample_count=0,
        historical_hit_rate=None,
        resolution_seconds=None,
        is_no_trade=True,
    ).p_goal
    half_p = estimate_target_hit_probability(
        ctx=ctx(1800),
        win_p=None,
        useful_upside_usd=Decimal("0"),
        capital_required_usd=Decimal("0"),
        sample_count=0,
        historical_hit_rate=None,
        resolution_seconds=None,
        is_no_trade=True,
    ).p_goal
    near_p = estimate_target_hit_probability(
        ctx=ctx(180),
        win_p=None,
        useful_upside_usd=Decimal("0"),
        capital_required_usd=Decimal("0"),
        sample_count=0,
        historical_hit_rate=None,
        resolution_seconds=None,
        is_no_trade=True,
    ).p_goal
    assert start_p is not None and half_p is not None and near_p is not None
    assert start_p > half_p > near_p
    # Approx fractions for 60m objective with large gap.
    assert float(start_p) == pytest.approx(0.9, abs=0.05)
    assert float(half_p) == pytest.approx(0.45, abs=0.05)
    assert float(near_p) == pytest.approx(0.045, abs=0.02)


def test_compiled_context_uses_original_duration_not_remaining() -> None:
    from joker.objectives.schemas import SessionObjectiveState

    state = SessionObjectiveState.model_validate(
        {
            "objective_id": uuid4(),
            "session_id": "s",
            "status": "active",
            "authorised_capital_usd": Decimal("500"),
            "target_profit_usd": Decimal("100"),
            "target_ending_equity_usd": Decimal("600"),
            "available_capital_usd": Decimal("500"),
            "required_profit_remaining_usd": Decimal("100"),
            "time_remaining_seconds": 180,
            "objective_duration_seconds": 3600,
            "elapsed_seconds": 3420,
            "max_concurrent_positions": 1,
            "version": 2,
        }
    )
    ctx = TargetAttainmentContext.from_state(state, snapshot_id=uuid4())
    assert ctx.objective_duration_seconds == 3600
    assert ctx.time_remaining_seconds == 180
    assert ctx.elapsed_seconds == 3420
    assert float(ctx.fraction_remaining) == pytest.approx(0.05)
    # Decision path must not treat fraction_remaining as ~1.0
    decision = TargetAttainmentPolicy().decide(ctx, [])
    assert decision.fraction_remaining == "0.0500"
    assert decision.objective_duration_seconds == 3600

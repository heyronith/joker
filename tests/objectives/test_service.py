"""SessionObjectiveService reserve/release/restart tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from joker.objectives.repository import ObjectiveRepository
from joker.objectives.service import ObjectiveServiceError, SessionObjectiveService

ET = ZoneInfo("America/New_York")


def _deadline(hours: float = 2.0) -> datetime:
    return datetime.now(tz=ET) + timedelta(hours=hours)


async def _armed_service(tmp_path: Path) -> SessionObjectiveService:
    repo = ObjectiveRepository(tmp_path / "obj.db")
    svc = SessionObjectiveService(repo, exchange_tz="America/New_York")
    definition = await svc.create_objective(
        session_id="sess-1",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=_deadline(),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    return svc


@pytest.mark.asyncio
async def test_create_confirm_and_reserve_release(tmp_path: Path) -> None:
    svc = await _armed_service(tmp_path)
    state = await svc.get_state()
    assert state.status == "active"
    assert state.available_capital_usd == Decimal("500.00")

    res = await svc.reserve_for_order(
        client_order_id="c1",
        estimated_premium_usd=Decimal("50.00"),
        objective_state_version=state.version,
    )
    assert res.status == "open"
    # idempotent
    res2 = await svc.reserve_for_order(
        client_order_id="c1",
        estimated_premium_usd=Decimal("50.00"),
        objective_state_version=(await svc.get_state()).version,
    )
    assert res2.reservation_id == res.reservation_id

    after = await svc.get_state()
    assert after.reserved_capital_usd == Decimal("50.00")
    assert after.available_capital_usd == Decimal("450.00")

    released = await svc.release_for_order(client_order_id="c1", reason="cancelled")
    assert released.reserved_capital_usd == Decimal("0.00")
    assert released.available_capital_usd == Decimal("500.00")


@pytest.mark.asyncio
async def test_stale_version_fails_closed(tmp_path: Path) -> None:
    svc = await _armed_service(tmp_path)
    with pytest.raises(ObjectiveServiceError, match="stale"):
        await svc.reserve_for_order(
            client_order_id="c2",
            estimated_premium_usd=10,
            objective_state_version=0,
        )


@pytest.mark.asyncio
async def test_insufficient_capital_fails(tmp_path: Path) -> None:
    svc = await _armed_service(tmp_path)
    state = await svc.get_state()
    with pytest.raises(ObjectiveServiceError, match="insufficient"):
        await svc.reserve_for_order(
            client_order_id="c3",
            estimated_premium_usd=Decimal("600.00"),
            objective_state_version=state.version,
        )


@pytest.mark.asyncio
async def test_target_reached_pauses_entries(tmp_path: Path) -> None:
    svc = await _armed_service(tmp_path)
    await svc.record_verified_outcome(realised_pnl_delta_usd=Decimal("100.00"))
    state = await svc.get_state()
    assert state.status == "target_reached"
    assert state.entries_paused is True
    with pytest.raises(ObjectiveServiceError, match="paused|target"):
        await svc.reserve_for_order(
            client_order_id="c4",
            estimated_premium_usd=10,
            objective_state_version=state.version,
        )


@pytest.mark.asyncio
async def test_deadline_blocks_entries(tmp_path: Path) -> None:
    repo = ObjectiveRepository(tmp_path / "dl.db")
    svc = SessionObjectiveService(repo, exchange_tz="America/New_York")
    past = datetime.now(tz=ET) + timedelta(seconds=2)
    definition = await svc.create_objective(
        session_id="sess-dl",
        authorised_capital_usd=200,
        target_profit_pct=10,
        deadline_exchange_time=past,
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    await asyncio.sleep(2.1)
    state = await svc.recompute_from_truth()
    assert state.status == "deadline_reached"
    assert state.time_remaining_seconds == 0
    with pytest.raises(ObjectiveServiceError, match="deadline"):
        await svc.reserve_for_order(
            client_order_id="late",
            estimated_premium_usd=10,
            objective_state_version=state.version,
        )


@pytest.mark.asyncio
async def test_restart_reconstructs_without_double_reservation(tmp_path: Path) -> None:
    db = tmp_path / "restart.db"
    svc = await _armed_service(tmp_path)
    # use same db path via repo
    svc = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    definition = await svc.create_objective(
        session_id="sess-r",
        authorised_capital_usd=300,
        target_profit_pct=20,
        deadline_exchange_time=_deadline(),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="open-1",
        estimated_premium_usd=40,
        objective_state_version=state.version,
    )

    recovered = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    loaded = await recovered.load_or_recover("sess-r")
    assert loaded is not None
    assert loaded.reserved_capital_usd == Decimal("40.00")
    assert loaded.available_capital_usd == Decimal("260.00")
    # existing reservation remains idempotent
    again = await recovered.reserve_for_order(
        client_order_id="open-1",
        estimated_premium_usd=40,
        objective_state_version=loaded.version,
    )
    assert again.client_order_id == "open-1"
    final = await recovered.get_state()
    assert final.reserved_capital_usd == Decimal("40.00")

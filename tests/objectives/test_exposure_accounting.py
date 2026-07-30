"""Filled / working exposure accounting invariants."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from joker.objectives.repository import CrashInjected, ObjectiveRepository
from joker.objectives.service import ObjectiveServiceError, SessionObjectiveService

ET = ZoneInfo("America/New_York")


async def _armed(tmp_path: Path, capital: int = 500) -> SessionObjectiveService:
    db = tmp_path / "exp.db"
    svc = SessionObjectiveService(ObjectiveRepository(db))
    definition = await svc.create_objective(
        session_id="exp",
        authorised_capital_usd=capital,
        target_profit_pct=20,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
        max_concurrent_positions=2,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    return svc


@pytest.mark.asyncio
async def test_full_fill_keeps_filled_exposure(tmp_path: Path) -> None:
    """1. $500 authorised, $300 full fill → $200 available."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    after = await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=3,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    assert after.filled_position_exposure_usd == Decimal("300.00")
    assert after.working_order_reservation_usd == Decimal("0.00")
    assert after.available_capital_usd == Decimal("200.00")


@pytest.mark.asyncio
async def test_second_entry_rejected_when_insufficient(tmp_path: Path) -> None:
    """2. Second $250 entry rejected after $300 fill on $500."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=3,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    state = await svc.get_state()
    with pytest.raises(ObjectiveServiceError, match="insufficient"):
        await svc.reserve_for_order(
            client_order_id="o2",
            estimated_premium_usd=250,
            quantity=1,
            premium_per_contract_usd=Decimal("2.50"),
            objective_state_version=state.version,
        )


@pytest.mark.asyncio
async def test_partial_fill_encumbers_filled_plus_working(tmp_path: Path) -> None:
    """3. $200 partial fill + $100 unfilled → $300 encumbered."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    after = await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=2,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=1,
    )
    assert after.filled_position_exposure_usd == Decimal("200.00")
    assert after.working_order_reservation_usd == Decimal("100.00")
    assert after.reserved_capital_usd == Decimal("300.00")
    assert after.available_capital_usd == Decimal("200.00")


@pytest.mark.asyncio
async def test_cancel_after_partial_releases_only_working(tmp_path: Path) -> None:
    """4. Cancellation after partial fill releases only $100 working."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=2,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=1,
    )
    after = await svc.release_for_order(client_order_id="o1", reason="cancelled")
    assert after.filled_position_exposure_usd == Decimal("200.00")
    assert after.working_order_reservation_usd == Decimal("0.00")
    assert after.available_capital_usd == Decimal("300.00")


@pytest.mark.asyncio
async def test_two_positions_cannot_exceed_ceiling(tmp_path: Path) -> None:
    """5. Two positions can never exceed the $500 aggregate ceiling."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="a",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="a",
        fill_quantity=3,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="b",
        estimated_premium_usd=200,
        quantity=2,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="b",
        fill_quantity=2,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    state = await svc.get_state()
    assert state.filled_position_exposure_usd == Decimal("500.00")
    assert state.available_capital_usd == Decimal("0.00")
    with pytest.raises(ObjectiveServiceError):
        await svc.reserve_for_order(
            client_order_id="c",
            estimated_premium_usd=1,
            quantity=1,
            premium_per_contract_usd=Decimal("0.01"),
            objective_state_version=state.version,
        )


@pytest.mark.asyncio
async def test_restart_preserves_filled_exposure(tmp_path: Path) -> None:
    """6. Restart with one open filled position preserves exposure."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=3,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    recovered = SessionObjectiveService(ObjectiveRepository(tmp_path / "exp.db"))
    loaded = await recovered.load_or_recover("exp")
    assert loaded is not None
    assert loaded.filled_position_exposure_usd == Decimal("300.00")
    assert loaded.available_capital_usd == Decimal("200.00")


@pytest.mark.asyncio
async def test_restart_preserves_partial_fill(tmp_path: Path) -> None:
    """7. Restart with a partial fill preserves filled plus working exposure."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=2,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=1,
    )
    recovered = SessionObjectiveService(ObjectiveRepository(tmp_path / "exp.db"))
    loaded = await recovered.load_or_recover("exp")
    assert loaded is not None
    assert loaded.filled_position_exposure_usd == Decimal("200.00")
    assert loaded.working_order_reservation_usd == Decimal("100.00")


@pytest.mark.asyncio
async def test_position_reduction_releases_proportional_cost_basis(tmp_path: Path) -> None:
    """8. Position reduction releases proportional cost basis."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=3,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    after = await svc.reduce_position_exposure(
        client_order_id="o1",
        closed_quantity=1,
        realised_pnl_delta_usd=10,
        final_close=False,
    )
    assert after.filled_position_exposure_usd == Decimal("200.00")
    assert after.available_capital_usd == Decimal("300.00")
    assert after.realised_pnl_usd == Decimal("10.00")


@pytest.mark.asyncio
async def test_final_close_releases_all_exposure(tmp_path: Path) -> None:
    """9. Final close releases all remaining exposure."""
    svc = await _armed(tmp_path, 500)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="o1",
        estimated_premium_usd=300,
        quantity=3,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="o1",
        fill_quantity=3,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    after = await svc.reduce_position_exposure(
        client_order_id="o1",
        closed_quantity=3,
        realised_pnl_delta_usd=25,
        final_close=True,
        open_position_count=0,
    )
    assert after.filled_position_exposure_usd == Decimal("0.00")
    assert after.available_capital_usd == Decimal("500.00")
    assert after.open_position_count == 0


@pytest.mark.asyncio
async def test_crash_injection_no_double_reservation(tmp_path: Path) -> None:
    db = tmp_path / "crash.db"
    repo = ObjectiveRepository(db)
    svc = SessionObjectiveService(repo)
    definition = await svc.create_objective(
        session_id="crash",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=1),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)

    for point in (
        "before_transaction",
        "after_exposure_write",
        "after_state_append",
        "after_audit_append",
    ):
        state = await svc.get_state()
        fired = {"hit": False}

        def _hook(p, expected=point, box=fired):
            if p == expected and not box["hit"]:
                box["hit"] = True
                raise CrashInjected(expected)

        repo.set_crash_hook(_hook)
        with pytest.raises(CrashInjected):
            await svc.reserve_for_order(
                client_order_id=f"c-{point}",
                estimated_premium_usd=50,
                quantity=1,
                premium_per_contract_usd=Decimal("0.50"),
                objective_state_version=state.version,
            )
        repo.set_crash_hook(None)
        # After crash, capital must not exceed authorised and no orphan double-reserve.
        recovered = await svc.recompute_from_truth()
        assert recovered.available_capital_usd <= Decimal("500.00")
        assert recovered.reserved_capital_usd <= Decimal("500.00")
        # Retry succeeds idempotently / cleanly
        state = await svc.get_state()
        await svc.reserve_for_order(
            client_order_id=f"ok-{point}",
            estimated_premium_usd=40,
            quantity=1,
            premium_per_contract_usd=Decimal("0.40"),
            objective_state_version=state.version,
        )
        await svc.release_for_order(client_order_id=f"ok-{point}", reason="cleanup")

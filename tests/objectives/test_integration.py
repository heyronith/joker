"""Integration coverage for objective recovery, graph gates, and fill lifecycle."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.objectives.repository import ObjectiveRepository
from joker.objectives.service import ObjectiveServiceError, SessionObjectiveService
from joker.objectives.sizing import DeterministicObjectiveSizer
from joker.runtime.objective_recovery import recover_session_objective

ET = ZoneInfo("America/New_York")


@pytest.fixture
async def armed_service(tmp_path: Path) -> SessionObjectiveService:
    db = tmp_path / "obj.db"
    svc = SessionObjectiveService(ObjectiveRepository(db))
    definition = await svc.create_objective(
        session_id="sess-1",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    return svc


@pytest.mark.asyncio
async def test_recover_session_objective_preserves_reservation(
    armed_service: SessionObjectiveService, tmp_path: Path
) -> None:
    state = await armed_service.get_state()
    await armed_service.reserve_for_order(
        client_order_id="co-1",
        estimated_premium_usd=75,
        objective_state_version=state.version,
    )
    recovered = SessionObjectiveService(ObjectiveRepository(tmp_path / "obj.db"))
    loaded = await recover_session_objective(
        recovered,
        session_id="sess-1",
        execution_runtime=None,
        unresolved_reconciliation=False,
    )
    assert loaded is not None
    assert loaded.reserved_capital_usd == Decimal("75.00")
    assert loaded.available_capital_usd == Decimal("425.00")


@pytest.mark.asyncio
async def test_recover_marks_unresolved_reconciliation_blocks_reserve(
    armed_service: SessionObjectiveService,
) -> None:
    await recover_session_objective(
        armed_service,
        session_id="sess-1",
        unresolved_reconciliation=True,
    )
    state = await armed_service.get_state()
    with pytest.raises(ObjectiveServiceError, match="reconciliation"):
        await armed_service.reserve_for_order(
            client_order_id="blocked",
            estimated_premium_usd=10,
            objective_state_version=state.version,
        )


@pytest.mark.asyncio
async def test_fill_converts_reservation_and_close_updates_pnl(
    armed_service: SessionObjectiveService,
) -> None:
    state = await armed_service.get_state()
    await armed_service.reserve_for_order(
        client_order_id="fill-1",
        estimated_premium_usd=50,
        objective_state_version=state.version,
    )
    after_fill = await armed_service.record_verified_outcome(
        client_order_id="fill-1",
        convert_reservation=True,
        open_position_count=1,
    )
    assert after_fill.open_position_count == 1
    res = armed_service._repo.get_reservation_by_client_order("fill-1")  # noqa: SLF001
    assert res is not None
    assert res.status == "converted"
    after_close = await armed_service.record_verified_outcome(
        client_order_id="fill-1",
        realised_pnl_delta_usd=25,
        open_position_count=0,
    )
    assert after_close.realised_pnl_usd == Decimal("25.00")
    assert after_close.open_position_count == 0


@pytest.mark.asyncio
async def test_deadline_status_blocks_new_reserve(
    armed_service: SessionObjectiveService,
) -> None:
    await armed_service.recompute_from_truth(
        force_status="deadline_reached",
        now=datetime.now(tz=ET) + timedelta(hours=5),
    )
    state = await armed_service.get_state()
    assert state.status == "deadline_reached"
    with pytest.raises(ObjectiveServiceError, match="deadline"):
        await armed_service.reserve_for_order(
            client_order_id="late",
            estimated_premium_usd=10,
            objective_state_version=state.version,
        )


def test_sizer_clamps_oversize_request() -> None:
    from joker.objectives.schemas import SessionObjectiveState

    sizer = DeterministicObjectiveSizer(prohibit_loss_multiplier=True)
    state = SessionObjectiveState.model_validate(
        {
            "objective_id": uuid4(),
            "session_id": "s",
            "status": "active",
            "authorised_capital_usd": Decimal("100"),
            "target_profit_usd": Decimal("20"),
            "target_ending_equity_usd": Decimal("120"),
            "reserved_capital_usd": Decimal("0"),
            "available_capital_usd": Decimal("100"),
            "realised_pnl_usd": Decimal("0"),
            "unrealised_pnl_usd": Decimal("0"),
            "progress_to_goal_pct": Decimal("0"),
            "required_profit_remaining_usd": Decimal("20"),
            "time_remaining_seconds": 3600,
            "version": 1,
            "max_concurrent_positions": 1,
        }
    )
    decision = sizer.size(
        state,
        strategy_id=uuid4(),
        premium_per_contract_usd=Decimal("1.00"),
        requested_quantity=50,
        expected_value_usd=5,
        estimated_win_probability=0.55,
    )
    # $1 premium * 100 multiplier = $100/contract → at most 1 contract from $100 capital
    assert decision.approved_quantity <= 1

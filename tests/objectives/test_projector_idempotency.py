"""Task-1 objective projector idempotency and Task-2 independence."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.events.schemas import EventType, make_event
from joker.objectives.projector import ObjectiveCapitalProjector
from joker.objectives.repository import ObjectiveRepository
from joker.objectives.service import SessionObjectiveService

ET = ZoneInfo("America/New_York")


@pytest.fixture
async def svc(tmp_path: Path) -> SessionObjectiveService:
    service = SessionObjectiveService(ObjectiveRepository(tmp_path / "proj.db"))
    definition = await service.create_objective(
        session_id="proj",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
        max_concurrent_positions=2,
        accepted_total_loss_risk=True,
    )
    await service.confirm_objective(definition.objective_id)
    return service


@pytest.mark.asyncio
async def test_projector_updates_without_task2(svc: SessionObjectiveService) -> None:
    projector = ObjectiveCapitalProjector(svc)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="p1",
        estimated_premium_usd=100,
        quantity=1,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    event = make_event(
        EventType.ORDER_FILLED,
        session_id="proj",
        source="test",
        exchange_timestamp=datetime.now(tz=ET),
        payload={
            "client_order_id": "p1",
            "qty": 1,
            "price": "1.00",
            "remaining_quantity": 0,
            "ledger_event_id": str(uuid4()),
        },
    )
    await projector.handle_domain_event(event)
    after = await svc.get_state()
    assert after.filled_position_exposure_usd == Decimal("100.00")
    assert after.available_capital_usd == Decimal("400.00")


@pytest.mark.asyncio
async def test_duplicate_fill_does_not_double_convert(
    svc: SessionObjectiveService,
) -> None:
    projector = ObjectiveCapitalProjector(svc)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="dup",
        estimated_premium_usd=100,
        quantity=1,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    ledger_id = str(uuid4())
    event = make_event(
        EventType.ORDER_FILLED,
        session_id="proj",
        source="test",
        exchange_timestamp=datetime.now(tz=ET),
        payload={
            "client_order_id": "dup",
            "qty": 1,
            "price": "1.00",
            "remaining_quantity": 0,
            "ledger_event_id": ledger_id,
        },
    )
    await projector.handle_domain_event(event)
    await projector.handle_domain_event(event)
    after = await svc.get_state()
    assert after.filled_position_exposure_usd == Decimal("100.00")
    assert after.available_capital_usd == Decimal("400.00")


@pytest.mark.asyncio
async def test_duplicate_close_does_not_double_pnl(
    svc: SessionObjectiveService,
) -> None:
    projector = ObjectiveCapitalProjector(svc)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="close1",
        estimated_premium_usd=100,
        quantity=1,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
    )
    await svc.apply_verified_fill(
        client_order_id="close1",
        fill_quantity=1,
        fill_price=Decimal("1.00"),
        remaining_working_quantity=0,
    )
    close_id = str(uuid4())
    event = make_event(
        EventType.POSITION_CLOSED,
        session_id="proj",
        source="test",
        exchange_timestamp=datetime.now(tz=ET),
        payload={
            "client_order_id": "close1",
            "closed_quantity": 1,
            "realized_pnl": "20",
            "ledger_event_id": close_id,
            "open_position_count": 0,
        },
    )
    await projector.handle_domain_event(event)
    await projector.handle_domain_event(event)
    after = await svc.get_state()
    assert after.realised_pnl_usd == Decimal("20.00")
    assert after.filled_position_exposure_usd == Decimal("0.00")


@pytest.mark.asyncio
async def test_duplicate_cancel_does_not_release_twice(
    svc: SessionObjectiveService,
) -> None:
    projector = ObjectiveCapitalProjector(svc)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="cx",
        estimated_premium_usd=80,
        quantity=1,
        premium_per_contract_usd=Decimal("0.80"),
        objective_state_version=state.version,
    )
    cancel_id = str(uuid4())
    event = make_event(
        EventType.ORDER_CANCELLED,
        session_id="proj",
        source="test",
        exchange_timestamp=datetime.now(tz=ET),
        payload={"client_order_id": "cx", "ledger_event_id": cancel_id},
    )
    await projector.handle_domain_event(event)
    await projector.handle_domain_event(event)
    after = await svc.get_state()
    assert after.working_order_reservation_usd == Decimal("0.00")
    assert after.available_capital_usd == Decimal("500.00")


@pytest.mark.asyncio
async def test_projection_failure_marks_truth_degraded(
    svc: SessionObjectiveService,
) -> None:
    projector = ObjectiveCapitalProjector(svc)
    event = make_event(
        EventType.ORDER_FILLED,
        session_id="proj",
        source="test",
        exchange_timestamp=datetime.now(tz=ET),
        payload={
            "client_order_id": "missing",
            "qty": 1,
            "price": "1.00",
            "ledger_event_id": str(uuid4()),
        },
    )
    with pytest.raises(Exception):
        await projector.handle_domain_event(event)
    assert svc.truth_degraded
    state = await svc.get_state()
    with pytest.raises(Exception):
        await svc.reserve_for_order(
            client_order_id="blocked",
            estimated_premium_usd=10,
            quantity=1,
            premium_per_contract_usd=Decimal("0.10"),
            objective_state_version=state.version,
        )

"""Regression: order-manager decisions must mint distinct immutable artifact ids."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from joker.agents.cognitive.order_management import OrderManagerAgent
from joker.cognition.context import ContextAssembler
from joker.cognition.exceptions import ArtifactConflictError
from joker.cognition.schemas import AgentRole
from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot
from joker.models.fake_provider import FakeModelProvider
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from joker.persistence.cognitive_repositories import OrderManagementRepository
from tests.agents.cognitive.conftest import make_router
from tests.integration.task3_production_harness import install_order_manager_factory


def _context(*, cycle_id: str, snapshot_id, session_id: str = "om-identity"):
    snap = MarketSnapshot(
        snapshot_id=snapshot_id,
        exchange_time=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        trading_date=date(2026, 7, 1),
        underlying=UnderlyingSnapshot(
            symbol="SPY",
            exchange_time=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
            last=Decimal("500"),
        ),
        data_quality_id=uuid4(),
    )
    return ContextAssembler().assemble(
        agent_role=AgentRole.ORDER_MANAGER,
        session_id=session_id,
        cycle_id=cycle_id,
        snapshot=snap,
    )


@pytest.mark.asyncio
async def test_repeated_order_manager_invocations_mint_distinct_decision_ids(
    tmp_path,
) -> None:
    fake = FakeModelProvider(available=True)
    install_order_manager_factory(fake)
    router = make_router(fake)
    agent = OrderManagerAgent()
    repo = OrderManagementRepository(tmp_path / "om_artifacts.db")
    await repo.initialize()

    ctx_a = _context(cycle_id="cycle-a", snapshot_id=uuid4())
    ctx_b = _context(cycle_id="cycle-b", snapshot_id=uuid4())

    first = await agent.manage(ctx_a, router, client_order_id="order-a")
    second = await agent.manage(ctx_b, router, client_order_id="order-b")

    assert first.decision_id != second.decision_id
    await repo.append(first)
    await repo.append(second)
    loaded_a = await repo.get_by_id(first.decision_id)
    loaded_b = await repo.get_by_id(second.decision_id)
    assert loaded_a is not None
    assert loaded_b is not None
    assert loaded_a.decision_id == first.decision_id
    assert loaded_b.decision_id == second.decision_id

    # Exact repeated request must remain idempotent (same decision_id).
    again = await agent.manage(ctx_a, router, client_order_id="order-a")
    assert again.decision_id == first.decision_id
    assert len({c.request.idempotency_key for c in fake.calls if c.request.role == "order_manager"}) == 2

    with pytest.raises(ArtifactConflictError):
        conflict = first.model_copy(
            update={
                "rationale_summary": "mutated payload with same decision_id",
                "model_call_id": uuid4(),
            }
        )
        await repo.append(conflict)

    await drain_aiosqlite_workers(timeout=0.5)

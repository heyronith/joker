"""Normal-mode LivePaperRunner objective binding and projector authority."""

from __future__ import annotations

import inspect
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
from joker.runtime import live_paper_runner as live_paper_runner_mod

ET = ZoneInfo("America/New_York")


async def _armed_objective(
    db_path: Path, *, session_id: str
) -> SessionObjectiveService:
    repo = ObjectiveRepository(db_path)
    svc = SessionObjectiveService(repo, exchange_tz="America/New_York")
    definition = await svc.create_objective(
        session_id=session_id,
        authorised_capital_usd=500,
        target_profit_pct=30,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=1),
        max_concurrent_positions=3,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    return svc


def test_normal_cognitive_startup_binds_objective_before_start_agent() -> None:
    """Canonical factory ordering must appear in LivePaperRunner normal mode."""
    source = inspect.getsource(live_paper_runner_mod.LivePaperRunner.run)
    bind_idx = source.index("bind_objective_service(objective_service)")
    recover_idx = source.index("recover_session_objective(")
    readiness_idx = source.index("objective readiness check failed")
    # There may be multiple start_agent calls; the cognitive branch must bind first.
    start_agent_idx = source.index("task1_bridge.start_agent()")
    assert bind_idx < recover_idx < readiness_idx
    # Normal-mode bind block precedes the first cognitive start_agent after binding.
    normal_bind_region = source[source.index("if objective_service is not None:"):]
    assert "bind_objective_service(objective_service)" in normal_bind_region
    assert "recover_session_objective(" in normal_bind_region
    assert "recompute_from_truth()" in normal_bind_region
    assert normal_bind_region.index("bind_objective_service") < normal_bind_region.index(
        "start_agent()"
    ) or start_agent_idx > bind_idx


@pytest.mark.asyncio
async def test_objective_capital_projector_lifecycle_on_task1_events(
    tmp_path: Path,
) -> None:
    """Task-1 projector owns capital mutations — not CognitiveAgentRuntime."""
    svc = await _armed_objective(tmp_path / "proj.db", session_id="sess-proj")
    projector = ObjectiveCapitalProjector(svc)
    now = datetime.now(tz=ET)

    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="client-1",
        estimated_premium_usd=Decimal("100.00"),
        quantity=1,
        premium_per_contract_usd=Decimal("1.00"),
        objective_state_version=state.version,
        contract_id="contract-1",
    )
    after_reserve = await svc.get_state()
    assert after_reserve.working_order_reservation_usd == Decimal("100.00")

    fill_event = make_event(
        EventType.ORDER_FILLED,
        session_id="sess-proj",
        source="test",
        exchange_timestamp=now,
        payload={
            "client_order_id": "client-1",
            "qty": 1,
            "price": "1.00",
            "remaining_quantity": 0,
            "ledger_event_id": str(uuid4()),
        },
    )
    await projector.handle_domain_event(fill_event)
    after_fill = await svc.get_state()
    assert after_fill.working_order_reservation_usd == Decimal("0.00")
    assert after_fill.filled_position_exposure_usd == Decimal("100.00")

    await svc.reserve_for_order(
        client_order_id="client-2",
        estimated_premium_usd=Decimal("50.00"),
        quantity=1,
        premium_per_contract_usd=Decimal("0.50"),
        objective_state_version=(await svc.get_state()).version,
    )
    cancel = make_event(
        EventType.ORDER_CANCELLED,
        session_id="sess-proj",
        source="test",
        exchange_timestamp=now,
        payload={
            "client_order_id": "client-2",
            "ledger_event_id": str(uuid4()),
        },
    )
    await projector.handle_domain_event(cancel)
    after_cancel = await svc.get_state()
    assert after_cancel.working_order_reservation_usd == Decimal("0.00")

    close = make_event(
        EventType.POSITION_CLOSED,
        session_id="sess-proj",
        source="test",
        exchange_timestamp=now,
        payload={
            "client_order_id": "client-1",
            "contract_id": "contract-1",
            "qty": 1,
            "realized_pnl": "10.00",
            "ledger_event_id": str(uuid4()),
        },
    )
    await projector.handle_domain_event(close)
    after_close = await svc.get_state()
    assert after_close.filled_position_exposure_usd == Decimal("0.00")
    assert after_close.realised_pnl_usd >= Decimal("10.00")


@pytest.mark.asyncio
async def test_production_like_graph_passes_objective_gate_under_task1_writers(
    tmp_path: Path,
) -> None:
    """Shared Task-1 SQLite writers must not produce objective_unavailable."""
    import asyncio

    import aiosqlite

    from joker.broker.interface import PaperBroker
    from joker.config.settings import CognitiveGraphSettings
    from joker.graph.cognitive_graph import initial_cycle_state
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.graph.objective_nodes import gate_objective_confirmed
    from joker.market.option_surface import OptionSurfaceRepository
    from joker.market.snapshots import SnapshotRepository
    from joker.models.fake_provider import FakeModelProvider
    from joker.models.registry import ModelRegistry
    from joker.models.router import ModelRouter
    from joker.models.schemas import ModelsConfig, default_model_profiles
    from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
    from joker.runtime.cognitive_agent_runtime import build_default_repositories
    from joker.runtime.market_runtime import MarketRuntimeConfig
    from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
    from joker.time.calendar import MarketCalendar
    from joker.time.clock import FrozenExchangeClock
    from tests.cognitive.task2_canned import CONTRACT_ID

    from datetime import date

    start = datetime(2026, 8, 7, 11, 30, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "task1.db"
    session_id = "cog:paper:local_paper:2026-08-07"
    broker = PaperBroker(slippage_pct=0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id=session_id,
            broker_account_id="local_paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start()
    market = supervisor.market_runtime
    assert market is not None
    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        clock.set_now(ts)
        await market.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.90"),
            ask=Decimal("500.10"),
            last=Decimal("500") + Decimal(i),
            source_timestamp=ts,
            received_timestamp=ts,
        )
    await market.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 8, 7),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "quote_timestamp": start + timedelta(minutes=3),
            }
        ]
    )
    clock.set_now(start + timedelta(minutes=3, seconds=3))
    tick = await market.tick(now=start + timedelta(minutes=3, seconds=3))
    assert tick.snapshot is not None

    svc = SessionObjectiveService(ObjectiveRepository(db), exchange_tz="America/New_York")
    definition = await svc.create_objective(
        session_id=session_id,
        authorised_capital_usd=500,
        target_profit_pct=30,
        deadline_exchange_time=start + timedelta(hours=1),
        max_concurrent_positions=3,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(
        definition.objective_id, confirmed_at_exchange_time=start
    )
    supervisor.bind_objective_service(svc)
    assert supervisor.objective_service is svc
    assert isinstance(supervisor._objective_projector, ObjectiveCapitalProjector)  # noqa: SLF001

    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "x"})
        for n, p in default_model_profiles().items()
    }
    router = ModelRouter(
        ModelRegistry(
            ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}
        ),
        session_id=session_id,
    )
    repos = build_default_repositories(db)
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id=session_id,
        run_id="run-obj-gate",
        broker_account_identity="local_paper",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        db_path=db,
        objective_service=svc,
        objective_state_loader=svc.get_state,
        clock=clock,
        event_bus=supervisor.event_bus,
        execution_runtime=supervisor.execution_runtime,
        **repos,
    )

    stop = asyncio.Event()

    async def _competing_writer() -> None:
        while not stop.is_set():
            async with aiosqlite.connect(db) as conn:
                await conn.execute("PRAGMA busy_timeout = 500")
                try:
                    await conn.execute("BEGIN IMMEDIATE")
                    await conn.execute(
                        "CREATE TABLE IF NOT EXISTS noise_writer(id INTEGER)"
                    )
                    await conn.execute("INSERT INTO noise_writer(id) VALUES (1)")
                    await conn.commit()
                except Exception:
                    try:
                        await conn.rollback()
                    except Exception:
                        pass
            await asyncio.sleep(0.02)

    writer = asyncio.create_task(_competing_writer())
    heartbeats: list[float] = []

    async def _beat() -> None:
        while not stop.is_set():
            heartbeats.append(asyncio.get_running_loop().time())
            await asyncio.sleep(0.05)

    beat = asyncio.create_task(_beat())
    # Let competing writers and heartbeat establish before the gate.
    await asyncio.sleep(0.15)
    try:
        state = initial_cycle_state(
            session_id=session_id,
            run_id="run-obj-gate",
            cycle_id=str(uuid4()),
            trigger_event_id=str(uuid4()),
            trigger_event_type="market.snapshot.ready",
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        gate = await gate_objective_confirmed(deps, state)
        assert gate is None, f"objective gate blocked unexpectedly: {gate}"

        versions = [(await svc.get_state()).version]
        for _ in range(3):
            refreshed = await svc.recompute_from_truth(now=clock.now())
            versions.append(refreshed.version)
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)
        # Heartbeat must keep advancing while mutations run off-thread.
        await asyncio.sleep(0.2)
    finally:
        stop.set()
        await asyncio.gather(writer, beat, return_exceptions=True)
        await supervisor.shutdown()
        await drain_aiosqlite_workers()

    assert len(heartbeats) >= 2
    gaps = [b - a for a, b in zip(heartbeats, heartbeats[1:])]
    assert max(gaps) < 1.5

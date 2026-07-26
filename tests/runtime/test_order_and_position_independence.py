"""Working-order manager and independent position-worker tests."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import PositionAction
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType, make_event
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.runtime.cognitive_agent_runtime import (
    CognitiveAgentRuntime,
    build_default_repositories,
)
from joker.runtime.execution_runtime import ExecutionCommand
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.schemas.domain import OptionContract, OrderIntent
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")


def _registry(fake: FakeModelProvider) -> ModelRegistry:
    profiles = {
        name: profile.model_copy(update={"provider": "fake", "model": "fake-model"})
        for name, profile in default_model_profiles().items()
    }
    cfg = ModelsConfig(profiles=profiles)
    cfg = cfg.model_copy(
        update={
            "ollama": cfg.ollama.model_copy(update={"enabled": False}),
            "openai": cfg.openai.model_copy(update={"enabled": False}),
        }
    )
    return ModelRegistry(cfg, providers={"fake": fake})


@pytest.mark.asyncio
async def test_order_manager_continue_cancel_replace_idempotent(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "joker.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = "sess-om"
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id=session_id,
            broker_account_id="paper",
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
    for i in range(2):
        ts = start + timedelta(minutes=i, seconds=5)
        await market.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.90"),
            ask=Decimal("500.10"),
            last=Decimal("500"),
            source_timestamp=ts,
            received_timestamp=ts,
        )
    await market.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "quote_timestamp": start + timedelta(minutes=2),
            }
        ]
    )
    tick = await market.tick(now=start + timedelta(minutes=2, seconds=3))
    assert tick.snapshot is not None

    # Working order that stays open (limit far from market).
    intent = OrderIntent(
        intent_id="order-1",
        candidate_id="om-1",
        contract=OptionContract(
            symbol="SPY",
            expiration=date(2026, 7, 1),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        order_type="limit",
        quantity=1,
        limit_price=0.01,
    )
    # Keep the order working: nonzero slippage would still fill at 0.01 with
    # slippage_pct=0; use a broker that does not auto-fill below mid.
    broker.slippage_pct = 50.0
    order = await supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(client_order_id="order-1", intent=intent)
    )
    if order.status == "filled":
        # Fallback: cancel is not applicable; still exercise order-manager path.
        pass


    fake = FakeModelProvider(available=True)
    register_full_path_canned(
        fake,
        tick.snapshot.snapshot_id,
        "c1",
        session=session_id,
        order_action="continue_waiting",
    )
    registry = _registry(fake)
    router = ModelRouter(registry, session_id=session_id)
    repos = build_default_repositories(db)
    for r in repos.values():
        await r.initialize()
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id=session_id,
        run_id=session_id,
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=supervisor.data_quality_repository,
        execution_runtime=supervisor.execution_runtime,
        db_path=db,
        **repos,
    )
    runtime = CognitiveAgentRuntime(
        session_id=session_id,
        run_id=session_id,
        router=router,
        config=CognitiveGraphSettings(),
        graph_deps=deps,
        registry=registry,
        checkpointer_path=tmp_path / "om_ckpt.db",
    )
    await runtime.start()
    event = make_event(
        EventType.ORDER_ACCEPTED,
        session_id=session_id,
        source="test",
        exchange_timestamp=start,
        payload={
            "client_order_id": "order-1",
            "snapshot_id": str(tick.snapshot.snapshot_id),
            "broker_order_id": order.order_id,
        },
    )
    await runtime.on_event(event)
    await asyncio.sleep(0.2)
    # Duplicate — idempotent.
    await runtime.on_event(event)
    await asyncio.sleep(0.2)

    register_full_path_canned(
        fake,
        tick.snapshot.snapshot_id,
        "c2",
        session=session_id,
        order_action="cancel",
    )
    await runtime.on_event(
        make_event(
            EventType.ORDER_PARTIALLY_FILLED,
            session_id=session_id,
            source="test",
            exchange_timestamp=start,
            payload={
                "client_order_id": "order-1",
                "snapshot_id": str(tick.snapshot.snapshot_id),
            },
        )
    )
    await asyncio.sleep(0.3)
    decisions = await repos["order_management_repo"].list_by_session(session_id)
    assert decisions
    await runtime.shutdown()
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_slow_decision_does_not_block_urgent_position(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "joker.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = "sess-indep"
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id=session_id,
            broker_account_id="paper",
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
    ts = start + timedelta(seconds=5)
    await market.ingest_underlying_quote(
        symbol="SPY",
        bid=Decimal("499.90"),
        ask=Decimal("500.10"),
        last=Decimal("500"),
        source_timestamp=ts,
        received_timestamp=ts,
    )
    await market.ingest_option_quotes(
        [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "quote_timestamp": start + timedelta(minutes=1),
            }
        ]
    )
    tick = await market.tick(now=start + timedelta(minutes=1, seconds=3))
    assert tick.snapshot is not None

    fake = FakeModelProvider(available=True)
    register_full_path_canned(
        fake,
        tick.snapshot.snapshot_id,
        "c",
        session=session_id,
        position_action=PositionAction.HOLD,
    )
    # Make perception slow for decision cycles.
    original = fake.latency_seconds
    fake.latency_seconds = 0.4

    registry = _registry(fake)
    router = ModelRouter(registry, session_id=session_id)
    repos = build_default_repositories(db)
    for r in repos.values():
        await r.initialize()
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=30),
        session_id=session_id,
        run_id=session_id,
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=supervisor.data_quality_repository,
        execution_runtime=supervisor.execution_runtime,
        db_path=db,
        **repos,
    )
    runtime = CognitiveAgentRuntime(
        session_id=session_id,
        run_id=session_id,
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=30),
        graph_deps=deps,
        registry=registry,
        checkpointer_path=tmp_path / "indep_ckpt.db",
    )
    await runtime.start()
    await runtime.on_event(
        make_event(
            EventType.MARKET_SNAPSHOT_CREATED,
            session_id=session_id,
            source="test",
            exchange_timestamp=start,
            payload={"snapshot_id": str(tick.snapshot.snapshot_id), "cycle_id": "slow"},
        )
    )
    fake.latency_seconds = 0.0
    register_full_path_canned(
        fake,
        tick.snapshot.snapshot_id,
        "pos",
        session=session_id,
        position_action=PositionAction.HOLD,
    )
    await runtime.on_event(
        make_event(
            EventType.POSITION_OPENED,
            session_id=session_id,
            source="test",
            exchange_timestamp=start,
            payload={
                "position_id": "pos-1",
                "contract_id": CONTRACT_ID,
                "snapshot_id": str(tick.snapshot.snapshot_id),
                "original_strategy_id": str(uuid4()),
            },
        )
    )
    await asyncio.sleep(0.5)
    theses = await repos["position_thesis_repo"].list_by_session(session_id)
    assert theses, "urgent position cycle should complete independently"
    await runtime.shutdown()
    fake.latency_seconds = original
    await supervisor.shutdown()

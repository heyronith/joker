"""Task 2 active-path integration: snapshot → execute → position EXIT → provenance."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import MetaDecisionAction, PositionAction
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType, make_event
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.context_hydrate import context_assembler_from_settings
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
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
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")


def _fake_registry(fake: FakeModelProvider) -> ModelRegistry:
    profiles = {
        name: profile.model_copy(update={"provider": "fake", "model": "fake-model"})
        for name, profile in default_model_profiles().items()
    }
    models_config = ModelsConfig(profiles=profiles)
    models_config = models_config.model_copy(
        update={
            "ollama": models_config.ollama.model_copy(update={"enabled": False}),
            "openai": models_config.openai.model_copy(update={"enabled": False}),
        }
    )
    return ModelRegistry(models_config, providers={"fake": fake})


async def _seed_market(market, start: datetime, clock: FrozenExchangeClock | None = None) -> object:
    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        if clock is not None:
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
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.00",
                "ask": "1.20",
                "last": "1.10",
                "quote_timestamp": start + timedelta(minutes=3),
            }
        ]
    )
    now = start + timedelta(minutes=3, seconds=3)
    if clock is not None:
        clock.set_now(now)
    tick = await market.tick(now=now)
    assert tick.snapshot is not None
    assert tick.snapshot.option_surface_id is not None
    return tick.snapshot


@pytest.mark.asyncio
async def test_task2_active_path_full_session(tmp_path) -> None:
    async def _run() -> None:
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        clock = FrozenExchangeClock(start, calendar=MarketCalendar())
        db = tmp_path / "joker.db"
        broker = PaperBroker(slippage_pct=0)
        session_id = "sess-t2-active"
        cycle_id = "cycle-entry"

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
        assert supervisor.market_runtime is not None
        assert supervisor.execution_runtime is not None
        snapshot = await _seed_market(supervisor.market_runtime, start, clock)
        snapshot_id = snapshot.snapshot_id

        fake = FakeModelProvider(available=True)
        strategy_id = register_full_path_canned(
            fake, snapshot_id, cycle_id, session=session_id
        )
        registry = _fake_registry(fake)
        router = ModelRouter(registry, session_id=session_id)
        repos = build_default_repositories(db)
        for repo in repos.values():
            await repo.initialize()
        router.set_model_call_repo(repos["model_call_repo"])

        ckpt = CognitiveCheckpointer(tmp_path / "cog_ckpt.db")
        saver = await ckpt.open()

        submitted: list[str] = []

        async def submit_callback(provenanced) -> object:
            submitted.append(provenanced.command.client_order_id)
            return await supervisor.execution_runtime.submit_execution_command(
                provenanced.command
            )

        async def projection_loader():
            return await supervisor.execution_runtime.project_session()

        deps = CognitiveGraphDeps(
            router=router,
            config=CognitiveGraphSettings(),
            session_id=session_id,
            run_id=session_id,
            context_assembler=context_assembler_from_settings(CognitiveGraphSettings()),
            snapshot_repo=SnapshotRepository(db),
            option_surface_repo=OptionSurfaceRepository(db),
            data_quality_repo=supervisor.data_quality_repository,
            submit_callback=submit_callback,
            event_bus=supervisor.event_bus,
            execution_runtime=supervisor.execution_runtime,
            projection_loader=projection_loader,
            checkpointer=saver,
            db_path=db,
            **repos,
        )
        graph = build_cognitive_graph(deps)
        state = initial_cycle_state(
            session_id=session_id,
            run_id=session_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(snapshot_id),
        )
        config = ainvoke_config(
            session_id=session_id, graph_kind="decision", cycle_id=cycle_id
        )
        result = await graph.ainvoke(state, config=config)
        assert result.get("execution_command_id") is not None
        assert len(submitted) == 1
        assert result["meta_decision"].action == MetaDecisionAction.EXECUTE
        assert result.get("world_model") is not None
        assert "Deterministic synthesis" not in (
            result["world_model"].market_structure.structure_summary or ""
        )

        # Verified entry fill → ledger position.
        projected = await supervisor.execution_runtime.project_session()
        assert projected.orders or projected.positions

        # Position HOLD via cognitive runtime (not manual PendingExit).
        runtime = CognitiveAgentRuntime(
            session_id=session_id,
            run_id=session_id,
            router=router,
            config=CognitiveGraphSettings(),
            graph_deps=deps,
            registry=registry,
            checkpointer_path=tmp_path / "runtime_ckpt.db",
        )
        await runtime.start()
        await runtime.on_event(
            make_event(
                EventType.POSITION_OPENED,
                session_id=session_id,
                source="test",
                exchange_timestamp=start,
                payload={
                    "position_id": "pos-1",
                    "contract_id": CONTRACT_ID,
                    "snapshot_id": str(snapshot_id),
                    "original_strategy_id": str(strategy_id),
                },
            )
        )
        await asyncio.sleep(0.3)

        # Slow new-entry cycle begins, then urgent EXIT position event wins independently.
        slow_started = asyncio.Event()
        slow_finished = asyncio.Event()

        async def _slow_entry() -> None:
            slow_started.set()
            await asyncio.sleep(0.5)
            await runtime.on_event(
                make_event(
                    EventType.MARKET_SNAPSHOT_CREATED,
                    session_id=session_id,
                    source="test",
                    exchange_timestamp=start,
                    payload={"snapshot_id": str(snapshot_id), "cycle_id": "slow-entry"},
                )
            )
            slow_finished.set()

        # Re-register EXIT action for position agents.
        register_full_path_canned(
            fake,
            snapshot_id,
            "cycle-exit",
            session=session_id,
            position_action=PositionAction.EXIT,
        )
        entry_task = asyncio.create_task(_slow_entry())
        await slow_started.wait()
        await runtime.on_event(
            make_event(
                EventType.POSITION_CHANGED,
                session_id=session_id,
                source="test",
                exchange_timestamp=start + timedelta(seconds=1),
                payload={
                    "position_id": "pos-1",
                    "contract_id": CONTRACT_ID,
                    "snapshot_id": str(snapshot_id),
                    "original_strategy_id": str(strategy_id),
                    "cycle_id": "urgent-exit",
                },
            )
        )
        # Position worker should complete without waiting for slow entry.
        await asyncio.sleep(0.4)
        await entry_task
        await runtime.shutdown()

        # Provenance queryable after shutdown.
        evidence = await repos["evidence_repo"].list_by_session(session_id)
        assert evidence
        worlds = await repos["world_model_repo"].list_by_session(session_id)
        assert worlds
        decisions = await repos["decision_repo"].list_meta_by_session(session_id)
        assert decisions
        theses = await repos["position_thesis_repo"].list_by_session(session_id)
        assert theses

        await ckpt.close()
        await supervisor.shutdown()

    await _run()


@pytest.mark.asyncio
async def test_crash_after_execution_submission_no_duplicate_order(tmp_path) -> None:
    async def _run() -> None:
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        clock = FrozenExchangeClock(start, calendar=MarketCalendar())
        db = tmp_path / "joker.db"
        broker = PaperBroker(slippage_pct=0)
        session_id = "sess-crash"
        cycle_id = "cycle-crash"

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
        snapshot = await _seed_market(supervisor.market_runtime, start, clock)
        fake = FakeModelProvider(available=True)
        register_full_path_canned(fake, snapshot.snapshot_id, cycle_id, session=session_id)
        registry = _fake_registry(fake)
        router = ModelRouter(registry, session_id=session_id)
        repos = build_default_repositories(db)
        for repo in repos.values():
            await repo.initialize()

        submitted: list[str] = []

        async def submit_callback(provenanced) -> object:
            submitted.append(provenanced.command.client_order_id)
            return await supervisor.execution_runtime.submit_execution_command(
                provenanced.command
            )

        ckpt = CognitiveCheckpointer(tmp_path / "crash_ckpt.db")
        saver = await ckpt.open()
        deps = CognitiveGraphDeps(
            router=router,
            config=CognitiveGraphSettings(),
            session_id=session_id,
            run_id=session_id,
            snapshot_repo=SnapshotRepository(db),
            option_surface_repo=OptionSurfaceRepository(db),
            data_quality_repo=supervisor.data_quality_repository,
            submit_callback=submit_callback,
            execution_runtime=supervisor.execution_runtime,
            checkpointer=saver,
            db_path=db,
            **repos,
        )
        graph = build_cognitive_graph(deps)
        state = initial_cycle_state(
            session_id=session_id,
            run_id=session_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(snapshot.snapshot_id),
        )
        config = ainvoke_config(
            session_id=session_id, graph_kind="decision", cycle_id=cycle_id
        )
        result = await graph.ainvoke(state, config=config)
        assert result.get("execution_command_id")
        assert len(submitted) == 1

        # Restart: same thread_id resume must not duplicate broker order.
        graph2 = build_cognitive_graph(deps)
        result2 = await graph2.ainvoke(None, config=config)
        assert result2.get("execution_command_id") == result.get("execution_command_id")
        assert len(submitted) == 1

        await ckpt.close()
        await supervisor.shutdown()

    await _run()

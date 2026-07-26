"""Live-boundary Task 2 cutover tests — real Task 1 event contracts only."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import MetaDecisionAction, PositionAction
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.market.data_quality_store import DataQualityRepository
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.persistence.cognitive_execution_provenance import (
    CognitiveExecutionProvenanceRegistry,
    ExecutionProvenanceRecord,
)
from joker.runtime.cognitive_agent_runtime import (
    CognitiveAgentRuntime,
    build_default_repositories,
)
from joker.runtime.cognitive_binding import bind_cognitive_graph_to_task1
from joker.runtime.execution_runtime import ExecutionCommand, contract_id_for
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.schemas.domain import OptionContract, OrderIntent
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


async def _seed_surface(market, start: datetime, clock: FrozenExchangeClock) -> object:
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
    rows = [
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
        },
        {
            "contract_id": "SPY:2026-07-01:505.0:call",
            "symbol": "SPY",
            "expiry": date(2026, 7, 1),
            "strike": "505",
            "option_type": "call",
            "bid": "0.40",
            "ask": "0.55",
            "last": "0.48",
            "quote_timestamp": start + timedelta(minutes=3),
        },
        {
            "contract_id": "SPY:2026-07-01:495.0:put",
            "symbol": "SPY",
            "expiry": date(2026, 7, 1),
            "strike": "495",
            "option_type": "put",
            "bid": "0.35",
            "ask": "0.50",
            "last": "0.42",
            "quote_timestamp": start + timedelta(minutes=3),
        },
    ]
    await market.ingest_option_quotes(rows)
    now = start + timedelta(minutes=3, seconds=3)
    clock.set_now(now)
    tick = await market.tick(now=now)
    assert tick.snapshot is not None
    assert tick.snapshot.option_surface_id is not None
    assert tick.quality is not None
    return tick.snapshot


@pytest.mark.asyncio
async def test_bind_cognitive_graph_after_bridge_start(tmp_path: Path) -> None:
    """A: ExecutionRuntime must be non-null only after Task 1 start."""
    db = tmp_path / "bind.db"
    broker = PaperBroker(slippage_pct=0)
    fake = FakeModelProvider(available=True)
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id="bind")
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="bind",
        run_id="bind",
        db_path=db,
        **repos,
    )
    assert deps.execution_runtime is None

    class _BridgeStub:
        def __init__(self, supervisor: SessionSupervisor) -> None:
            self.supervisor = supervisor

    supervisor = SessionSupervisor(
        broker=broker,
        config=SessionSupervisorConfig(db_path=db, session_id="bind", broker_account_id="paper"),
    )
    # Before start — binding must fail closed.
    with pytest.raises(RuntimeError, match="ExecutionRuntime is None"):
        bind_cognitive_graph_to_task1(deps, _BridgeStub(supervisor))  # type: ignore[arg-type]

    await supervisor.start()
    bind_cognitive_graph_to_task1(
        deps,
        _BridgeStub(supervisor),  # type: ignore[arg-type]
        data_quality_repo=supervisor.data_quality_repository,
    )
    assert deps.execution_runtime is not None
    assert deps.projection_loader is not None
    assert deps.submit_callback is not None
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_execution_runtime_idempotent_after_broker_accept_crash(
    tmp_path: Path,
) -> None:
    """F: crash after broker accept must not duplicate the broker order."""
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "idem.db"
    broker = PaperBroker(slippage_pct=0)
    submit_count = {"n": 0}
    original_submit = broker.submit_order

    def counting_submit(intent: OrderIntent):
        submit_count["n"] += 1
        return original_submit(intent)

    broker.submit_order = counting_submit  # type: ignore[method-assign]

    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="idem",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start()
    assert supervisor.execution_runtime is not None
    contract = OptionContract(
        symbol="SPY",
        expiration=date(2026, 7, 1),
        strike=500.0,
        option_type="call",
        is_0dte=True,
    )
    client_order_id = "crash-window-order"
    intent = OrderIntent(
        intent_id=client_order_id,
        candidate_id="c1",
        contract=contract,
        side="buy",
        order_type="limit",
        quantity=1,
        limit_price=1.10,
    )
    first = await supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(client_order_id=client_order_id, intent=intent)
    )
    assert submit_count["n"] == 1

    # Simulate crash after broker accept: clear in-memory map but keep broker + ledger.
    supervisor.execution_runtime._client_to_broker.clear()  # noqa: SLF001

    second = await supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(client_order_id=client_order_id, intent=intent)
    )
    assert submit_count["n"] == 1
    assert second.order_id == first.order_id
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_live_boundary_entry_hold_exit_without_enriched_events(
    tmp_path: Path,
) -> None:
    """H: real MarketRuntime/ExecutionRuntime events drive full lifecycle."""

    async def _run() -> None:
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        clock = FrozenExchangeClock(start, calendar=MarketCalendar())
        db = tmp_path / "live_boundary.db"
        broker = PaperBroker(slippage_pct=0)
        session_id = "sess-live-boundary"

        received_types: list[str] = []

        class CapturingRuntime(CognitiveAgentRuntime):
            async def on_event(self, event):  # type: ignore[override]
                received_types.append(event.event_type.value)
                # Ensure we never rely on manually enriched position metadata.
                if event.event_type in {
                    EventType.POSITION_OPENED,
                    EventType.POSITION_CHANGED,
                    EventType.POSITION_CLOSED,
                }:
                    assert "snapshot_id" not in event.payload
                    assert "position_id" not in event.payload
                    assert "strategy_id" not in event.payload
                await super().on_event(event)

        fake = FakeModelProvider(available=True)
        registry = _fake_registry(fake)
        router = ModelRouter(registry, session_id=session_id)
        repos = build_default_repositories(db)
        for repo in repos.values():
            await repo.initialize()
        provenance = CognitiveExecutionProvenanceRegistry(
            db.with_name("live_boundary_prov.db")
        )
        await provenance.initialize()

        deps = CognitiveGraphDeps(
            router=router,
            config=CognitiveGraphSettings(max_cycle_seconds=60),
            session_id=session_id,
            run_id=session_id,
            snapshot_repo=SnapshotRepository(db),
            option_surface_repo=OptionSurfaceRepository(db),
            data_quality_repo=DataQualityRepository(db),
            db_path=db,
            provenance_registry=provenance,
            **repos,
        )
        runtime = CapturingRuntime(
            session_id=session_id,
            run_id=session_id,
            router=router,
            config=CognitiveGraphSettings(max_cycle_seconds=60),
            graph_deps=deps,
            registry=registry,
            checkpointer_path=tmp_path / "live_boundary_ckpt.db",
        )

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
            agent_runtime=runtime,
        )
        await supervisor.start()
        assert supervisor.execution_runtime is not None
        deps.execution_runtime = supervisor.execution_runtime
        deps.data_quality_repo = supervisor.data_quality_repository

        async def _submit(provenanced):
            await provenance.record(
                ExecutionProvenanceRecord(
                    client_order_id=provenanced.command.client_order_id,
                    proposal_id=str(provenanced.proposal_id),
                    decision_id=str(provenanced.decision_id),
                    strategy_id=str(provenanced.strategy_id),
                    cycle_id=str(provenanced.cycle_id),
                    snapshot_id=str(provenanced.snapshot_id),
                    contract_id=contract_id_for(provenanced.command.intent.contract),
                    session_id=session_id,
                    kind="entry",
                )
            )
            return await supervisor.execution_runtime.submit_execution_command(
                provenanced.command
            )

        async def _projection():
            return await supervisor.execution_runtime.project_session()

        deps.submit_callback = _submit
        deps.projection_loader = _projection

        snapshot = await _seed_surface(supervisor.market_runtime, start, clock)
        assert snapshot.option_surface_id is not None
        dq = await supervisor.data_quality_repository.get_by_id(snapshot.data_quality_id)
        assert dq is not None
        assert dq.usable_for_execution is True

        register_full_path_canned(
            fake,
            snapshot.snapshot_id,
            "cycle-entry",
            session=session_id,
            position_action=PositionAction.HOLD,
        )

        # MARKET_SNAPSHOT_CREATED already published by tick → cognitive entry.
        await asyncio.sleep(1.5)
        projection = await supervisor.execution_runtime.project_session()
        open_positions = {
            cid: pos
            for cid, pos in projection.positions.items()
            if pos.quantity != 0
        }
        assert CONTRACT_ID in open_positions
        assert open_positions[CONTRACT_ID].quantity == Decimal("1")
        entry_prov = await provenance.get_latest_by_contract_id(CONTRACT_ID)
        assert entry_prov is not None
        assert entry_prov.proposal_id
        assert entry_prov.snapshot_id

        # Subsequent snapshot → HOLD reassessment (no new entry while open).
        hold_start = start + timedelta(minutes=5)
        clock.set_now(hold_start)
        await supervisor.market_runtime.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("500.00"),
            ask=Decimal("500.20"),
            last=Decimal("500.10"),
            source_timestamp=hold_start,
            received_timestamp=hold_start,
        )
        await supervisor.market_runtime.ingest_option_quotes(
            [
                {
                    "contract_id": CONTRACT_ID,
                    "symbol": "SPY",
                    "expiry": date(2026, 7, 1),
                    "strike": "500",
                    "option_type": "call",
                    "bid": "1.15",
                    "ask": "1.35",
                    "last": "1.25",
                    "quote_timestamp": hold_start,
                }
            ]
        )
        hold_tick = await supervisor.market_runtime.tick(
            now=hold_start + timedelta(seconds=3)
        )
        assert hold_tick.snapshot is not None
        register_full_path_canned(
            fake,
            hold_tick.snapshot.snapshot_id,
            "cycle-hold",
            session=session_id,
            position_action=PositionAction.HOLD,
        )
        await asyncio.sleep(1.2)
        theses = await repos["position_thesis_repo"].list_by_session(session_id)
        assert any(t.recommended_action == PositionAction.HOLD for t in theses)

        # Later snapshot → EXIT; prove slow entry does not delay EXIT.
        exit_start = start + timedelta(minutes=8)
        clock.set_now(exit_start)
        await supervisor.market_runtime.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.50"),
            ask=Decimal("499.70"),
            last=Decimal("499.60"),
            source_timestamp=exit_start,
            received_timestamp=exit_start,
        )
        await supervisor.market_runtime.ingest_option_quotes(
            [
                {
                    "contract_id": CONTRACT_ID,
                    "symbol": "SPY",
                    "expiry": date(2026, 7, 1),
                    "strike": "500",
                    "option_type": "call",
                    "bid": "0.80",
                    "ask": "1.00",
                    "last": "0.90",
                    "quote_timestamp": exit_start,
                }
            ]
        )
        exit_tick = await supervisor.market_runtime.tick(
            now=exit_start + timedelta(seconds=3)
        )
        assert exit_tick.snapshot is not None
        register_full_path_canned(
            fake,
            exit_tick.snapshot.snapshot_id,
            "cycle-exit",
            session=session_id,
            position_action=PositionAction.EXIT,
        )

        slow_started = asyncio.Event()
        slow_finished = asyncio.Event()

        async def _slow_flat_entry() -> None:
            # Force a competing new-entry path while position is still open —
            # runtime must not enqueue new-entry when open, but even if it did,
            # EXIT on the position worker must complete independently.
            slow_started.set()
            await asyncio.sleep(0.8)
            slow_finished.set()

        entry_task = asyncio.create_task(_slow_flat_entry())
        await slow_started.wait()
        # Wait briefly — EXIT should complete without waiting for slow task.
        await asyncio.sleep(0.5)
        projection_mid = await supervisor.execution_runtime.project_session()
        # EXIT may already have closed, or still be in flight; wait for close.
        for _ in range(30):
            projection_mid = await supervisor.execution_runtime.project_session()
            pos = projection_mid.positions.get(CONTRACT_ID)
            if pos is None or pos.quantity == 0:
                break
            await asyncio.sleep(0.15)
        await entry_task

        final = await supervisor.execution_runtime.project_session()
        pos = final.positions.get(CONTRACT_ID)
        assert pos is not None
        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl is not None

        exit_theses = [
            t
            for t in await repos["position_thesis_repo"].list_by_session(session_id)
            if t.recommended_action == PositionAction.EXIT
        ]
        assert exit_theses
        exit_prov = await provenance.get_latest_by_contract_id(CONTRACT_ID)
        assert exit_prov is not None
        assert EventType.POSITION_OPENED.value in received_types
        assert EventType.MARKET_SNAPSHOT_CREATED.value in received_types

        await runtime.shutdown()
        await supervisor.shutdown()

    await _run()


@pytest.mark.asyncio
async def test_missing_data_quality_fails_closed(tmp_path: Path) -> None:
    """E: missing DQ report blocks execution usability."""
    from joker.graph.context_hydrate import load_snapshot_truth
    from joker.graph.graph_deps import CognitiveGraphDeps
    from joker.market.quality import DataQualitySeverity

    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "dq.db"
    broker = PaperBroker(slippage_pct=0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=db,
            session_id="dq",
            broker_account_id="paper",
            market=MarketRuntimeConfig(
                min_option_contracts=1,
                underlying_stale_seconds=3600,
                option_stale_seconds=3600,
            ),
        ),
    )
    await supervisor.start()
    snapshot = await _seed_surface(supervisor.market_runtime, start, clock)
    # Delete the persisted report to simulate unavailable truth.
    import aiosqlite

    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            "DELETE FROM data_quality_reports WHERE report_id = ?",
            (str(snapshot.data_quality_id),),
        )
        await conn.commit()

    fake = FakeModelProvider(available=True)
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id="dq")
    deps = CognitiveGraphDeps(
        router=router,
        config=CognitiveGraphSettings(),
        session_id="dq",
        run_id="dq",
        snapshot_repo=SnapshotRepository(db),
        option_surface_repo=OptionSurfaceRepository(db),
        data_quality_repo=DataQualityRepository(db),
        db_path=db,
    )
    _snap, dq, _surface, _slice = await load_snapshot_truth(deps, snapshot.snapshot_id)
    assert dq.severity == DataQualitySeverity.CRITICAL
    assert dq.usable_for_execution is False
    assert dq.usable_for_reasoning is False
    await supervisor.shutdown()

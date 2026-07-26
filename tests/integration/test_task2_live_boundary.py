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
from joker.cognition.schemas import PositionAction
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import ainvoke_config
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
from joker.runtime.order_action_gateway import OrderActionGateway, ensure_order_action_gateway
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.schemas.domain import OptionContract, OrderIntent
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")
FAR_CONTRACT_ID = "SPY:2026-07-01:580.0:call"
EXPECTED_REALIZED_PNL = Decimal("10")  # exit 1.20 − entry 1.10 × 100


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
            "is_0dte": True,
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
            "is_0dte": True,
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
            "is_0dte": True,
        },
        {
            "contract_id": FAR_CONTRACT_ID,
            "symbol": "SPY",
            "expiry": date(2026, 7, 1),
            "strike": "580",
            "option_type": "call",
            "bid": "0.05",
            "ask": "0.15",
            "last": "0.10",
            "quote_timestamp": start + timedelta(minutes=3),
            "is_0dte": True,
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
    """A: ExecutionRuntime ready after Task 1 start; agent starts after bind."""
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
        run_id="bind-run",
        db_path=db,
        **repos,
    )
    assert deps.execution_runtime is None

    class _BridgeStub:
        def __init__(self, supervisor: SessionSupervisor) -> None:
            self.supervisor = supervisor

    runtime = CognitiveAgentRuntime(
        session_id="bind",
        run_id="bind-run",
        router=router,
        config=CognitiveGraphSettings(),
        graph_deps=deps,
        registry=registry,
        checkpointer_path=tmp_path / "bind_ckpt.db",
    )
    supervisor = SessionSupervisor(
        broker=broker,
        config=SessionSupervisorConfig(
            db_path=db, session_id="bind", run_id="bind-run", broker_account_id="paper"
        ),
        agent_runtime=runtime,
    )
    with pytest.raises(RuntimeError, match="ExecutionRuntime is None"):
        bind_cognitive_graph_to_task1(deps, _BridgeStub(supervisor))  # type: ignore[arg-type]

    # Phase 1: Task 1 truth without starting the cognitive agent.
    await supervisor.start(start_agent=False)
    assert supervisor.execution_runtime is not None
    assert runtime._started is False  # noqa: SLF001
    bind_cognitive_graph_to_task1(
        deps,
        _BridgeStub(supervisor),  # type: ignore[arg-type]
        data_quality_repo=supervisor.data_quality_repository,
    )
    assert deps.execution_runtime is not None
    assert deps.order_action_gateway is not None
    # Phase 2: agent start/resume only after bind.
    await supervisor.start_agent_runtime()
    assert runtime._started is True  # noqa: SLF001
    await runtime.shutdown()
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
    """H: real Task 1 events + slow decision graph; EXIT via authoritative gateway."""

    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "live_boundary.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = "sess-live-boundary"

    received_types: list[str] = []
    gateway_exit_ids: list[str] = []

    class CapturingRuntime(CognitiveAgentRuntime):
        async def on_event(self, event):  # type: ignore[override]
            received_types.append(event.event_type.value)
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
    deps.option_surface_repo = supervisor.option_surface_repository
    deps.snapshot_repo = supervisor.snapshot_repository

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
    ensure_order_action_gateway(deps)
    assert deps.order_action_gateway is not None

    original_gateway_submit = deps.order_action_gateway.submit

    async def _tracking_submit(request):
        result = await original_gateway_submit(request)
        if request.action.value == "exit" and result.submitted:
            gateway_exit_ids.append(result.client_order_id)
        return result

    deps.order_action_gateway.submit = _tracking_submit  # type: ignore[method-assign]

    snapshot = await _seed_surface(supervisor.market_runtime, start, clock)
    assert snapshot.option_surface_id is not None
    surface0 = await supervisor.option_surface_repository.get_by_id(
        snapshot.option_surface_id
    )
    assert surface0 is not None
    assert FAR_CONTRACT_ID in {c.contract_id for c in surface0.contracts}

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

    await asyncio.sleep(1.5)
    projection = await supervisor.execution_runtime.project_session()
    open_positions = {
        cid: pos for cid, pos in projection.positions.items() if pos.quantity != 0
    }
    assert CONTRACT_ID in open_positions
    assert open_positions[CONTRACT_ID].quantity == Decimal("1")
    entry_prov = await provenance.get_latest_by_contract_id(CONTRACT_ID)
    assert entry_prov is not None
    assert entry_prov.proposal_id
    assert entry_prov.snapshot_id
    assert entry_prov.kind == "entry"

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
                "is_0dte": True,
            },
            {
                "contract_id": FAR_CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "580",
                "option_type": "call",
                "bid": "0.04",
                "ask": "0.14",
                "last": "0.09",
                "quote_timestamp": hold_start,
                "is_0dte": True,
            },
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

    # Later market move while an actual slow new-entry decision graph is mid-flight.
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
                "is_0dte": True,
            },
            {
                "contract_id": FAR_CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "580",
                "option_type": "call",
                "bid": "0.03",
                "ask": "0.12",
                "last": "0.08",
                "quote_timestamp": exit_start,
                "is_0dte": True,
            },
        ]
    )

    slow_started = asyncio.Event()
    slow_finished = asyncio.Event()
    original_complete = fake.complete_structured

    async def _slow_complete_structured(*, request, output_type):
        # Delay a middle decision-graph role so EXIT can race ahead.
        if request.cycle_id == "slow-entry-graph" and request.role in {
            "options_microstructure",
            "world_model_synthesiser",
            "meta_decision",
        }:
            slow_started.set()
            await asyncio.sleep(1.5)
        return await original_complete(request=request, output_type=output_type)

    fake.complete_structured = _slow_complete_structured  # type: ignore[method-assign]

    register_full_path_canned(
        fake,
        hold_tick.snapshot.snapshot_id,
        "slow-entry-graph",
        session=session_id,
        position_action=PositionAction.HOLD,
    )
    from dataclasses import replace

    # Avoid sharing the session checkpointer with a cancellable background graph.
    slow_deps = replace(deps, checkpointer=None)
    slow_graph = build_cognitive_graph(slow_deps)
    slow_state = initial_cycle_state(
        session_id=session_id,
        run_id=session_id,
        cycle_id="slow-entry-graph",
        trigger_event_id=str(uuid4()),
        trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
        snapshot_id=str(hold_tick.snapshot.snapshot_id),
    )

    async def _run_slow_decision_graph() -> None:
        try:
            await slow_graph.ainvoke(slow_state)
        finally:
            slow_finished.set()

    exit_tick = None
    entry_task = asyncio.create_task(_run_slow_decision_graph())
    try:
        await asyncio.wait_for(slow_started.wait(), timeout=5.0)

        # Update only position roles so the in-flight slow decision graph keeps its canned IDs.
        from joker.cognition.schemas import PositionThesisVersion
        from uuid import uuid4 as _uuid4

        exit_mc = _uuid4()
        exit_thesis = PositionThesisVersion(
            position_id=CONTRACT_ID,
            contract_id=CONTRACT_ID,
            session_id=session_id,
            snapshot_id=hold_tick.snapshot.snapshot_id,
            original_strategy_id=_uuid4(),
            current_thesis="exit now",
            recommended_action=PositionAction.EXIT,
            recommended_quantity=1,
            recommended_limit_price=Decimal("1.20"),
            confidence=0.7,
            prompt_version="2.0.0",
            model_call_id=exit_mc,
        )
        fake.set_canned_for_role("position_thesis", exit_thesis)
        fake.set_canned_for_role(
            "position_decision",
            exit_thesis.model_copy(
                update={
                    "thesis_version_id": _uuid4(),
                    "recommended_action": PositionAction.EXIT,
                }
            ),
        )

        # Independent position EXIT snapshot while the slow decision graph is still running.
        exit_tick = await supervisor.market_runtime.tick(
            now=exit_start + timedelta(seconds=3)
        )
        assert exit_tick.snapshot is not None
        exit_thesis_bound = exit_thesis.model_copy(
            update={
                "snapshot_id": exit_tick.snapshot.snapshot_id,
                "thesis_version_id": _uuid4(),
            }
        )
        fake.set_canned_for_role("position_thesis", exit_thesis_bound)
        fake.set_canned_for_role(
            "position_decision",
            exit_thesis_bound.model_copy(
                update={
                    "thesis_version_id": _uuid4(),
                    "recommended_action": PositionAction.EXIT,
                }
            ),
        )

        for _ in range(40):
            projection_mid = await supervisor.execution_runtime.project_session()
            pos = projection_mid.positions.get(CONTRACT_ID)
            if pos is not None and pos.quantity == 0:
                break
            await asyncio.sleep(0.15)
        assert not slow_finished.is_set(), "EXIT must finish before the slow entry graph"

        final = await supervisor.execution_runtime.project_session()
        pos = final.positions.get(CONTRACT_ID)
        assert pos is not None
        assert pos.quantity == Decimal("0")
        assert pos.realized_pnl == EXPECTED_REALIZED_PNL
    finally:
        if not entry_task.done():
            entry_task.cancel()
            try:
                await entry_task
            except (asyncio.CancelledError, Exception):
                pass

    exit_theses = [
        t
        for t in await repos["position_thesis_repo"].list_by_session(session_id)
        if t.recommended_action == PositionAction.EXIT
    ]
    assert exit_theses
    assert gateway_exit_ids, "EXIT must pass through OrderActionGateway"

    debates = await repos["debate_repo"].list_by_session(session_id)
    assert debates, "position execution criticism must be persisted"

    exit_prov = await provenance.get_latest_by_contract_id(CONTRACT_ID)
    assert exit_prov is not None
    assert exit_prov.kind == "exit"
    assert exit_prov.snapshot_id
    assert exit_prov.decision_id

    assert EventType.POSITION_OPENED.value in received_types
    assert EventType.MARKET_SNAPSHOT_CREATED.value in received_types

    latest_surface = await supervisor.option_surface_repository.get_by_id(
        exit_tick.snapshot.option_surface_id  # type: ignore[union-attr]
    )
    assert latest_surface is not None
    assert FAR_CONTRACT_ID in {c.contract_id for c in latest_surface.contracts}

    await runtime.shutdown()
    await supervisor.shutdown()


@pytest.mark.asyncio
async def test_missing_data_quality_fails_closed(tmp_path: Path) -> None:
    """E: missing DQ report blocks execution usability."""
    from joker.graph.context_hydrate import load_snapshot_truth
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

"""Genuine paper-session active path: Task2 → gateway → fill → POSITION_CLOSED → episode."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import PositionAction
from joker.config.settings import CognitiveGraphSettings
from joker.evaluation.agentic_graph import EVALUATOR_ROLES
from joker.evolution.agent_schemas import EvaluatorAgentScores
from joker.evolution.config import EvolutionSettings
from joker.evolution.runtime import EvolutionRuntime
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
)
from joker.runtime.cognitive_agent_runtime import (
    CognitiveAgentRuntime,
    build_default_repositories,
)
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.order_action_gateway import ensure_order_action_gateway
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned
from tests.integration.task3_production_harness import (
    install_paper_path_factories,
    set_position_action,
)

ET = ZoneInfo("America/New_York")
FAR_CONTRACT_ID = "SPY:2026-07-01:580.0:call"
EXPECTED_REALIZED_PNL = Decimal("10")


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


def _register_evaluators(fake: FakeModelProvider) -> None:
    scores = EvaluatorAgentScores(
        thesis_quality=Decimal("0.7"),
        evidence_grounding_score=Decimal("0.7"),
        calibration_score=Decimal("0.7"),
        execution_quality=Decimal("0.7"),
        efficiency_score=Decimal("0.5"),
    )
    for role in EVALUATOR_ROLES:
        fake.set_canned_for_role(role, scores)


async def _seed_surface(market, start: datetime, clock: FrozenExchangeClock):
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
    return tick.snapshot


@pytest.mark.asyncio
async def test_task3_paper_session_active_path_auto_episode(tmp_path) -> None:
    """evolution.enabled → Task2 order → gateway → PaperBroker fill → auto episode."""
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "task3_active_paper.db"
    broker = PaperBroker(slippage_pct=0)
    session_id = "sess-t3-active-paper"
    gateway_entry_ids: list[str] = []
    gateway_exit_ids: list[str] = []

    fake = FakeModelProvider(available=True)
    registry = _fake_registry(fake)
    router = ModelRouter(registry, session_id=session_id)
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
    provenance = CognitiveExecutionProvenanceRegistry(
        db.with_name("task3_active_prov.db")
    )
    await provenance.initialize()
    _register_evaluators(fake)

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
    agent = CognitiveAgentRuntime(
        session_id=session_id,
        run_id=session_id,
        router=router,
        config=CognitiveGraphSettings(max_cycle_seconds=60),
        graph_deps=deps,
        registry=registry,
        checkpointer_path=tmp_path / "task3_active_ckpt.db",
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
        agent_runtime=agent,
    )
    await supervisor.start()
    assert supervisor.execution_runtime is not None
    deps.execution_runtime = supervisor.execution_runtime
    deps.data_quality_repo = supervisor.data_quality_repository
    deps.option_surface_repo = supervisor.option_surface_repository
    deps.snapshot_repo = supervisor.snapshot_repository
    deps.event_bus = supervisor.event_bus

    async def _submit(provenanced):
        return await supervisor.execution_runtime.submit_execution_command(
            provenanced.command
        )

    async def _projection():
        return await supervisor.execution_runtime.project_session()

    deps.submit_callback = _submit
    deps.projection_loader = _projection
    ensure_order_action_gateway(deps)
    assert deps.order_action_gateway is not None
    original_submit = deps.order_action_gateway.submit

    async def _tracking_submit(request):
        result = await original_submit(request)
        if result.submitted:
            if request.action.value in {"entry", "probe"}:
                gateway_entry_ids.append(result.client_order_id)
            if request.action.value == "exit":
                gateway_exit_ids.append(result.client_order_id)
        return result

    deps.order_action_gateway.submit = _tracking_submit  # type: ignore[method-assign]

    deps.order_action_gateway.submit = _tracking_submit  # type: ignore[method-assign]

    evolution = EvolutionRuntime(
        db_path=db,
        settings=EvolutionSettings(enabled=True),
        session_id=session_id,
        run_id=session_id,
        event_bus=supervisor.event_bus,
        execution_runtime=supervisor.execution_runtime,
        model_router=router,
        cognitive_graph_deps=deps,
    )
    await evolution.prepare()
    evolution.subscribe_events()
    agent.bind_evolution_runtime(evolution)
    await evolution.start_workers()
    await evolution.resume()
    assert evolution._started is True
    champ = await evolution.configuration_for_new_cycle()
    assert champ is not None

    snapshot = await _seed_surface(supervisor.market_runtime, start, clock)
    register_full_path_canned(
        fake,
        snapshot.snapshot_id,
        "cycle-entry",
        session=session_id,
        position_action=PositionAction.HOLD,
    )
    install_paper_path_factories(fake, session_id=session_id)
    set_position_action(fake, PositionAction.HOLD)
    _register_evaluators(fake)

    entries_before = len(gateway_entry_ids)
    for _ in range(80):
        projection = await supervisor.execution_runtime.project_session()
        pos = projection.positions.get(CONTRACT_ID)
        if pos is not None and pos.quantity != 0 and len(gateway_entry_ids) > entries_before:
            break
        await asyncio.sleep(0.15)
    else:
        await evolution.shutdown()
        await agent.shutdown()
        await supervisor.shutdown()
        pytest.fail("entry fill never opened a position via OrderActionGateway")

    assert gateway_entry_ids, "entry must pass through OrderActionGateway"

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
    set_position_action(fake, PositionAction.EXIT)
    _register_evaluators(fake)
    exit_tick = await supervisor.market_runtime.tick(
        now=exit_start + timedelta(seconds=3)
    )
    assert exit_tick.snapshot is not None
    set_position_action(fake, PositionAction.EXIT)
    exits_before = len(gateway_exit_ids)

    for i in range(80):
        projection = await supervisor.execution_runtime.project_session()
        pos = projection.positions.get(CONTRACT_ID)
        flat = pos is not None and pos.quantity == 0
        exited = len(gateway_exit_ids) > exits_before
        if exited and flat:
            break
        if i > 0 and i % 15 == 0:
            set_position_action(fake, PositionAction.EXIT)
            now = clock.now()
            tick = await supervisor.market_runtime.tick(now=now + timedelta(seconds=1))
            clock.set_now(now + timedelta(seconds=1))
            assert tick.snapshot is not None
        await asyncio.sleep(0.15)
    else:
        await evolution.shutdown()
        await agent.shutdown()
        await supervisor.shutdown()
        pytest.fail("EXIT never closed the position via OrderActionGateway")

    assert gateway_exit_ids, "EXIT must pass through OrderActionGateway"
    final = await supervisor.execution_runtime.project_session()
    assert final.positions[CONTRACT_ID].realized_pnl == EXPECTED_REALIZED_PNL

    await supervisor.event_bus.drain(timeout=10.0)
    episodes = []
    for _ in range(50):
        episodes = await evolution._repos["episodes"].list_by_session(session_id)
        closed = [e for e in episodes if e.action_class == "closed_trade"]
        if closed:
            episodes = closed
            break
        await asyncio.sleep(0.1)
    else:
        await evolution.shutdown()
        await agent.shutdown()
        await supervisor.shutdown()
        pytest.fail("EvolutionRuntime never auto-compiled a closed_trade episode")

    episode = episodes[0]
    assert episode.completed is True, episode.completeness_findings
    assert episode.realised_pnl == EXPECTED_REALIZED_PNL
    assert episode.contract_id == CONTRACT_ID
    assert episode.entry_order_ids
    assert episode.exit_order_ids
    assert episode.quantity == Decimal("1")
    assert episode.initial_snapshot_id is not None
    # Evaluation worker should consume the episode without a manual evaluate() call.
    for _ in range(40):
        evaluations = await evolution._repos["evaluations"].list_by_episode(
            episode.episode_id
        )
        if evaluations:
            break
        await asyncio.sleep(0.1)
    else:
        await evolution.shutdown()
        await agent.shutdown()
        await supervisor.shutdown()
        pytest.fail("automatic evaluation never persisted")

    await evolution.shutdown()
    await agent.shutdown()
    await supervisor.shutdown()
    from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers

    # Best-effort only: session-scoped loop may retain unrelated aiosqlite workers.
    await drain_aiosqlite_workers(timeout=0.5)

"""Crash/resume checkpoint matrix for cognitive decision and position graphs."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.runtime.cognitive_agent_runtime import build_default_repositories
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned

ET = ZoneInfo("America/New_York")

CHECKPOINT_NODES = (
    "hydrate_context",
    "perception",
    "synthesise_world_model",
    "discovery",
    "strategy",
    "debate",
    "meta_decision",
    "entry_tactician",
    "submit_execution_command",
)


@pytest.mark.asyncio
async def test_decision_graph_checkpoint_resume_after_submit(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "joker.db"
    session_id = "sess-ckpt"
    cycle_id = "cycle-ckpt"
    supervisor = SessionSupervisor(
        broker=PaperBroker(slippage_pct=0),
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
                "expiry": date(2026, 7, 1),
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

    fake = FakeModelProvider(available=True)
    register_full_path_canned(fake, tick.snapshot.snapshot_id, cycle_id, session=session_id)
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
    registry = ModelRegistry(cfg, providers={"fake": fake})
    router = ModelRouter(registry, session_id=session_id)
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
    router.set_model_call_repo(repos["model_call_repo"])

    submitted: list[str] = []

    async def submit_callback(provenanced):
        submitted.append(provenanced.command.client_order_id)
        return await supervisor.execution_runtime.submit_execution_command(
            provenanced.command
        )

    ckpt = CognitiveCheckpointer(tmp_path / "ckpt.db")
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
        snapshot_id=str(tick.snapshot.snapshot_id),
    )
    config = ainvoke_config(
        session_id=session_id, graph_kind="decision", cycle_id=cycle_id
    )
    result = await graph.ainvoke(state, config=config)
    assert result.get("execution_command_id")
    assert len(submitted) == 1
    model_calls_before = len(fake.calls)

    # Resume after crash: completed cycle must not re-submit or re-call models.
    graph2 = build_cognitive_graph(deps)
    result2 = await graph2.ainvoke(None, config=config)
    assert result2.get("execution_command_id") == result.get("execution_command_id")
    assert len(submitted) == 1
    assert len(fake.calls) == model_calls_before

    traces = [t.node_name for t in (result.get("node_trace") or [])]
    for node in CHECKPOINT_NODES:
        assert any(node in name for name in traces), f"missing node coverage for {node}"

    await ckpt.close()
    await supervisor.shutdown()

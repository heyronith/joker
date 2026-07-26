"""Task 2 integration replay with FakeModelProvider canned outputs."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cognition.schemas import MetaDecisionAction
from joker.config.settings import CognitiveGraphSettings
from joker.events.schemas import EventType
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
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


@pytest.mark.asyncio
async def test_task2_replay_perception_to_execute_provenance(tmp_path) -> None:
    async def _run() -> None:
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        clock = FrozenExchangeClock(start, calendar=MarketCalendar())
        db = tmp_path / "joker.db"
        broker = PaperBroker(slippage_pct=0)
        session_id = "sess-t2"
        cycle_id = "cycle-t2"

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

        market = supervisor.market_runtime
        for i in range(3):
            ts = start + timedelta(minutes=i, seconds=5)
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
        snapshot_id = tick.snapshot.snapshot_id

        fake = FakeModelProvider(available=True)
        register_full_path_canned(fake, snapshot_id, cycle_id, session=session_id)
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
        registry = ModelRegistry(models_config, providers={"fake": fake})
        router = ModelRouter(registry, session_id=session_id)
        submitted: list = []
        repos = build_default_repositories(db)
        for repo in repos.values():
            await repo.initialize()
        router.set_model_call_repo(repos["model_call_repo"])

        snapshot_repo = SnapshotRepository(db)
        await snapshot_repo.initialize()

        async def submit_callback(provenanced) -> object:
            submitted.append(provenanced.command)
            return await supervisor.execution_runtime.submit_execution_command(
                provenanced.command
            )

        deps = CognitiveGraphDeps(
            router=router,
            config=CognitiveGraphSettings(),
            session_id=session_id,
            run_id=session_id,
            snapshot_repo=snapshot_repo,
            option_surface_repo=OptionSurfaceRepository(db),
            submit_callback=submit_callback,
            event_bus=supervisor.event_bus,
            execution_runtime=supervisor.execution_runtime,
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
        result = await graph.ainvoke(state)
        assert result.get("execution_command_id") is not None
        assert len(submitted) == 1
        assert submitted[0].intent.contract.symbol == "SPY"
        assert result.get("meta_decision") is not None
        assert result["meta_decision"].action == MetaDecisionAction.EXECUTE
        assert result.get("world_model") is not None

        evidence = await repos["evidence_repo"].list_by_session(session_id)
        assert len(evidence) >= 5

        await supervisor.shutdown()

    await _run()

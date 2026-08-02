"""Production Task-3 repository wiring for historical EV — no private seeding."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.cli.session_confirm import build_objective_engines
from joker.config.settings import AppSettings, CognitiveGraphSettings
from joker.events.schemas import EventType
from joker.evolution.repositories import build_evolution_repositories
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.objectives.execution_quote import build_current_option_quote_loader
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from joker.runtime.cognitive_agent_runtime import build_default_repositories
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.order_action_gateway import OrderActionGateway
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned
from tests.objectives.historical_fixtures import (
    persist_compiler_produced_history,
    persist_positive_history,
)

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_live_objective_engines_load_persisted_task3_history(tmp_path) -> None:
    db = tmp_path / "prod.db"
    apply_objective_migrations(db)
    evo = build_evolution_repositories(db)
    for repo in evo.values():
        await repo.initialize()
    as_of = datetime(2026, 7, 1, 14, 0, tzinfo=ET)
    rows = await persist_positive_history(
        episode_repo=evo["episodes"],
        evaluation_repo=evo["evaluations"],
        as_of=as_of,
        n=20,
        pnl=Decimal("12.00"),
        strategy_family="breakout_continuation",
    )
    obj_repo = ObjectiveRepository(db)
    app = AppSettings()
    app = app.model_copy(
        update={
            "objective": app.objective.model_copy(
                update={
                    "historical_outcomes": app.objective.historical_outcomes.model_copy(
                        update={
                            "minimum_samples_for_ev": 20,
                            "minimum_effective_sample_size": 15,
                            "require_lower_confidence_bound_positive": True,
                            "require_same_strategy_family": True,
                            "minimum_similarity": 0.10,
                        }
                    )
                }
            )
        }
    )
    engines = build_objective_engines(
        app,
        episode_repository=evo["episodes"],
        evaluation_repository=evo["evaluations"],
        dataset_repository=evo["datasets"],
        objective_repository=obj_repo,
    )
    hist = engines.historical_outcome_service
    assert hist.uses_repository_loaders is True
    assert engines.source_diagnostic.cold_start is False
    assert not hasattr(hist, "_seeded") or hist._seeded == []

    summary = await hist.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=as_of,
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    assert summary.sample_count >= 20
    assert summary.valid_for_ev is True
    persisted_eps = {r[0].episode_id for r in rows}
    persisted_evals = {r[1].evaluation_id for r in rows}
    assert set(summary.comparable_episode_ids).issubset(persisted_eps)
    assert set(summary.evaluation_ids).issubset(persisted_evals)


@pytest.mark.asyncio
async def test_missing_configured_repositories_cold_starts(tmp_path) -> None:
    app = AppSettings()
    engines = build_objective_engines(app)
    assert engines.historical_outcome_service.uses_repository_loaders is False
    assert engines.source_diagnostic.cold_start is True
    assert engines.source_diagnostic.reason is not None
    summary = await engines.historical_outcome_service.summarize_for_strategy(
        objective_id=uuid4(),
        strategy_id=uuid4(),
        snapshot_id=uuid4(),
        as_of_timestamp=datetime.now(tz=ET),
        direction="bullish",
        strategy_family="breakout_continuation",
    )
    assert summary.sample_count == 0
    assert summary.valid_for_ev is False


@pytest.mark.asyncio
async def test_production_full_graph_positive_ev_reaches_paper_broker(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "prod_graph.db"
    session_id = "prod-sess"
    cycle_id = "prod-cycle"
    broker = PaperBroker(slippage_pct=0)
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
    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        clock.set_now(ts)
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
                "ask": "1.10",
                "quote_timestamp": start + timedelta(minutes=3),
            }
        ]
    )
    clock.set_now(start + timedelta(minutes=3, seconds=3))
    tick = await market.tick(now=start + timedelta(minutes=3, seconds=3))
    assert tick.snapshot is not None

    apply_objective_migrations(db)
    evo = build_evolution_repositories(db)
    for repo in evo.values():
        await repo.initialize()
    rows = await persist_compiler_produced_history(
        episode_repo=evo["episodes"],
        evaluation_repo=evo["evaluations"],
        as_of=start + timedelta(minutes=3),
        n=20,
        pnl=Decimal("16.00"),
        strategy_family="breakout_continuation",
    )
    obj_repo = ObjectiveRepository(db)
    objective_service = SessionObjectiveService(obj_repo, require_positive_expected_value=True)
    definition = await objective_service.create_objective(
        session_id=session_id,
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=4),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await objective_service.confirm_objective(definition.objective_id)

    app = AppSettings()
    app = app.model_copy(
        update={
            "objective": app.objective.model_copy(
                update={
                    "enabled": True,
                    "historical_outcomes": app.objective.historical_outcomes.model_copy(
                        update={
                            "minimum_samples_for_ev": 20,
                            "minimum_effective_sample_size": 15,
                            "require_lower_confidence_bound_positive": True,
                            "require_same_strategy_family": True,
                            "minimum_similarity": 0.10,
                        }
                    ),
                }
            )
        }
    )
    engines = build_objective_engines(
        app,
        episode_repository=evo["episodes"],
        evaluation_repository=evo["evaluations"],
        dataset_repository=evo["datasets"],
        objective_repository=objective_service.repository,
    )
    assert engines.historical_outcome_service.uses_repository_loaders

    fake = FakeModelProvider(available=True)
    register_full_path_canned(fake, tick.snapshot.snapshot_id, cycle_id, session=session_id)
    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "fake"})
        for n, p in default_model_profiles().items()
    }
    cfg = ModelsConfig(profiles=profiles)
    cfg = cfg.model_copy(
        update={
            "ollama": cfg.ollama.model_copy(update={"enabled": False}),
            "openai": cfg.openai.model_copy(update={"enabled": False}),
        }
    )
    router = ModelRouter(
        ModelRegistry(cfg, providers={"fake": fake}), session_id=session_id
    )
    repos = build_default_repositories(db)
    for repo in repos.values():
        await repo.initialize()
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
        execution_runtime=supervisor.execution_runtime,
        checkpointer=saver,
        db_path=db,
        clock=clock,
        objective_service=objective_service,
        objective_state_loader=objective_service.get_state,
        max_quote_age_seconds=3600,
        max_relative_spread=0.50,
        **engines.as_deps_kwargs(),
        **repos,
    )
    deps.current_option_quote_loader = build_current_option_quote_loader(
        deps, max_quote_age_seconds=3600, max_relative_spread=0.50
    )
    deps.order_action_gateway = OrderActionGateway(deps)
    submitted: list[str] = []
    original = deps.order_action_gateway.submit

    async def _track(request):
        result = await original(request)
        if result.submitted:
            submitted.append(result.client_order_id)
        return result

    deps.order_action_gateway.submit = _track  # type: ignore[method-assign]

    try:
        graph = build_cognitive_graph(deps)
        state = initial_cycle_state(
            session_id=session_id,
            run_id=session_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        result = await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        summaries = result.get("_historical_summaries") or []
        valid_summary = next((s for s in summaries if s.get("valid_for_ev")), None)
        assert valid_summary is not None
        assert valid_summary["sample_count"] >= 20
        estimates = result.get("_strategy_estimates") or []
        est = next((e for e in estimates if e.get("valid")), None)
        assert est is not None
        persisted_eps = {str(r[0].episode_id) for r in rows}
        persisted_evals = {str(r[1].evaluation_id) for r in rows}
        assert set(valid_summary["comparable_episode_ids"]).issubset(persisted_eps)
        assert set(valid_summary["evaluation_ids"]).issubset(persisted_evals)
        assert len(submitted) == 1
        assert result.get("execution_command_id")
    finally:
        await ckpt.close()
        await supervisor.shutdown()
        await drain_aiosqlite_workers(timeout=0.5)

"""Compiled cognitive graph proves factual historical EV can reach PaperBroker."""

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
from tests.objectives.historical_fixtures import persist_positive_history

ET = ZoneInfo("America/New_York")


async def _prepare_stack(
    tmp_path,
    *,
    pnl: Decimal,
    n: int = 20,
    kill_switch: bool = False,
    option_ask: str = "1.20",
    option_bid: str = "1.00",
    option_quotes: list[dict] | None = None,
    max_concurrent_positions: int = 1,
    maximum_authorised_contracts: int = 20,
    objective_duration: timedelta = timedelta(hours=4),
):
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    db = tmp_path / "joker.db"
    session_id = "sess-hist-ev"
    cycle_id = "cycle-hist-ev"
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
            last=Decimal("500") + Decimal(i),
            source_timestamp=ts,
            received_timestamp=ts,
        )
    await market.ingest_option_quotes(
        option_quotes
        or [
            {
                "contract_id": CONTRACT_ID,
                "symbol": "SPY",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": option_bid,
                "ask": option_ask,
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
    as_of = start + timedelta(minutes=3)
    await persist_positive_history(
        episode_repo=evo["episodes"],
        evaluation_repo=evo["evaluations"],
        as_of=as_of,
        n=n,
        pnl=pnl,
        strategy_family="breakout_continuation",
    )

    obj_repo = ObjectiveRepository(db)
    objective_service = SessionObjectiveService(
        obj_repo, require_positive_expected_value=True
    )
    deadline = start + objective_duration
    definition = await objective_service.create_objective(
        session_id=session_id,
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=deadline,
        max_concurrent_positions=max_concurrent_positions,
        accepted_total_loss_risk=True,
    )
    await objective_service.confirm_objective(
        definition.objective_id,
        confirmed_at_exchange_time=start,
    )

    app = AppSettings()
    app = app.model_copy(
        update={
            "objective": app.objective.model_copy(
                update={
                    "enabled": True,
                    "maximum_authorised_contracts": maximum_authorised_contracts,
                    "require_positive_expected_value": True,
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
        objective_repository=obj_repo,
    )
    assert engines.historical_outcome_service.uses_repository_loaders

    fake = FakeModelProvider(available=True)
    register_full_path_canned(
        fake, tick.snapshot.snapshot_id, cycle_id, session=session_id
    )
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

    ckpt = CognitiveCheckpointer(tmp_path / "ckpt.db")
    saver = await ckpt.open()

    async def _obj_loader():
        return await objective_service.get_state()

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
        objective_state_loader=_obj_loader,
        max_quote_age_seconds=3600,
        max_relative_spread=0.50,
        kill_switch=kill_switch,
        **engines.as_deps_kwargs(),
        **repos,
    )
    deps.current_option_quote_loader = build_current_option_quote_loader(
        deps, max_quote_age_seconds=3600, max_relative_spread=0.50
    )
    deps.order_action_gateway = OrderActionGateway(deps)
    deps.require_objective_dependencies()

    submitted: list[str] = []
    gateway = deps.order_action_gateway
    assert gateway is not None
    original_submit = gateway.submit

    async def _tracking_submit(request):
        result = await original_submit(request)
        if result.submitted:
            submitted.append(result.client_order_id)
        return result

    gateway.submit = _tracking_submit  # type: ignore[method-assign]

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
    return {
        "graph": graph,
        "state": state,
        "config": config,
        "submitted": submitted,
        "broker": broker,
        "objective_service": objective_service,
        "hist": engines.historical_outcome_service,
        "supervisor": supervisor,
        "snapshot_id": tick.snapshot.snapshot_id,
        "checkpointer": ckpt,
        "market": market,
        "clock": clock,
        "start": start,
        "gateway": gateway,
        "original_submit": original_submit,
        "engines": engines,
        "evo": evo,
        "deps": deps,
    }


async def _teardown_stack(stack: dict) -> None:
    await stack["checkpointer"].close()
    await stack["supervisor"].shutdown()
    await drain_aiosqlite_workers(timeout=0.5)


def _valid_estimate(result: dict):
    estimates = result.get("_strategy_estimates") or []
    return next((e for e in estimates if e.get("valid")), None)


@pytest.mark.asyncio
async def test_full_compiled_graph_positive_ev_reaches_paper_execution(tmp_path) -> None:
    stack = await _prepare_stack(tmp_path, pnl=Decimal("15.00"), n=20)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        est = _valid_estimate(result)
        assert est is not None
        assert est.get("expected_value_usd") is not None
        assert Decimal(str(est["expected_value_usd"])) > 0
        summaries = result.get("_historical_summaries") or []
        assert any(s.get("valid_for_ev") for s in summaries)
        assert est.get("historical_summary_id")
        assert int(est.get("sample_count") or 0) >= 20
        assert len(stack["submitted"]) == 1
        assert result.get("execution_command_id")
        state = await stack["objective_service"].get_state()
        assert state.total_encumbered_usd > 0
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_full_compiled_graph_missing_ev_does_not_reach_execution(tmp_path) -> None:
    stack = await _prepare_stack(tmp_path, pnl=Decimal("15.00"), n=5)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        estimates = result.get("_strategy_estimates") or []
        if estimates:
            assert estimates[0].get("expected_value_usd") is None or not estimates[0].get(
                "valid"
            )
        assert stack["submitted"] == []
        state = await stack["objective_service"].get_state()
        assert state.working_order_reservation_usd == 0
        assert state.filled_position_exposure_usd == 0
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_full_compiled_graph_negative_ev_does_not_reach_execution(tmp_path) -> None:
    stack = await _prepare_stack(tmp_path, pnl=Decimal("-12.00"), n=20)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        estimates = result.get("_strategy_estimates") or []
        assert estimates
        assert not estimates[0].get("valid")
        assert stack["submitted"] == []
        state = await stack["objective_service"].get_state()
        assert state.total_encumbered_usd == 0
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_kill_switch_blocks_positive_ev_before_paper_submission(tmp_path) -> None:
    stack = await _prepare_stack(
        tmp_path, pnl=Decimal("15.00"), n=20, kill_switch=True
    )
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        est = _valid_estimate(result)
        assert est is not None
        assert stack["submitted"] == []
        assert not result.get("execution_command_id")
        assert stack["broker"].list_open_orders() == []
        state = await stack["objective_service"].get_state()
        assert state.working_order_reservation_usd == 0
        assert state.total_encumbered_usd == 0
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_full_compiled_graph_quote_change_blocks_entry(tmp_path) -> None:
    """Premium rises before gateway; gateway reloads Task-1 quote and rejects."""
    stack = await _prepare_stack(
        tmp_path,
        pnl=Decimal("8.00"),
        n=20,
        option_ask="1.00",
        option_bid="0.90",
    )
    try:
        gateway = stack["gateway"]
        original = stack["original_submit"]
        submitted: list[str] = []

        async def _worse_quote_then_submit(request):
            market = stack["market"]
            clock = stack["clock"]
            start = stack["start"]
            later = start + timedelta(minutes=5)
            clock.set_now(later)
            await market.ingest_option_quotes(
                [
                    {
                        "contract_id": CONTRACT_ID,
                        "symbol": "SPY",
                        "expiry": date(2026, 7, 1),
                        "strike": "500",
                        "option_type": "call",
                        "bid": "1.10",
                        "ask": "1.20",
                        "quote_timestamp": later,
                    }
                ]
            )
            await market.tick(now=later + timedelta(seconds=1))
            result = await original(request)
            if result.submitted:
                submitted.append(result.client_order_id)
            return result

        gateway.submit = _worse_quote_then_submit  # type: ignore[method-assign]
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert submitted == []
        assert stack["submitted"] == []
        state = await stack["objective_service"].get_state()
        assert state.working_order_reservation_usd == 0
        # Estimate may have been valid before quote move
        estimates = result.get("_strategy_estimates") or []
        assert estimates
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_full_compiled_graph_current_quote_positive_ev_submits_once(
    tmp_path,
) -> None:
    stack = await _prepare_stack(
        tmp_path,
        pnl=Decimal("15.00"),
        n=20,
        option_ask="1.05",
        option_bid="1.00",
    )
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert len(stack["submitted"]) == 1
        assert result.get("execution_command_id")
        state = await stack["objective_service"].get_state()
        assert state.working_order_reservation_usd > 0 or state.total_encumbered_usd > 0
    finally:
        await _teardown_stack(stack)

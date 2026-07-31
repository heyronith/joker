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
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.repricing import reprice_long_option_estimate
from joker.objectives.schemas import StrategyObjectiveEstimate
from joker.objectives.service import SessionObjectiveService
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from joker.runtime.cognitive_agent_runtime import build_default_repositories
from joker.runtime.market_runtime import MarketRuntimeConfig
from joker.runtime.order_action_gateway import OrderActionGateway
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned
from tests.objectives.historical_fixtures import seed_positive_history

ET = ZoneInfo("America/New_York")


async def _prepare_stack(tmp_path, *, pnl: Decimal, n: int = 20, kill_switch: bool = False):
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

    apply_objective_migrations(db)
    obj_repo = ObjectiveRepository(db)
    objective_service = SessionObjectiveService(
        obj_repo, require_positive_expected_value=True
    )
    # Deadline must be relative to wall-clock (objective recompute uses now()),
    # not the frozen market clock used for snapshot ingestion.
    deadline = datetime.now(tz=ET) + timedelta(hours=4)
    definition = await objective_service.create_objective(
        session_id=session_id,
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=deadline,
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
                    "require_positive_expected_value": True,
                    "historical_outcomes": app.objective.historical_outcomes.model_copy(
                        update={
                            "minimum_samples_for_ev": 20,
                            "minimum_effective_sample_size": 15,
                            "require_lower_confidence_bound_positive": True,
                            "require_same_strategy_family": False,
                            "minimum_similarity": 0.10,
                        }
                    ),
                }
            )
        }
    )
    engines = build_objective_engines(app)
    hist = engines["historical_outcome_service"]
    hist._repo = obj_repo  # noqa: SLF001
    hist._settings = engines["historical_outcome_settings"]
    seed_positive_history(hist, as_of=start + timedelta(minutes=3), n=n, pnl=pnl)

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
        objective_service=objective_service,
        objective_state_loader=_obj_loader,
        feasibility_engine=engines["feasibility_engine"],
        objective_strategy_scorer=engines["objective_strategy_scorer"],
        capital_sizer=engines["capital_sizer"],
        historical_outcome_service=hist,
        historical_outcome_settings=hist._settings,
        kill_switch=kill_switch,
        **repos,
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
        "hist": hist,
        "supervisor": supervisor,
        "snapshot_id": tick.snapshot.snapshot_id,
        "checkpointer": ckpt,
    }


async def _teardown_stack(stack: dict) -> None:
    await stack["checkpointer"].close()
    await stack["supervisor"].shutdown()
    await drain_aiosqlite_workers(timeout=0.5)


@pytest.mark.asyncio
async def test_full_compiled_graph_positive_ev_reaches_paper_execution(tmp_path) -> None:
    stack = await _prepare_stack(tmp_path, pnl=Decimal("15.00"), n=20)
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        assert result.get("_strategy_estimates")
        est = result["_strategy_estimates"][0]
        assert est.get("expected_value_usd") is not None
        assert Decimal(str(est["expected_value_usd"])) > 0
        assert est.get("valid") is True
        summaries = result.get("_historical_summaries") or []
        assert summaries and summaries[0].get("valid_for_ev") is True
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
    """Kill switch remains stronger than confirmed objective + positive historical EV."""
    stack = await _prepare_stack(
        tmp_path, pnl=Decimal("15.00"), n=20, kill_switch=True
    )
    try:
        result = await stack["graph"].ainvoke(stack["state"], config=stack["config"])
        estimates = result.get("_strategy_estimates") or []
        assert estimates
        assert estimates[0].get("valid") is True
        assert Decimal(str(estimates[0]["expected_value_usd"])) > 0
        summaries = result.get("_historical_summaries") or []
        assert summaries and summaries[0].get("valid_for_ev") is True
        assert stack["submitted"] == []
        assert not result.get("execution_command_id")
        assert stack["broker"].list_open_orders() == []
        assert stack["broker"].list_positions() == []
        state = await stack["objective_service"].get_state()
        assert state.working_order_reservation_usd == 0
        assert state.filled_position_exposure_usd == 0
        assert state.total_encumbered_usd == 0
    finally:
        await _teardown_stack(stack)


@pytest.mark.asyncio
async def test_full_compiled_graph_quote_change_blocks_entry(tmp_path) -> None:
    stack = await _prepare_stack(tmp_path, pnl=Decimal("8.00"), n=20)
    try:
        obj_state = await stack["objective_service"].get_state()
        summary = await stack["hist"].summarize_for_strategy(
            objective_id=obj_state.objective_id,
            strategy_id=uuid4(),
            snapshot_id=stack["snapshot_id"],
            as_of_timestamp=datetime.now(tz=ET),
            direction="bullish",
            strategy_family="bullish",
        )
        est = StrategyObjectiveEstimate(
            strategy_id=uuid4(),
            objective_id=obj_state.objective_id,
            snapshot_id=stack["snapshot_id"],
            expected_value_usd=Decimal("8.00"),
            capital_required_usd=Decimal("100"),
            maximum_loss_usd=Decimal("100"),
            calculation_method="calibrated_episode_average",
            quote_inputs={
                "premium_per_contract": "1.00",
                "quantity": 1,
                "slippage_per_contract": "0.00",
            },
            valid=True,
            historical_summary_id=summary.summary_id,
            sample_count=20,
        )
        stack["objective_service"].save_strategy_estimate(est)
        repriced = reprice_long_option_estimate(
            est,
            current_premium_per_contract_usd=Decimal("1.20"),
            quantity=1,
            request_snapshot_id=uuid4(),
            current_slippage_per_contract_usd=Decimal("0.00"),
        )
        assert repriced.valid is False
        assert repriced.repriced_expected_value_usd is not None
        assert repriced.repriced_expected_value_usd <= 0
        state = await stack["objective_service"].get_state()
        assert state.working_order_reservation_usd == 0
    finally:
        await _teardown_stack(stack)

"""Public LivePaper / cognitive session factory acceptance for historical EV."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings
from joker.events.schemas import EventType
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from joker.runtime.cognitive_session_factory import prepare_cognitive_paper_session
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned
from tests.objectives.historical_fixtures import (
    make_closed_episode,
    persist_compiler_produced_history,
)

ET = ZoneInfo("America/New_York")


def _app(tmp_path, *, kill_switch: bool = False) -> AppSettings:
    app = AppSettings(db_path=str(tmp_path / "live.db"))
    return app.model_copy(
        update={
            "live_trading_enabled": False,
            "evolution": app.evolution.model_copy(update={"enabled": True}),
            "objective": app.objective.model_copy(
                update={
                    "enabled": True,
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
                    "execution": app.objective.execution.model_copy(
                        update={"maximum_buy_limit_above_ask_pct": 5.0}
                    ),
                }
            ),
            "risk": app.risk.model_copy(update={"kill_switch": kill_switch}),
            "data_quality": app.data_quality.model_copy(
                update={
                    "option_stale_seconds": 3600,
                    "maximum_relative_spread": 0.50,
                }
            ),
        }
    )


async def _ingest_market(session, *, ask: str = "1.10", bid: str = "1.00"):
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    market = session.supervisor.market_runtime
    assert market is not None
    clock = getattr(session.supervisor, "clock", None)
    for i in range(3):
        ts = start + timedelta(minutes=i, seconds=5)
        if clock is not None and hasattr(clock, "set_now"):
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
                "bid": bid,
                "ask": ask,
                "quote_timestamp": start + timedelta(minutes=3),
            }
        ]
    )
    later = start + timedelta(minutes=3, seconds=3)
    if clock is not None and hasattr(clock, "set_now"):
        clock.set_now(later)
    tick = await market.tick(now=later)
    assert tick.snapshot is not None
    return tick, later


async def _prepare_confirmed_session(
    tmp_path,
    *,
    session_id: str,
    kill_switch: bool = False,
):
    app = _app(tmp_path, kill_switch=kill_switch)
    apply_objective_migrations(tmp_path / "live.db")
    obj_repo = ObjectiveRepository(tmp_path / "live.db")
    objective_service = SessionObjectiveService(
        obj_repo, require_positive_expected_value=True
    )
    definition = await objective_service.create_objective(
        session_id=session_id,
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=4),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await objective_service.confirm_objective(definition.objective_id)
    from joker.models.fake_provider import FakeModelProvider

    fake = FakeModelProvider(available=True)
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    session = await prepare_cognitive_paper_session(
        app_settings=app,
        objective_service=objective_service,
        broker=PaperBroker(slippage_pct=0),
        db_path=tmp_path / "live.db",
        session_id=session_id,
        fake_model_provider=fake,
        clock=FrozenExchangeClock(start, calendar=MarketCalendar()),
        start_cognitive_agent=True,
    )
    assert session.agent_runtime._started is True
    # Public path started the cognitive runtime; pause event + evolution
    # workers so this acceptance test owns the single controlled graph invoke
    # (avoids dual writers on the shared SQLite file).
    await session.agent_runtime.pause_event_workers()
    await session.evolution_runtime.stop_workers()
    return session, objective_service, fake


async def _bind_quote_loader(session, as_of) -> CognitiveCheckpointer:
    ckpt = CognitiveCheckpointer(session.db_path.with_name("ckpt.db"))
    saver = await ckpt.open()
    session.graph_deps.checkpointer = saver
    session.graph_deps.clock = FrozenExchangeClock(as_of, calendar=MarketCalendar())
    session.graph_deps.max_quote_age_seconds = 3600
    session.graph_deps.max_relative_spread = 0.50
    from joker.objectives.execution_quote import build_current_option_quote_loader

    session.graph_deps.current_option_quote_loader = build_current_option_quote_loader(
        session.graph_deps, max_quote_age_seconds=3600, max_relative_spread=0.50
    )
    return ckpt


async def _track_submits(gateway):
    submitted: list[str] = []
    original = gateway.submit

    async def _track(request):
        result = await original(request)
        if result.submitted:
            submitted.append(result.client_order_id)
        return result

    gateway.submit = _track  # type: ignore[method-assign]
    return submitted


@pytest.mark.asyncio
async def test_process_restart_recovers_stable_objective_and_working_component(
    tmp_path,
) -> None:
    from joker.persistence.cognitive_execution_provenance import (
        PortfolioComponentStatus,
        PortfolioExecutionComponentRecord,
        stable_portfolio_client_order_id,
    )
    from joker.runtime.cognitive_session import stable_cognitive_session_id
    from joker.runtime.execution_runtime import ExecutionCommand
    from joker.schemas.domain import OptionContract, OrderIntent

    app = _app(tmp_path)
    db = tmp_path / "live.db"
    clock = FrozenExchangeClock(
        datetime(2026, 7, 1, 10, 0, tzinfo=ET), calendar=MarketCalendar()
    )
    stable_session_id = stable_cognitive_session_id(
        trading_date=clock.trading_date(), account_identity="local_paper"
    )
    apply_objective_migrations(db)
    repository = ObjectiveRepository(db)
    first_service = SessionObjectiveService(repository)
    definition = await first_service.create_objective(
        session_id=stable_session_id,
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime(2026, 7, 1, 15, 30, tzinfo=ET),
        max_concurrent_positions=2,
        accepted_total_loss_risk=True,
    )
    await first_service.confirm_objective(
        definition.objective_id,
        confirmed_at_exchange_time=clock.now(),
    )
    broker = PaperBroker(slippage_pct=2.0)
    from joker.models.fake_provider import FakeModelProvider

    first_run_id = str(uuid4())
    first = await prepare_cognitive_paper_session(
        app_settings=app,
        objective_service=first_service,
        broker=broker,
        db_path=db,
        run_id=first_run_id,
        fake_model_provider=FakeModelProvider(available=True),
        clock=clock,
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    decision_id = str(uuid4())
    first_tuple_id = str(uuid4())
    second_tuple_id = str(uuid4())
    first_client_order_id = stable_portfolio_client_order_id(
        decision_id, first_tuple_id
    )
    contract = OptionContract(
        expiration=clock.trading_date(), strike=500.0, option_type="call"
    )
    order = await first.supervisor.execution_runtime.submit_execution_command(
        ExecutionCommand(
            client_order_id=first_client_order_id,
            intent=OrderIntent(
                intent_id=first_client_order_id,
                candidate_id="restart-fixture",
                contract=contract,
                side="buy",
                quantity=1,
                limit_price=1.0,
                position_intent="BUY_TO_OPEN",
            ),
            broker_account_id="local_paper",
            provenance={"fixture": "stable_restart"},
        )
    )
    assert order.status == "open"
    portfolio_repo = first.graph_deps.provenance_registry.portfolio_executions
    common = {
        "session_id": stable_session_id,
        "origin_run_id": first_run_id,
        "broker_account_identity": "local_paper",
        "trading_date": clock.trading_date().isoformat(),
        "target_portfolio_decision_id": decision_id,
        "selected_portfolio_id": str(uuid4()),
        "component_count": 2,
        "strategy_id": str(uuid4()),
        "authorized_quantity": 1,
        "capital_allocation": Decimal("100"),
        "remaining_quantity": 1,
        "original_decision_snapshot_id": str(uuid4()),
        "evaluated_objective_version": 1,
        "evaluated_timestamp": clock.now().isoformat(),
        "extra": {},
    }
    await portfolio_repo.authorize(
        PortfolioExecutionComponentRecord(
            **common,
            authorized_position_tuple_id=first_tuple_id,
            component_index=0,
            contract_id=CONTRACT_ID,
            client_order_id=first_client_order_id,
            broker_order_id=order.order_id,
            status=PortfolioComponentStatus.WORKING,
            submitted_quantity=1,
        )
    )
    await portfolio_repo.authorize(
        PortfolioExecutionComponentRecord(
            **common,
            authorized_position_tuple_id=second_tuple_id,
            component_index=1,
            contract_id="SPY:2026-07-01:501:call",
            client_order_id=stable_portfolio_client_order_id(
                decision_id, second_tuple_id
            ),
            status=PortfolioComponentStatus.AUTHORIZED,
        )
    )
    await first.shutdown()

    second_run_id = str(uuid4())
    second_service = SessionObjectiveService(ObjectiveRepository(db))
    second = await prepare_cognitive_paper_session(
        app_settings=app,
        objective_service=second_service,
        broker=broker,
        db_path=db,
        run_id=second_run_id,
        fake_model_provider=FakeModelProvider(available=True),
        clock=clock,
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    try:
        assert second.session_id == first.session_id == stable_session_id
        assert second.run_id != first.run_id
        recovered_objective = await second.objective_service.get_state()
        assert str(recovered_objective.objective_id) == str(definition.objective_id)
        second.agent_runtime._decision_graph = build_cognitive_graph(second.graph_deps)
        await second.agent_runtime._resume_pending_portfolio_executions()
        records = await second.graph_deps.provenance_registry.portfolio_executions.list_by_decision(
            decision_id
        )
        assert [record.status for record in records] == [
            PortfolioComponentStatus.WORKING,
            PortfolioComponentStatus.AUTHORIZED,
        ]
        assert records[0].origin_run_id == first_run_id
        assert records[0].last_resumed_run_id == second_run_id
        assert records[0].resume_count == 1
        assert len(broker.list_open_orders()) == 1
    finally:
        await second.shutdown()


@pytest.mark.asyncio
async def test_public_live_runner_positive_ev_reaches_paper_broker(tmp_path) -> None:
    session, objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="live-sess"
    )
    try:
        hist = session.historical_outcome_service
        assert hist is not None
        assert hist.uses_repository_loaders
        evo_eps = session.evolution_runtime.repositories["episodes"]
        assert hist._episode_loader is not None

        tick, as_of = await _ingest_market(session, ask="1.10", bid="1.00")
        rows = await persist_compiler_produced_history(
            episode_repo=evo_eps,
            evaluation_repo=session.evolution_runtime.repositories["evaluations"],
            as_of=as_of,
            n=20,
            pnl=Decimal("18.00"),
        )
        assert all(ep.strategy_family == "breakout_continuation" for ep, _ in rows)
        assert all(ep.completed for ep, _ in rows)
        assert all(
            "historical_ev_eligible=false" not in ep.completeness_findings
            for ep, _ in rows
        )
        assert all(
            ep.entry_decision_event_id is not None
            and ep.terminal_event_id is not None
            and ep.entry_decision_event_id in ep.market_event_ids
            and ep.terminal_event_id in ep.market_event_ids
            for ep, _ in rows
        )

        cycle_id = "live-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        gateway = session.order_action_gateway
        assert gateway is not None
        submitted = await _track_submits(gateway)

        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        result = await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        estimates = result.get("_strategy_estimates") or []
        valid = next((e for e in estimates if e.get("valid")), None)
        assert valid is not None
        assert len(submitted) == 1
        assert result.get("execution_command_id")
        obj_state = await objective_service.get_state()
        # PaperBroker may fill immediately, converting reservation → filled exposure.
        encumbered = (
            obj_state.working_order_reservation_usd
            + obj_state.filled_position_exposure_usd
        )
        assert encumbered >= Decimal("110.00")
        assert isinstance(session.broker, PaperBroker)
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_kill_switch_blocks_entry(tmp_path) -> None:
    session, objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="kill-sess", kill_switch=True
    )
    try:
        tick, as_of = await _ingest_market(session)
        await persist_compiler_produced_history(
            episode_repo=session.evolution_runtime.repositories["episodes"],
            evaluation_repo=session.evolution_runtime.repositories["evaluations"],
            as_of=as_of,
            n=20,
            pnl=Decimal("18.00"),
        )
        cycle_id = "kill-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        submitted = await _track_submits(session.order_action_gateway)
        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        assert submitted == []
        obj_state = await objective_service.get_state()
        assert obj_state.working_order_reservation_usd == 0
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_insufficient_history_blocks(tmp_path) -> None:
    session, _objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="cold-sess"
    )
    try:
        tick, as_of = await _ingest_market(session)
        await persist_compiler_produced_history(
            episode_repo=session.evolution_runtime.repositories["episodes"],
            evaluation_repo=session.evolution_runtime.repositories["evaluations"],
            as_of=as_of,
            n=5,
            pnl=Decimal("18.00"),
        )
        cycle_id = "cold-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        submitted = await _track_submits(session.order_action_gateway)
        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        result = await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        assert submitted == []
        estimates = result.get("_strategy_estimates") or []
        assert not any(e.get("valid") for e in estimates)
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_extreme_limit_blocks_submission(tmp_path) -> None:
    """Public path: extreme buy limit vs ask must not reach PaperBroker."""
    session, objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="limit-sess"
    )
    try:
        tick, as_of = await _ingest_market(session, ask="1.10", bid="1.00")
        await persist_compiler_produced_history(
            episode_repo=session.evolution_runtime.repositories["episodes"],
            evaluation_repo=session.evolution_runtime.repositories["evaluations"],
            as_of=as_of,
            n=20,
            pnl=Decimal("18.00"),
        )
        cycle_id = "limit-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        gateway = session.order_action_gateway
        assert gateway is not None
        original = gateway.submit
        submitted: list[str] = []

        async def _force_extreme(request):
            result = await original(replace(request, limit_price=99.0))
            if result.submitted:
                submitted.append(result.client_order_id)
            return result

        gateway.submit = _force_extreme  # type: ignore[method-assign]
        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        assert submitted == []
        obj_state = await objective_service.get_state()
        assert obj_state.working_order_reservation_usd == 0
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_missing_strategy_metadata_blocks(tmp_path) -> None:
    session, _objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="meta-sess"
    )
    try:
        tick, as_of = await _ingest_market(session)
        ep_repo = session.evolution_runtime.repositories["episodes"]
        ev_repo = session.evolution_runtime.repositories["evaluations"]
        for i in range(20):
            episode, evaluation = make_closed_episode(
                pnl=Decimal("18.00"),
                as_of=as_of,
                hours_before=24 + i,
                strategy_family=None,
                findings=(
                    "historical_strategy_family_missing",
                    "historical_ev_eligible=false",
                ),
            )
            await ep_repo.append(episode)
            await ev_repo.append(evaluation)
        cycle_id = "meta-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        submitted = await _track_submits(session.order_action_gateway)
        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        result = await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        assert submitted == []
        estimates = result.get("_strategy_estimates") or []
        assert not any(e.get("valid") for e in estimates)
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_horizon_failure_blocks(tmp_path) -> None:
    session, _objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="horizon-sess"
    )
    try:
        tick, as_of = await _ingest_market(session)
        ep_repo = session.evolution_runtime.repositories["episodes"]
        ev_repo = session.evolution_runtime.repositories["evaluations"]
        for i in range(20):
            episode, evaluation = make_closed_episode(
                pnl=Decimal("18.00"),
                as_of=as_of,
                hours_before=24 + i,
                completed=False,
                findings=(
                    "authoritative_horizon_incomplete",
                    "truth_degraded=true",
                    "promotion_eligible=false",
                    "historical_ev_eligible=false",
                ),
            )
            await ep_repo.append(episode)
            await ev_repo.append(evaluation)
        cycle_id = "horizon-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        submitted = await _track_submits(session.order_action_gateway)
        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        result = await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        assert submitted == []
        estimates = result.get("_strategy_estimates") or []
        assert not any(e.get("valid") for e in estimates)
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_history_uses_factual_entry_anchors(tmp_path) -> None:
    session, _, _ = await _prepare_confirmed_session(
        tmp_path, session_id="anchor-sess"
    )
    try:
        _, as_of = await _ingest_market(session)
        rows = await persist_compiler_produced_history(
            episode_repo=session.evolution_runtime.repositories["episodes"],
            evaluation_repo=session.evolution_runtime.repositories["evaluations"],
            as_of=as_of,
            n=5,
            pnl=Decimal("18.00"),
        )
        for ep, _ in rows:
            assert ep.entry_decision_event_id is not None
            assert ep.terminal_event_id is not None
            assert ep.entry_decision_event_id in ep.market_event_ids
            assert ep.terminal_event_id in ep.market_event_ids
            assert ep.completed is True
            assert "authoritative_horizon_entry_missing" not in ep.completeness_findings
            assert "historical_ev_eligible=false" not in ep.completeness_findings
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


@pytest.mark.asyncio
async def test_public_live_runner_non_positive_ev_blocks(tmp_path) -> None:
    session, _objective_service, fake = await _prepare_confirmed_session(
        tmp_path, session_id="neg-ev-sess"
    )
    try:
        tick, as_of = await _ingest_market(session)
        # Losing history → non-positive EV under default policy.
        await persist_compiler_produced_history(
            episode_repo=session.evolution_runtime.repositories["episodes"],
            evaluation_repo=session.evolution_runtime.repositories["evaluations"],
            as_of=as_of,
            n=20,
            pnl=Decimal("-18.00"),
        )
        cycle_id = "neg-ev-cycle"
        register_full_path_canned(
            fake, tick.snapshot.snapshot_id, cycle_id, session=session.session_id
        )
        ckpt = await _bind_quote_loader(session, as_of)
        submitted = await _track_submits(session.order_action_gateway)
        graph = build_cognitive_graph(session.graph_deps)
        state = initial_cycle_state(
            session_id=session.session_id,
            run_id=session.run_id,
            cycle_id=cycle_id,
            trigger_event_id=str(uuid4()),
            trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
            snapshot_id=str(tick.snapshot.snapshot_id),
        )
        result = await graph.ainvoke(
            state,
            config=ainvoke_config(
                session_id=session.session_id, graph_kind="decision", cycle_id=cycle_id
            ),
        )
        assert submitted == []
        estimates = result.get("_strategy_estimates") or []
        assert not any(e.get("valid") for e in estimates)
        await ckpt.close()
    finally:
        await session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)

"""Paper vs live: identical commands after the real public graph compile path."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings
from joker.events.schemas import EventType
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer, ainvoke_config
from joker.models.fake_provider import FakeModelProvider
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from joker.persistence.migrations import apply_task1_migrations
from joker.runtime.cognitive_session_factory import (
    prepare_cognitive_live_session,
    prepare_cognitive_paper_session,
)
from joker.runtime.execution_runtime import ExecutionCommand
from joker.runtime.order_action_gateway import OrderActionGateway
from joker.schemas.domain import BrokerOrder
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.broker._live_helpers import make_live_client
from tests.cognitive.task2_canned import CONTRACT_ID, register_full_path_canned
from tests.objectives.historical_fixtures import persist_compiler_produced_history

ET = ZoneInfo("America/New_York")


def _command_fields(cmd: ExecutionCommand) -> dict:
    intent = cmd.intent
    contract = intent.contract
    return {
        "contract": (
            f"{contract.symbol}:{contract.expiration.isoformat()}:"
            f"{contract.strike}:{contract.option_type}"
        ),
        "side": intent.side,
        "quantity": intent.quantity,
        "limit_price": intent.limit_price,
        "position_intent": intent.position_intent,
        "broker_account_id": cmd.broker_account_id,
    }


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


async def _bind_quote_loader(session, as_of) -> CognitiveCheckpointer:
    ckpt = CognitiveCheckpointer(session.db_path.with_name(f"ckpt-{session.session_id}.db"))
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


async def _confirmed_objective(db, *, session_id: str) -> SessionObjectiveService:
    apply_task1_migrations(db)
    apply_objective_migrations(db)
    obj_repo = ObjectiveRepository(db)
    objective_service = SessionObjectiveService(
        obj_repo, require_positive_expected_value=False
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
    return objective_service


@pytest.mark.asyncio
async def test_public_graph_paper_live_equivalence(tmp_path, monkeypatch) -> None:
    """Run the same deterministic graph cycle through both public session factories."""
    paper_db = tmp_path / "equiv-paper.db"
    live_db = tmp_path / "equiv-live.db"
    paper_objective = await _confirmed_objective(paper_db, session_id="equiv-paper")
    live_objective = await _confirmed_objective(live_db, session_id="equiv-live")

    objective_cfg = {
        "enabled": True,
        "require_positive_expected_value": False,
        "historical_outcomes": {
            "minimum_samples_for_ev": 1,
            "minimum_effective_sample_size": 1,
            "require_lower_confidence_bound_positive": False,
            "require_same_strategy_family": False,
            "minimum_similarity": 0.01,
        },
        "execution": {"maximum_buy_limit_above_ask_pct": 5.0},
    }
    data_quality = {
        "option_stale_seconds": 3600,
        "maximum_relative_spread": 0.50,
    }
    paper_app = AppSettings(
        db_path=str(paper_db),
        live_trading_enabled=False,
        evolution={"enabled": True},
        objective=objective_cfg,
        cognitive_graph={"enabled": True},
        data_quality=data_quality,
    )
    live_app = AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(live_db),
        broker={"provider": "webull_live"},
        evolution={"enabled": True},
        objective=objective_cfg,
        cognitive_graph={"enabled": True},
        data_quality=data_quality,
    )
    # Shared FakeModelProvider so canned role bindings are identical.
    fake = FakeModelProvider(available=True)
    clock = FrozenExchangeClock(
        datetime(2026, 7, 1, 10, 0, tzinfo=ET), calendar=MarketCalendar()
    )

    paper_session = await prepare_cognitive_paper_session(
        app_settings=paper_app,
        objective_service=paper_objective,
        broker=PaperBroker(slippage_pct=0),
        db_path=paper_db,
        session_id="equiv-paper",
        fake_model_provider=fake,
        clock=clock,
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    live_client, _, _ = make_live_client(tmp_path, capture_only=True)
    live_session = await prepare_cognitive_live_session(
        app_settings=live_app,
        objective_service=live_objective,
        broker=live_client,
        db_path=live_db,
        session_id="equiv-live",
        fake_model_provider=fake,
        clock=clock,
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    paper_ckpt = live_ckpt = None
    try:
        # Seed identical market truth into both MarketRuntimes.
        paper_tick, as_of = await _ingest_market(paper_session)
        live_tick, _ = await _ingest_market(live_session)
        assert paper_tick.snapshot is not None and live_tick.snapshot is not None

        for session in (paper_session, live_session):
            await persist_compiler_produced_history(
                episode_repo=session.evolution_runtime.repositories["episodes"],
                evaluation_repo=session.evolution_runtime.repositories["evaluations"],
                as_of=as_of,
                n=20,
                pnl=Decimal("18.00"),
            )

        # Intercept submit only — do not skip validate/compile or load_snapshot_truth.
        monkeypatch.setattr(
            OrderActionGateway,
            "_maybe_live_preview",
            AsyncMock(return_value=None),
        )

        paper_cmds: list[ExecutionCommand] = []
        live_cmds: list[ExecutionCommand] = []

        async def _intercept_paper(cmd: ExecutionCommand) -> BrokerOrder:
            paper_cmds.append(cmd)
            raise RuntimeError("stop-before-broker")

        async def _intercept_live(cmd: ExecutionCommand) -> BrokerOrder:
            live_cmds.append(cmd)
            raise RuntimeError("stop-before-broker")

        paper_session.graph_deps.execution_runtime.submit_execution_command = (  # type: ignore[method-assign]
            _intercept_paper
        )
        live_session.graph_deps.execution_runtime.submit_execution_command = (  # type: ignore[method-assign]
            _intercept_live
        )
        paper_session.bridge.execution_runtime.submit_execution_command = (  # type: ignore[method-assign]
            _intercept_paper
        )
        live_session.bridge.execution_runtime.submit_execution_command = (  # type: ignore[method-assign]
            _intercept_live
        )

        paper_ckpt = await _bind_quote_loader(paper_session, as_of)
        live_ckpt = await _bind_quote_loader(live_session, as_of)

        cycle_id = "equiv-cycle"
        # Same canned agent outputs keyed to each session's snapshot + session id.
        register_full_path_canned(
            fake,
            paper_tick.snapshot.snapshot_id,
            cycle_id,
            session=paper_session.session_id,
        )
        register_full_path_canned(
            fake,
            live_tick.snapshot.snapshot_id,
            cycle_id,
            session=live_session.session_id,
        )

        async def _run_graph(session, snapshot_id: str) -> None:
            graph = build_cognitive_graph(session.graph_deps)
            state = initial_cycle_state(
                session_id=session.session_id,
                run_id=session.run_id,
                cycle_id=cycle_id,
                trigger_event_id=str(uuid4()),
                trigger_event_type=EventType.MARKET_SNAPSHOT_CREATED.value,
                snapshot_id=str(snapshot_id),
            )
            try:
                await graph.ainvoke(
                    state,
                    config=ainvoke_config(
                        session_id=session.session_id,
                        graph_kind="decision",
                        cycle_id=cycle_id,
                    ),
                )
            except RuntimeError as exc:
                if "stop-before-broker" not in str(exc):
                    raise

        await _run_graph(paper_session, paper_tick.snapshot.snapshot_id)
        await _run_graph(live_session, live_tick.snapshot.snapshot_id)

        assert len(paper_cmds) == 1, "paper graph must compile a real command"
        assert len(live_cmds) == 1, "live graph must compile a real command"
        paper_fields = _command_fields(paper_cmds[0])
        live_fields = _command_fields(live_cmds[0])
        for key in ("contract", "side", "quantity", "limit_price", "position_intent"):
            assert paper_fields[key] == live_fields[key], key
        # Same compiled trade; distinct broker surfaces (paper vs WebullLiveClient).
        assert isinstance(paper_session.broker, PaperBroker)
        assert paper_session.broker.__class__.__name__ != live_session.broker.__class__.__name__
    finally:
        if paper_ckpt is not None:
            await paper_ckpt.close()
        if live_ckpt is not None:
            await live_ckpt.close()
        await paper_session.shutdown()
        await live_session.shutdown()
        await drain_aiosqlite_workers(timeout=1.0)


def test_live_capture_mode_preserves_payload_without_placement(tmp_path) -> None:
    """Lightweight capture payload check — no broker placement."""
    from tests.broker._live_helpers import make_intent

    client, api, _ = make_live_client(tmp_path, capture_only=True)
    intent = make_intent()
    order = client.submit_order(intent)
    assert order.status == "pending"
    assert api.placed == []
    assert len(client.captured_payloads) == 1
    payload = client.captured_payloads[0]
    assert payload["limit_price"] == "1.10"
    assert payload["quantity"] == "1"
    assert payload["position_intent"] == "BUY_TO_OPEN"
    assert payload["side"] == "BUY"

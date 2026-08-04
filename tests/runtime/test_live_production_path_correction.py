"""Correction tests for production model/market path and fill/session/preflight truth."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.app.safety import SafetyMode
from joker.broker.account_truth import hash_account_id
from joker.broker.interface import PaperBroker
from joker.broker.webull_live import create_mock_live_trade_api
from joker.cognition.exceptions import CognitiveRuntimeConfigurationError
from joker.config.settings import AppSettings
from joker.data.webull_api import MockWebullMarketApi, WebullCandle, WebullQuote
from joker.data.webull_options_api import MockWebullOptionsMarketApi
from joker.events.bus import InProcessAsyncEventBus
from joker.ledger.schemas import LedgerEventType
from joker.ledger.store import SqliteLedgerStore
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.persistence.migrations import apply_task1_migrations
from joker.runtime.cognitive_session import live_gated_cognitive_session_id
from joker.runtime.cognitive_session_factory import prepare_cognitive_live_session
from joker.runtime.cognitive_startup import CognitiveStartupResult, ProviderAvailabilityReport
from joker.runtime.execution_runtime import ExecutionRuntime
from joker.runtime.live_activation import create_live_activation
from joker.runtime.live_preflight import run_production_preflight
from joker.runtime.live_trading import LiveTradingError, LiveTradingRunner
from joker.schemas.domain import BrokerOrder, OptionContract
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock, SessionPhase
from tests.broker._live_helpers import live_env, make_intent, make_live_client, prepare_journal_for_intent
from tests.cognitive.task2_canned import CONTRACT_ID

ET = ZoneInfo("America/New_York")


def _live_app(tmp_path, **kwargs) -> AppSettings:
    base = dict(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(tmp_path / "live.db"),
        broker={"provider": "webull_live"},
        evolution={"enabled": True},
        objective={"enabled": True, "require_positive_expected_value": False},
        cognitive_graph={"enabled": True},
        agents={"mock_agents": False},
    )
    base.update(kwargs)
    return AppSettings(**base)


async def _confirmed_objective(
    tmp_path, *, capital: Decimal = Decimal("500"), session_id: str = "s"
):
    db = tmp_path / "live.db"
    apply_task1_migrations(db)
    apply_objective_migrations(db)
    repo = ObjectiveRepository(db)
    svc = SessionObjectiveService(repo, require_positive_expected_value=False)
    definition = await svc.create_objective(
        session_id=session_id,
        authorised_capital_usd=capital,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=4),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    return svc, definition


async def _exec_runtime(tmp_path, broker, *, session_id: str = "s"):
    now = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    clock = FrozenExchangeClock(now, calendar=MarketCalendar())
    ledger = SqliteLedgerStore(tmp_path / f"{session_id}.db")
    await ledger.initialize()
    bus = InProcessAsyncEventBus()
    rt = ExecutionRuntime(
        broker=broker,
        ledger_store=ledger,
        event_bus=bus,
        clock=clock,
        session_id=session_id,
        broker_account_id="acct",
    )
    return rt, ledger, bus


@pytest.mark.asyncio
async def test_normal_live_session_uses_configured_model_providers(
    tmp_path, monkeypatch
) -> None:
    app = _live_app(tmp_path)
    profiles = default_model_profiles()
    cfg = ModelsConfig(
        profiles={
            n: p.model_copy(update={"provider": "ollama", "model": "qwen"})
            for n, p in profiles.items()
        }
    )
    cfg = cfg.model_copy(
        update={"ollama": cfg.ollama.model_copy(update={"enabled": True})}
    )
    app = app.model_copy(update={"models": cfg})
    seen: dict = {}

    async def _capture(models_cfg, *, mock_agents=False, registry=None):
        seen["mock_agents"] = mock_agents
        seen["providers"] = {
            n: p.provider for n, p in models_cfg.profiles.items() if p.enabled
        }
        fake = FakeModelProvider(available=True)
        remapped = {
            n: p.model_copy(update={"provider": "fake", "model": "fake-model"})
            for n, p in default_model_profiles().items()
        }
        reg = ModelRegistry(ModelsConfig(profiles=remapped), providers={"fake": fake})
        return CognitiveStartupResult(
            registry=reg,
            availability=ProviderAvailabilityReport(
                ollama_enabled=True,
                ollama_healthy=True,
                openai_enabled=False,
                openai_healthy=False,
                fake_forced=False,
                mandatory_profiles=tuple(remapped),
                healthy_mandatory_profiles=tuple(remapped),
            ),
            mock_session=False,
            remapped_to_fake=False,
            details={},
        )

    monkeypatch.setattr(
        "joker.runtime.cognitive_session_factory.validate_cognitive_providers",
        _capture,
    )
    svc, _ = await _confirmed_objective(tmp_path)
    client, _, _ = make_live_client(tmp_path, capture_only=True)
    session = await prepare_cognitive_live_session(
        app_settings=app,
        objective_service=svc,
        broker=client,
        db_path=tmp_path / "live.db",
        session_id="model-sess",
        start_cognitive_agent=False,
        start_evolution_workers=False,
    )
    try:
        assert seen["mock_agents"] is False
        assert all(p == "ollama" for p in seen["providers"].values())
        assert "fake" not in set(seen["providers"].values())
    finally:
        await session.shutdown()


@pytest.mark.asyncio
async def test_live_provider_failure_blocks_startup(tmp_path, monkeypatch) -> None:
    app = _live_app(tmp_path)

    async def _boom(cfg, *, mock_agents=False, registry=None):
        raise CognitiveRuntimeConfigurationError("ollama unhealthy")

    monkeypatch.setattr(
        "joker.runtime.cognitive_session_factory.validate_cognitive_providers",
        _boom,
    )
    svc, definition = await _confirmed_objective(tmp_path)
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=definition.objective_id,
        authorized_capital_usd=Decimal("500"),
    )
    runner = LiveTradingRunner(
        app_settings=app,
        env=live_env(),
        objective_service=svc,
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        capture_only=True,
        db_path=tmp_path / "live.db",
    )
    with pytest.raises(CognitiveRuntimeConfigurationError, match="ollama unhealthy"):
        await runner.start(
            start_cognitive_agent=False,
            start_evolution_workers=False,
            start_market_loop=False,
        )


@pytest.mark.asyncio
async def test_live_runner_receives_real_market_observations(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("WEBULL_MARKET_DATA_ENABLED", "true")
    from joker.data import webull_capability

    monkeypatch.setattr(webull_capability, "capability_usable_for_shadow", lambda: True)

    svc, definition = await _confirmed_objective(tmp_path)
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=definition.objective_id,
        authorized_capital_usd=Decimal("500"),
    )
    now = datetime.now(timezone.utc)
    quote = WebullQuote(
        symbol="SPY",
        price=500.0,
        bid=499.9,
        ask=500.1,
        timestamp=now,
        delayed=False,
    )
    candles = [
        WebullCandle(
            timestamp=now - timedelta(minutes=i),
            open=500,
            high=501,
            low=499,
            close=500,
            volume=1000,
        )
        for i in range(12, 0, -1)
    ]
    stock_api = MockWebullMarketApi(
        quote=quote, candles=candles, stream_quotes=[quote, quote]
    )
    today = date.today()
    contract = OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=today,
        strike=500.0,
        option_type="call",
        contract_id=CONTRACT_ID,
        source="webull_opra",
    )
    call = OptionSnapshot(
        contract=contract,
        bid=1.0,
        ask=1.1,
        mid=1.05,
        spread_pct=9.5,
        quote_timestamp=now,
        delayed=False,
        source="webull_opra",
        is_synthetic=False,
    )
    options_api = MockWebullOptionsMarketApi(
        contracts=[call.contract],
        snapshots={call.contract.contract_id: call},
    )
    runner = LiveTradingRunner(
        app_settings=_live_app(tmp_path),
        env=live_env(WEBULL_MARKET_DATA_ENABLED=True, WEBULL_APP_KEY="k"),
        objective_service=svc,
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        stock_api=stock_api,
        options_api=options_api,
        capture_only=True,
        db_path=tmp_path / "live.db",
    )
    session = await runner.start(
        fake_model_provider=FakeModelProvider(available=True),
        start_cognitive_agent=False,
        start_evolution_workers=False,
        start_market_loop=True,
    )
    try:
        assert runner._market_loop is not None
        assert runner._market_loop.observations_received >= 1
        assert stock_api.snapshot_calls >= 1
        assert runner.health().market_data_healthy is True
    finally:
        await runner.shutdown()


@pytest.mark.asyncio
async def test_partial_fill_without_quantity_does_not_mutate_truth(tmp_path) -> None:
    client, api, _ = make_live_client(tmp_path)
    intent = make_intent(qty=2)
    prepare_journal_for_intent(client, intent)
    client.journal.transition(  # type: ignore[union-attr]
        account_id_hash=client.account_id_hash,
        client_order_id=intent.intent_id,
        status="accepted",
        broker_order_id="brok-1",
    )
    api._orders[intent.intent_id] = {
        "client_order_id": intent.intent_id,
        "status": "PARTIAL_FILLED",
        "quantity": 2,
        "side": "BUY",
        "symbol": "SPY",
        "strike": 500,
        "expirationDate": date.today().isoformat(),
        "optionType": "call",
        # no filled_quantity / cumulative_filled_quantity
    }
    assert client.get_order(intent.intent_id) is None
    assert client._truth_unavailable is True

    rt, ledger, bus = await _exec_runtime(tmp_path, client, session_id="s-missing")
    try:
        bogus = BrokerOrder(
            order_id=intent.intent_id,
            intent_id=intent.intent_id,
            status="partially_filled",
            contract=OptionContract(
                symbol="SPY",
                expiration=date.today(),
                strike=500.0,
                option_type="call",
            ),
            side="buy",
            quantity=2,
            limit_price=1.10,
            filled_quantity=0,
            remaining_quantity=2,
        )
        written = await rt.on_broker_update(bogus, client_order_id=intent.intent_id)
        events = await ledger.get_by_session("s-missing")
        fills = [
            e
            for e in events
            if e.event_type
            in {LedgerEventType.PARTIAL_FILL, LedgerEventType.FINAL_FILL}
        ]
        assert written == []
        assert fills == []
    finally:
        await ledger.close()
        await bus.close()


@pytest.mark.asyncio
async def test_two_equal_partial_fills_are_both_persisted(tmp_path) -> None:
    broker = PaperBroker(slippage_pct=0)
    rt, ledger, bus = await _exec_runtime(tmp_path, broker, session_id="s-two")
    cid = "q" * 32
    contract = OptionContract(
        symbol="SPY", expiration=date(2026, 7, 1), strike=500.0, option_type="call"
    )
    try:
        p1 = BrokerOrder(
            order_id=cid,
            intent_id=cid,
            status="partially_filled",
            contract=contract,
            side="buy",
            quantity=3,
            limit_price=1.10,
            filled_quantity=1,
            remaining_quantity=2,
            average_fill_price=1.10,
        )
        broker._orders[cid] = p1
        await rt.on_broker_update(p1, client_order_id=cid)
        p2 = p1.model_copy(update={"filled_quantity": 2, "remaining_quantity": 1})
        broker._orders[cid] = p2
        await rt.on_broker_update(p2, client_order_id=cid)
        events = await ledger.get_by_session("s-two")
        p_fills = [
            e
            for e in events
            if e.client_order_id == cid and e.event_type == LedgerEventType.PARTIAL_FILL
        ]
        assert len(p_fills) == 2
        assert len({e.idempotency_key for e in p_fills}) == 2
        assert all(":1:" in e.idempotency_key or ":2:" in e.idempotency_key for e in p_fills)
        assert all(str(e.quantity) == "1" for e in p_fills)
    finally:
        await ledger.close()
        await bus.close()


@pytest.mark.asyncio
async def test_default_live_session_id_survives_restart(tmp_path) -> None:
    svc, definition = await _confirmed_objective(tmp_path)
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=definition.objective_id,
        authorized_capital_usd=Decimal("500"),
    )
    expected = live_gated_cognitive_session_id(
        account_id_hash=activation.account_id_hash,
    )
    fake = FakeModelProvider(available=True)

    async def _start():
        runner = LiveTradingRunner(
            app_settings=_live_app(tmp_path),
            env=live_env(),
            objective_service=svc,
            activation=activation,
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            capture_only=True,
            db_path=tmp_path / "live.db",
        )
        await runner.start(
            fake_model_provider=fake,
            start_cognitive_agent=False,
            start_evolution_workers=False,
            start_market_loop=False,
        )
        sid = runner.session_id
        await runner.shutdown()
        return sid

    assert await _start() == expected
    assert await _start() == expected
    assert expected.startswith("cog:live:")


@pytest.mark.asyncio
async def test_activation_capital_must_match_objective(tmp_path) -> None:
    svc, definition = await _confirmed_objective(tmp_path, capital=Decimal("500"))
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=definition.objective_id,
        authorized_capital_usd=Decimal("999"),
    )
    runner = LiveTradingRunner(
        app_settings=_live_app(tmp_path),
        env=live_env(),
        objective_service=svc,
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        capture_only=True,
        db_path=tmp_path / "live.db",
    )
    with pytest.raises(LiveTradingError, match="authorized_capital_usd"):
        await runner.start(
            fake_model_provider=FakeModelProvider(available=True),
            start_cognitive_agent=False,
            start_evolution_workers=False,
            start_market_loop=False,
        )


def test_pnl_baseline_uses_exchange_trading_date(tmp_path, monkeypatch) -> None:
    fixed = date(2026, 7, 15)
    monkeypatch.setattr(
        "joker.runtime.cognitive_session.exchange_trading_date",
        lambda **kwargs: fixed,
    )
    client, _, _ = make_live_client(tmp_path)
    client._session_id = "sess-baseline"
    _ = client.get_account_truth()
    rows = sqlite3.connect(client._baseline_store._db_path).execute(  # type: ignore[union-attr]
        "SELECT trading_date FROM session_pnl_baseline"
    ).fetchall()
    assert any(r[0] == fixed.isoformat() for r in rows), rows


def test_preflight_not_ready_without_current_snapshot(tmp_path) -> None:
    apply_task1_migrations(tmp_path / "live.db")
    app = _live_app(tmp_path)
    env = live_env(WEBULL_MARKET_DATA_ENABLED=True, WEBULL_APP_KEY="k")
    with patch("joker.runtime.live_preflight.SystemExchangeClock") as clock_cls:
        clock = MagicMock()
        clock.session_phase.return_value = SessionPhase.REGULAR
        clock.trading_date.return_value = date(2026, 7, 1)
        clock_cls.return_value = clock
        report = run_production_preflight(
            app_settings=app,
            env=env,
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            check_market_data=True,
        )
    assert report.market_session_open is True
    assert report.live_snapshot_ok is False
    assert report.operational_ready is False
    assert report.ok is False
    assert any("live_snapshot: fail" in c for c in report.checks)


def test_preflight_rejects_stale_or_non_0dte_surface(tmp_path) -> None:
    """Non-0DTE linked surface fails the typed readiness chain."""
    from joker.market.option_surface import OptionContractSnapshot, OptionSurfaceSnapshot
    from joker.market.quality import DataQualityReport, DataQualitySeverity
    from joker.market.snapshots import MarketSnapshot, UnderlyingSnapshot

    db = tmp_path / "live.db"
    apply_task1_migrations(db)
    trading_day = date(2026, 7, 1)
    now = datetime.now(timezone.utc)
    surface_id = uuid4()
    dq_id = uuid4()
    snap_id = uuid4()
    contract = OptionContractSnapshot(
        contract_id="SPY260702C00500000",
        symbol="SPY",
        expiry=date(2026, 7, 2),
        strike=Decimal("500"),
        option_type="call",
        bid=Decimal("1.00"),
        ask=Decimal("1.10"),
        quote_timestamp=now,
        quote_age_ms=0,
    )
    surface = OptionSurfaceSnapshot(
        surface_id=surface_id,
        exchange_time=now,
        trading_date=trading_day,
        underlying_symbol="SPY",
        contracts=(contract,),
    )
    quality = DataQualityReport(
        report_id=dq_id,
        snapshot_id=snap_id,
        severity=DataQualitySeverity.OK,
        usable_for_execution=True,
    )
    snapshot = MarketSnapshot(
        snapshot_id=snap_id,
        exchange_time=now,
        trading_date=trading_day,
        underlying=UnderlyingSnapshot(
            symbol="SPY", exchange_time=now, last=Decimal("500"), source="t"
        ),
        option_surface_id=surface_id,
        data_quality_id=dq_id,
    )
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO market_snapshots VALUES (?, ?, ?, ?, ?)",
        (
            str(snap_id),
            trading_day.isoformat(),
            now.isoformat(),
            snapshot.model_dump_json(),
            now.isoformat(),
        ),
    )
    conn.execute(
        "INSERT INTO option_surfaces VALUES (?, ?, ?, ?, ?, ?)",
        (
            str(surface_id),
            trading_day.isoformat(),
            now.isoformat(),
            "SPY",
            surface.model_dump_json(),
            now.isoformat(),
        ),
    )
    conn.execute(
        """
        INSERT INTO data_quality_reports
            (report_id, snapshot_id, session_id, severity,
             usable_for_reasoning, usable_for_execution, payload, created_at)
        VALUES (?, ?, ?, ?, 1, 1, ?, ?)
        """,
        (
            str(dq_id),
            str(snap_id),
            "t",
            "ok",
            quality.model_dump_json(),
            now.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    app = _live_app(tmp_path)
    env = live_env(WEBULL_MARKET_DATA_ENABLED=True, WEBULL_APP_KEY="k")
    with patch("joker.runtime.live_preflight.SystemExchangeClock") as clock_cls:
        clock = MagicMock()
        clock.session_phase.return_value = SessionPhase.REGULAR
        clock.trading_date.return_value = trading_day
        clock_cls.return_value = clock
        report = run_production_preflight(
            app_settings=app,
            env=env,
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            check_market_data=True,
        )
    assert report.current_0dte_surface_ok is False
    assert report.operational_ready is False
    assert any("0DTE" in c or "0dte" in c.lower() for c in report.checks)

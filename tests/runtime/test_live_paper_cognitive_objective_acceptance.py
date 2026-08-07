"""Actual LivePaperRunner cognitive acceptance with objective persistence."""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import aiosqlite
import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.risk.capital import CapitalBudget, CapitalPlan
from joker.config.settings import AppSettings, EnvSettings
from joker.data.webull_api import MockWebullMarketApi, WebullCandle, WebullQuote
from joker.data.webull_options_api import MockWebullOptionsMarketApi
from joker.objectives.projector import ObjectiveCapitalProjector
from joker.objectives.repository import ObjectiveRepository
from joker.objectives.service import SessionObjectiveService
from joker.persistence.aiosqlite_lifecycle import drain_aiosqlite_workers
from joker.runtime.compatibility import CompatibilityLivePaperBridge
from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner
from joker.runtime.session_supervisor import SessionSupervisor
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot
from tests.cognitive.live_acceptance_canned import register_request_bound_canned

ET = ZoneInfo("America/New_York")
SESSION_ID = "cog:paper:local_paper:live-accept-2026-08-07"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _env() -> EnvSettings:
    return EnvSettings(  # type: ignore[arg-type]
        _env_file=None,
        OPENAI_API_KEY="sk-test-key-for-unit-tests-only",
        OPENAI_MODEL="gpt-5.4-mini",
        WEBULL_APP_KEY="k",
        WEBULL_APP_SECRET="s",
        WEBULL_MARKET_DATA_ENABLED=True,
        WEBULL_LIVE_TRADING_ENABLED=False,
        WEBULL_PAPER_TRADING_ENABLED=False,
        WEBULL_ACCESS_TOKEN="tok",
    )


def _candles(n: int = 12) -> list[WebullCandle]:
    base = _now() - timedelta(minutes=n)
    out: list[WebullCandle] = []
    price = 550.0
    for i in range(n):
        ts = base + timedelta(minutes=i)
        out.append(
            WebullCandle(
                timestamp=ts,
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price + 0.2,
                volume=1000,
            )
        )
        price += 0.3
    return out


def _quote() -> WebullQuote:
    return WebullQuote(
        symbol="SPY",
        price=553.0,
        bid=552.9,
        ask=553.1,
        timestamp=_now() - timedelta(minutes=10),
        delayed=True,
    )


def _option_contract(option_type: str, strike: float) -> OptionContractMetadata:
    today = date.today()
    return OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=today,
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        contract_id=(
            f"SPY{today.strftime('%y%m%d')}"
            f"{'C' if option_type == 'call' else 'P'}"
            f"{int(strike * 1000):08d}"
        ),
        source="webull_opra",
    )


def _option_snap(option_type: str, strike: float) -> OptionSnapshot:
    contract = _option_contract(option_type, strike)
    return OptionSnapshot(
        contract=contract,
        bid=1.0,
        ask=1.1,
        mid=1.05,
        spread_pct=9.5,
        quote_timestamp=_now() - timedelta(seconds=2),
        delayed=True,
        source="webull_opra",
        is_synthetic=False,
    )


def _app(tmp_path: Path) -> AppSettings:
    app = AppSettings.model_validate(
        {
            "mode": "PAPER",
            "live_trading_enabled": False,
            "db_path": str(tmp_path / "joker.db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
            "data_dir": str(tmp_path),
            "agents": {
                "runtime": "cognitive_graph",
                "mock_agents": True,
                "intraday_enabled": False,
                "decision_interval_seconds": 1.0,
            },
            "data": {"default_provider": "webull", "quote_poll_interval_seconds": 0.5},
            "risk": {
                "allow_delayed_quotes": True,
                "feed_max_silence_seconds": 120,
                "delayed_quote_max_age_seconds": 900,
                "quote_max_age_seconds": 120,
                "max_premium_usd": 500,
                "kill_switch": False,
            },
            "data_quality": {
                "option_stale_seconds": 3600,
                "maximum_relative_spread": 0.50,
            },
            "evolution": {"enabled": False},
            "objective": {
                "enabled": True,
                "policy": "target_attainment",
                "require_positive_expected_value": True,
                "target_attainment": {"enabled": True, "minimum_calibrated_samples": 1},
            },
            "full_chain_optimizer": {
                "enabled": True,
                "maximum_quote_age_seconds": 3600,
                "maximum_surface_age_seconds": 3600,
                "maximum_decision_age_seconds": 90,
                "maximum_relative_spread": 0.50,
            },
            "cognitive_graph": {"enabled": True},
        }
    )
    return app


def test_live_paper_runner_binds_projector_before_agent_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Executable proof: bind → projector → recover/readiness → start_agent."""
    monkeypatch.setenv("WEBULL_MARKET_DATA_ENABLED", "true")
    monkeypatch.setenv("WEBULL_PAPER_TRADING_ENABLED", "false")
    monkeypatch.setenv("WEBULL_LIVE_TRADING_ENABLED", "false")
    monkeypatch.delenv("WEBULL_PAPER_ACCOUNT_ID", raising=False)
    from joker.data import webull_capability

    monkeypatch.setattr(webull_capability, "capability_usable_for_shadow", lambda: True)

    task1_db = tmp_path / "joker_task1.db"
    repo = ObjectiveRepository(task1_db)
    svc = SessionObjectiveService(
        repo,
        exchange_tz="America/New_York",
        objective_policy="target_attainment",
        require_positive_expected_value=True,
    )

    async def _arm() -> None:
        definition = await svc.create_objective(
            session_id=SESSION_ID,
            authorised_capital_usd=500,
            target_profit_pct=20,
            deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
            max_concurrent_positions=1,
            accepted_total_loss_risk=True,
        )
        await svc.confirm_objective(definition.objective_id)

    asyncio.run(_arm())

    timeline: list[str] = []
    real_bind = SessionSupervisor.bind_objective_service

    def _tracking_bind(self: SessionSupervisor, objective_service: Any) -> None:
        timeline.append("bind_objective_service")
        real_bind(self, objective_service)
        projector = getattr(self, "_objective_projector", None)
        assert isinstance(projector, ObjectiveCapitalProjector)
        assert self.objective_service is objective_service
        timeline.append("projector_installed")

    monkeypatch.setattr(SessionSupervisor, "bind_objective_service", _tracking_bind)

    from joker.runtime import objective_recovery as objective_recovery_mod

    real_recover = objective_recovery_mod.recover_session_objective

    async def _tracking_recover(*args: Any, **kwargs: Any) -> Any:
        timeline.append("recover_session_objective")
        return await real_recover(*args, **kwargs)

    monkeypatch.setattr(
        objective_recovery_mod, "recover_session_objective", _tracking_recover
    )

    real_recompute = SessionObjectiveService.recompute_from_truth

    async def _tracking_recompute(self: SessionObjectiveService, *a: Any, **k: Any) -> Any:
        # First readiness recompute after recover is the cognitive startup check.
        if "readiness_recompute" not in timeline and "recover_session_objective" in timeline:
            timeline.append("readiness_recompute")
        return await real_recompute(self, *a, **k)

    monkeypatch.setattr(
        SessionObjectiveService, "recompute_from_truth", _tracking_recompute
    )

    real_start = CompatibilityLivePaperBridge.start_agent

    def _tracking_start(self: CompatibilityLivePaperBridge) -> None:
        timeline.append("start_agent")
        assert self.supervisor.objective_service is svc
        assert isinstance(
            getattr(self.supervisor, "_objective_projector", None),
            ObjectiveCapitalProjector,
        )
        return real_start(self)

    monkeypatch.setattr(CompatibilityLivePaperBridge, "start_agent", _tracking_start)

    quote = _quote()
    call = _option_snap("call", 553.0)
    put = _option_snap("put", 553.0)
    runner = LivePaperRunner(_app(tmp_path), _env())
    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=2.0,
            mock_agents=True,
            require_options=True,
            webull_api=MockWebullMarketApi(
                quote=quote, candles=_candles(), stream_quotes=[quote, quote]
            ),
            webull_options_api=MockWebullOptionsMarketApi(
                contracts=[call.contract, put.contract],
                snapshots={
                    call.contract.contract_id: call,
                    put.contract.contract_id: put,
                },
            ),
            broker=PaperBroker(initial_balance=25000.0, slippage_pct=0.0),
            objective_service=svc,
            cognitive_session_id_override=SESSION_ID,
            capital_budget=CapitalBudget(
                plan=CapitalPlan(
                    authorized_usd=500.0,
                    target_profit_pct=20.0,
                    max_concurrent_positions=1,
                    max_contracts_per_trade=5,
                    min_contracts_per_trade=1,
                )
            ),
        )
    )
    asyncio.run(drain_aiosqlite_workers(timeout=1.0))

    assert "bind_objective_service" in timeline
    assert "projector_installed" in timeline
    assert "recover_session_objective" in timeline
    assert "readiness_recompute" in timeline
    assert "start_agent" in timeline
    bind_i = timeline.index("bind_objective_service")
    proj_i = timeline.index("projector_installed")
    recover_i = timeline.index("recover_session_objective")
    ready_i = timeline.index("readiness_recompute")
    start_i = timeline.index("start_agent")
    assert bind_i < proj_i < recover_i < ready_i < start_i
    assert not any("objective_unavailable" in e for e in (result.errors + result.failures))


def test_live_paper_runner_normal_cognitive_path_under_contention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real LivePaperRunner cognitive cycle with shared Task-1 writers."""
    monkeypatch.setenv("WEBULL_MARKET_DATA_ENABLED", "true")
    monkeypatch.setenv("WEBULL_PAPER_TRADING_ENABLED", "false")
    monkeypatch.setenv("WEBULL_LIVE_TRADING_ENABLED", "false")
    monkeypatch.delenv("WEBULL_PAPER_ACCOUNT_ID", raising=False)
    from joker.data import webull_capability

    monkeypatch.setattr(webull_capability, "capability_usable_for_shadow", lambda: True)

    task1_db = tmp_path / "joker_task1.db"
    repo = ObjectiveRepository(task1_db)
    svc = SessionObjectiveService(
        repo,
        exchange_tz="America/New_York",
        objective_policy="target_attainment",
        require_positive_expected_value=True,
    )

    async def _arm() -> None:
        definition = await svc.create_objective(
            session_id=SESSION_ID,
            authorised_capital_usd=500,
            target_profit_pct=20,
            deadline_exchange_time=datetime.now(tz=ET) + timedelta(hours=2),
            max_concurrent_positions=1,
            accepted_total_loss_risk=True,
        )
        await svc.confirm_objective(definition.objective_id)

    asyncio.run(_arm())

    call = _option_snap("call", 553.0)
    put = _option_snap("put", 553.0)
    contract_id = call.contract.contract_id

    from joker.persistence import aiosqlite_lifecycle as aiosqlite_lifecycle_mod
    from joker.runtime import cognitive_startup as cognitive_startup_mod

    real_drain = aiosqlite_lifecycle_mod.drain_aiosqlite_workers

    async def _drain_bounded(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", 2.0)
        return await real_drain(*args, **kwargs)

    monkeypatch.setattr(
        aiosqlite_lifecycle_mod, "drain_aiosqlite_workers", _drain_bounded
    )

    real_validate = cognitive_startup_mod.validate_cognitive_providers

    async def _validate_and_can(*args: Any, **kwargs: Any) -> Any:
        result = await real_validate(*args, **kwargs)
        fake = result.registry.get_provider("fake")
        register_request_bound_canned(
            fake, session=SESSION_ID, contract_id=contract_id
        )
        return result

    monkeypatch.setattr(
        cognitive_startup_mod, "validate_cognitive_providers", _validate_and_can
    )

    timeline: list[str] = []
    real_bind = SessionSupervisor.bind_objective_service

    def _tracking_bind(self: SessionSupervisor, objective_service: Any) -> None:
        timeline.append("bind")
        real_bind(self, objective_service)
        assert self.objective_service is svc
        assert isinstance(
            getattr(self, "_objective_projector", None), ObjectiveCapitalProjector
        )
        timeline.append("projector")

    monkeypatch.setattr(SessionSupervisor, "bind_objective_service", _tracking_bind)

    real_start = CompatibilityLivePaperBridge.start_agent
    observed: list[str] = []
    stop_writers = threading.Event()
    writers_seen = threading.Event()

    preferred = {
        "chain.universe.built",
        "contract.grid.scored",
        "portfolio.grid.scored",
        "debate.review.completed",
        "target.wait.selected",
        "target.portfolio.selected",
        "graph.cycle.completed",
    }

    def _on_event(event_type: str, _payload: dict[str, Any]) -> None:
        observed.append(event_type)

    submit_count = {"n": 0}
    broker = PaperBroker(initial_balance=25000.0, slippage_pct=0.0)
    real_submit = broker.submit_order

    def _count_submit(intent):  # type: ignore[no-untyped-def]
        submit_count["n"] += 1
        return real_submit(intent)

    broker.submit_order = _count_submit  # type: ignore[method-assign]

    def _writer_thread() -> None:
        """Contend during objective gate / early persistence, then stop.

        Long-lived BEGIN IMMEDIATE writers across the full ~40s canned cycle
        starve bridge ``run_coro`` pumping and hang shutdown. The dedicated
        downstream contention suite covers sustained IMMEDIATE load; here we
        keep a real shared Task-1 writer overlapping startup + early graph.
        """
        writers_seen.set()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            end = time.monotonic() + 12.0
            while not stop_writers.is_set() and time.monotonic() < end:
                try:
                    async with aiosqlite.connect(task1_db) as conn:
                        await conn.execute("PRAGMA busy_timeout = 100")
                        await conn.execute("BEGIN IMMEDIATE")
                        await conn.execute(
                            "CREATE TABLE IF NOT EXISTS accept_noise(id INTEGER)"
                        )
                        await conn.execute("INSERT INTO accept_noise(id) VALUES (1)")
                        await conn.commit()
                except Exception:
                    pass
                await asyncio.sleep(0.04)

        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    writer = threading.Thread(target=_writer_thread, name="task1-contention", daemon=True)

    def _tracking_start(self: CompatibilityLivePaperBridge) -> None:
        assert "bind" in timeline and "projector" in timeline
        timeline.append("start_agent")
        real_start(self)
        if not writer.is_alive():
            writer.start()

    monkeypatch.setattr(CompatibilityLivePaperBridge, "start_agent", _tracking_start)

    quote = _quote()
    app = _app(tmp_path)
    assert app.mode is SafetyMode.PAPER
    assert app.live_trading_enabled is False
    runner = LivePaperRunner(app, _env())
    try:
        result = runner.run(
            LivePaperRunConfig(
                duration_seconds=55.0,
                mock_agents=True,
                require_options=True,
                webull_api=MockWebullMarketApi(
                    quote=quote,
                    candles=_candles(),
                    stream_quotes=[quote] * 40,
                ),
                webull_options_api=MockWebullOptionsMarketApi(
                    contracts=[call.contract, put.contract],
                    snapshots={
                        call.contract.contract_id: call,
                        put.contract.contract_id: put,
                    },
                ),
                broker=broker,
                objective_service=svc,
                cognitive_session_id_override=SESSION_ID,
                capital_budget=CapitalBudget(
                    plan=CapitalPlan(
                        authorized_usd=500.0,
                        target_profit_pct=20.0,
                        max_concurrent_positions=1,
                        max_contracts_per_trade=5,
                        min_contracts_per_trade=1,
                    )
                ),
            ),
            on_event=_on_event,
        )
    finally:
        stop_writers.set()
        writer.join(timeout=3.0)
        asyncio.run(drain_aiosqlite_workers(timeout=1.0))

    assert writers_seen.is_set(), "competing Task-1 writer never started"
    assert timeline.index("bind") < timeline.index("projector") < timeline.index(
        "start_agent"
    )
    assert "strategy.thesis.generated" in observed, (
        f"missing thesis event; observed={observed[:40]} errors={result.errors}"
    )
    assert preferred.intersection(observed), (
        f"thesis observed but no downstream cognitive evidence: {observed}"
    )
    assert "graph.cycle.started" in observed

    joined_errors = " ".join(result.errors + result.failures).lower()
    assert "objective_unavailable" not in joined_errors
    assert "database is locked" not in joined_errors
    assert submit_count["n"] == 0, "external/local broker submission must not occur"
    assert result.broker_kind in {"local_paper", "paper"} or "paper" in (
        result.broker_label or ""
    ).lower()

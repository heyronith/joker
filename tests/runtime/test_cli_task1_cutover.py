"""CLI paper path reaches SessionSupervisor / MarketRuntime / ExecutionRuntime / ledger."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from joker.broker.webull import WebullClient
from joker.broker.webull_trade_api import MockWebullTradeApi
from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings, EnvSettings
from joker.runtime.compatibility import CompatibilityLivePaperBridge, ExecutionDelegatingBroker
from joker.runtime.execution_runtime import ExecutionRuntime
from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner
from joker.runtime.market_runtime import MarketRuntime
from joker.runtime.session_supervisor import SessionSupervisor
from joker.schemas.domain import BrokerOrder, OptionContract, OrderIntent


def _webull_recovery_runner(tmp_path: Path) -> tuple[LivePaperRunner, MockWebullTradeApi]:
    app = AppSettings(db_path=str(tmp_path / "app.db"))
    app = app.model_copy(
        update={
            "broker": app.broker.model_copy(update={"provider": "webull_paper"}),
            "agents": app.agents.model_copy(update={"runtime": "cognitive_graph"}),
        }
    )
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="test-ci-key-not-real",
        WEBULL_MARKET_DATA_ENABLED=False,
        WEBULL_LIVE_TRADING_ENABLED=False,
        WEBULL_PAPER_TRADING_ENABLED=True,
        WEBULL_PAPER_ACCOUNT_ID="PAPER_ACCT_1",
        WEBULL_APP_KEY="paper-key",
        WEBULL_APP_SECRET="paper-secret",
        WEBULL_ACCESS_TOKEN="paper-token",
    )
    return LivePaperRunner(app, env), MockWebullTradeApi(account_id="PAPER_ACCT_1")


def test_paper_path_task1_cutover_components(tmp_path: Path) -> None:
    """Prove the paper session wiring reaches Task 1 runtimes and ledger."""
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker,
        db_path=tmp_path / "cli_task1.db",
        session_id="cli-sess",
        run_id="cli-run",
    )
    bridge.start()
    try:
        assert isinstance(bridge.supervisor, SessionSupervisor)
        assert isinstance(bridge.market_runtime, MarketRuntime)
        assert isinstance(bridge.execution_runtime, ExecutionRuntime)
        assert bridge.supervisor.ledger_store is not None

        obs = bridge.ingest_underlying_quote(
            symbol="SPY",
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            source="cli_test",
        )
        assert obs.symbol == "SPY"

        contract = OptionContract(
            symbol="SPY",
            expiration=date(2026, 7, 1),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        )
        intent = OrderIntent(
            candidate_id="cli",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.0,
        )
        wrapped = ExecutionDelegatingBroker(inner=broker, bridge=bridge)
        order = wrapped.submit_order(intent)
        assert order.order_id
        projected = bridge.project_session()
        assert intent.intent_id in projected.orders
        events = bridge.run_coro(
            bridge.supervisor.ledger_store.get_by_session("cli-sess")  # type: ignore[union-attr]
        )
        assert any(
            e.event_type.value.endswith("fill") or "submission" in e.event_type.value
            for e in events
        )
    finally:
        bridge.shutdown()


def test_live_paper_runner_constructs_bridge_on_run_start(tmp_path: Path, monkeypatch) -> None:
    """LivePaperRunner.run instantiates CompatibilityLivePaperBridge (cutover)."""
    app = AppSettings(db_path=str(tmp_path / "app.db"))
    # EnvSettings fields use aliases; CI has no .env — set env vars + construct explicitly.
    monkeypatch.setenv("OPENAI_API_KEY", "test-ci-key-not-real")
    monkeypatch.setenv("WEBULL_MARKET_DATA_ENABLED", "true")
    monkeypatch.setenv("WEBULL_LIVE_TRADING_ENABLED", "false")
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="test-ci-key-not-real",
        WEBULL_MARKET_DATA_ENABLED=True,
        WEBULL_LIVE_TRADING_ENABLED=False,
    )
    runner = LivePaperRunner(app, env)

    created: dict[str, CompatibilityLivePaperBridge] = {}
    real_bridge_cls = CompatibilityLivePaperBridge

    def tracking_bridge(*args, **kwargs):
        bridge = real_bridge_cls(*args, **kwargs)
        created["bridge"] = bridge
        return bridge

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.CompatibilityLivePaperBridge",
        tracking_bridge,
    )
    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.resolve_live_paper_broker",
        lambda *a, **k: MagicMock(
            client=PaperBroker(slippage_pct=0),
            kind="local_paper",
            label="paper",
            auto_orders=True,
        ),
    )

    class BoomProvider:
        def __init__(self, *a, **k):
            pass

        def authenticate(self):
            return False

        def close(self):
            return None

    # Shared market loop constructs the stock provider; patch at that import site.
    monkeypatch.setattr(
        "joker.runtime.live_market_data_loop.WebullMarketDataProvider",
        BoomProvider,
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=1,
            mock_agents=True,
            require_options=False,
            approved_playbook=None,
            broker=PaperBroker(slippage_pct=0),
        )
    )
    assert "bridge" in created
    assert isinstance(created["bridge"].supervisor, SessionSupervisor)
    assert runner.task1_bridge is None
    assert result.errors


def test_reconciliation_only_recovery_starts_without_option_surface(
    tmp_path: Path, monkeypatch
) -> None:
    app = AppSettings(db_path=str(tmp_path / "app.db"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-ci-key-not-real")
    monkeypatch.setenv("WEBULL_MARKET_DATA_ENABLED", "true")
    monkeypatch.setenv("WEBULL_LIVE_TRADING_ENABLED", "false")
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="test-ci-key-not-real",
        WEBULL_MARKET_DATA_ENABLED=True,
        WEBULL_LIVE_TRADING_ENABLED=False,
    )
    runner = LivePaperRunner(app, env)
    broker = PaperBroker(slippage_pct=0)

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.resolve_live_paper_broker",
        lambda *a, **k: MagicMock(
            client=broker,
            kind="local_paper",
            label="paper",
            auto_orders=True,
        ),
    )

    class ForbiddenMarketLoop:
        def __init__(self, *args, **kwargs):
            raise AssertionError("market loop must not start during reconciliation-only recovery")

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.LiveMarketDataLoop",
        ForbiddenMarketLoop,
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=True,
            reconciliation_only_recovery=True,
            broker=broker,
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"
    assert result.options_available is False


def test_broker_only_runner_does_not_call_validate_cognitive_providers(
    tmp_path: Path, monkeypatch
) -> None:
    runner, api = _webull_recovery_runner(tmp_path)

    monkeypatch.setattr(
        "joker.runtime.cognitive_startup.validate_cognitive_providers",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cognitive provider validation must be skipped")
        ),
    )
    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.LiveMarketDataLoop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("market loop must not start during broker-only recovery")
        ),
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=True,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override="cog:paper:webull:test:2026-08-05",
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"
    assert result.broker_kind == "webull_paper"


def test_broker_only_runner_does_not_construct_model_router(
    tmp_path: Path, monkeypatch
) -> None:
    runner, api = _webull_recovery_runner(tmp_path)

    monkeypatch.setattr(
        "joker.models.router.ModelRouter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ModelRouter must not be constructed for broker-only recovery")
        ),
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=True,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override="cog:paper:webull:test:2026-08-05",
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"
    assert result.broker_kind == "webull_paper"


def test_broker_only_runner_does_not_start_cognitive_agent_or_resume_cycles(
    tmp_path: Path, monkeypatch
) -> None:
    runner, api = _webull_recovery_runner(tmp_path)

    monkeypatch.setattr(
        "joker.runtime.session_supervisor.SessionSupervisor.start_agent_runtime",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("broker-only recovery must not start the cognitive agent")
        ),
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=True,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override="cog:paper:webull:test:2026-08-05",
        )
    )
    assert result.feed_health == "RECOVERY_ONLY"
    assert result.broker_kind == "webull_paper"


def test_broker_only_runner_polls_working_webull_paper_order(
    tmp_path: Path, monkeypatch
) -> None:
    runner, api = _webull_recovery_runner(tmp_path)
    observed: dict[str, list[str]] = {"polled": [], "statuses": []}
    real_bridge = CompatibilityLivePaperBridge

    class TrackingBridge(real_bridge):
        def start(self, *, start_agent: bool = True) -> None:
            super().start(start_agent=start_agent)
            broker = self.supervisor.execution_runtime._broker  # type: ignore[attr-defined]
            assert isinstance(broker, WebullClient)
            self._recovery_polled = False
            contract = OptionContract(
                symbol="SPY",
                expiration=date(2026, 8, 5),
                strike=500.0,
                option_type="call",
                is_0dte=True,
            )
            intent = OrderIntent(
                intent_id="wb-recovery-1",
                candidate_id="recovery",
                contract=contract,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=1.0,
            )
            broker._intent_by_order["wb-recovery-1"] = intent  # type: ignore[attr-defined]
            seeded = BrokerOrder(
                order_id="wb-recovery-1",
                intent_id="wb-recovery-1",
                status="open",
                contract=contract,
                side="buy",
                quantity=1,
                limit_price=1.0,
            )
            broker._orders["wb-recovery-1"] = seeded  # type: ignore[attr-defined]
            api._orders["wb-recovery-1"] = {
                "client_order_id": "wb-recovery-1",
                "order_id": "wb-recovery-1",
                "status": "OPEN",
                "symbol": "SPY",
                "side": "BUY",
                "quantity": "1",
                "limit_price": "1.0",
                "instrument_type": "OPTION",
                "legs": [
                    {
                        "symbol": "SPY",
                        "option_type": "CALL",
                        "option_expire_date": "2026-08-05",
                        "strike_price": "500",
                    }
                ],
            }
            self.execution_runtime._client_to_broker["wb-recovery-1"] = "wb-recovery-1"  # type: ignore[index]

        def project_session(self):
            if getattr(self, "_recovery_polled", False):
                return SimpleNamespace(orders=[], positions={})
            return SimpleNamespace(
                orders=[
                    SimpleNamespace(
                        client_order_id="wb-recovery-1",
                        status="accepted",
                    )
                ],
                positions={},
            )

        def poll_order_status(self, client_order_id: str):
            self._recovery_polled = True
            observed["polled"].append(client_order_id)
            order = super().poll_order_status(client_order_id)
            observed["statuses"].append(getattr(order, "status", "missing"))
            return order

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.CompatibilityLivePaperBridge",
        TrackingBridge,
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=True,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override="cog:paper:webull:test:2026-08-05",
        )
    )
    assert observed["polled"]
    assert observed["statuses"]
    assert result.broker_kind == "webull_paper"


@pytest.mark.parametrize("final_status", ["FILLED", "REJECTED"])
def test_broker_only_runner_processes_terminal_webull_updates(
    tmp_path: Path, monkeypatch, final_status: str
) -> None:
    runner, api = _webull_recovery_runner(tmp_path)
    observed: list[str] = []
    real_bridge = CompatibilityLivePaperBridge

    class TrackingBridge(real_bridge):
        def start(self, *, start_agent: bool = True) -> None:
            super().start(start_agent=start_agent)
            broker = self.supervisor.execution_runtime._broker  # type: ignore[attr-defined]
            assert isinstance(broker, WebullClient)
            self._recovery_polled = False
            contract = OptionContract(
                symbol="SPY",
                expiration=date(2026, 8, 5),
                strike=500.0,
                option_type="call",
                is_0dte=True,
            )
            intent = OrderIntent(
                intent_id="wb-recovery-2",
                candidate_id="recovery",
                contract=contract,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=1.0,
            )
            broker._intent_by_order["wb-recovery-2"] = intent  # type: ignore[attr-defined]
            seeded = BrokerOrder(
                order_id="wb-recovery-2",
                intent_id="wb-recovery-2",
                status="open",
                contract=contract,
                side="buy",
                quantity=1,
                limit_price=1.0,
            )
            broker._orders["wb-recovery-2"] = seeded  # type: ignore[attr-defined]
            api._orders["wb-recovery-2"] = {
                "client_order_id": "wb-recovery-2",
                "order_id": "wb-recovery-2",
                "status": final_status,
                "symbol": "SPY",
                "side": "BUY",
                "quantity": "1",
                "limit_price": "1.0",
                "instrument_type": "OPTION",
                "legs": [
                    {
                        "symbol": "SPY",
                        "option_type": "CALL",
                        "option_expire_date": "2026-08-05",
                        "strike_price": "500",
                    }
                ],
            }
            self.execution_runtime._client_to_broker["wb-recovery-2"] = "wb-recovery-2"  # type: ignore[index]

        def project_session(self):
            if getattr(self, "_recovery_polled", False):
                return SimpleNamespace(orders=[], positions={})
            return SimpleNamespace(
                orders=[
                    SimpleNamespace(
                        client_order_id="wb-recovery-2",
                        status="accepted",
                    )
                ],
                positions={},
            )

        def poll_order_status(self, client_order_id: str):
            self._recovery_polled = True
            order = super().poll_order_status(client_order_id)
            observed.append(getattr(order, "status", "missing"))
            return order

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.CompatibilityLivePaperBridge",
        TrackingBridge,
    )

    result = runner.run(
        LivePaperRunConfig(
            duration_seconds=0.1,
            mock_agents=True,
            require_options=True,
            reconciliation_only_recovery=True,
            trade_api=api,
            cognitive_session_id_override="cog:paper:webull:test:2026-08-05",
        )
    )
    assert observed
    assert final_status.lower() in observed
    assert result.working_orders_remaining == 0


def test_broker_only_runner_never_falls_back_to_local_paper(
    tmp_path: Path, monkeypatch
) -> None:
    runner, api = _webull_recovery_runner(tmp_path)
    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.resolve_live_paper_broker",
        lambda *args, **kwargs: SimpleNamespace(
            client=PaperBroker(slippage_pct=0),
            kind="local_paper",
            label="paper",
            auto_orders=True,
        ),
    )

    with pytest.raises(RuntimeError, match="refusing PaperBroker fallback|non-Webull broker"):
        runner.run(
            LivePaperRunConfig(
                duration_seconds=0.1,
                mock_agents=True,
                require_options=True,
                reconciliation_only_recovery=True,
                trade_api=api,
                cognitive_session_id_override="cog:paper:webull:test:2026-08-05",
            )
        )

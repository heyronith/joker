"""CLI paper path reaches SessionSupervisor / MarketRuntime / ExecutionRuntime / ledger."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings, EnvSettings
from joker.runtime.compatibility import CompatibilityLivePaperBridge, ExecutionDelegatingBroker
from joker.runtime.execution_runtime import ExecutionCommand, ExecutionRuntime
from joker.runtime.live_paper_runner import LivePaperRunConfig, LivePaperRunner
from joker.runtime.market_runtime import MarketRuntime
from joker.runtime.session_supervisor import SessionSupervisor
from joker.schemas.domain import OptionContract, OrderIntent, Playbook


def test_paper_path_task1_cutover_components(tmp_path: Path, monkeypatch) -> None:
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

        # Market path
        obs = bridge.ingest_underlying_quote(
            symbol="SPY",
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            source="cli_test",
        )
        assert obs.symbol == "SPY"

        # Execution path → append-only ledger
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
        assert any(e.event_type.value.endswith("fill") or "submission" in e.event_type.value for e in events)
    finally:
        bridge.shutdown()


def test_live_paper_runner_constructs_bridge_on_run_start(tmp_path: Path, monkeypatch) -> None:
    """LivePaperRunner.run instantiates CompatibilityLivePaperBridge (cutover)."""
    app = AppSettings(db_path=str(tmp_path / "app.db"))
    env = EnvSettings(
        webull_market_data_enabled=True,
        webull_live_trading_enabled=False,
    )
    # Bypass safety / auth by forcing early failure after bridge start.
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

    # Force auth failure after bridge starts.
    class BoomProvider:
        def __init__(self, *a, **k):
            pass

        def authenticate(self):
            return False

    monkeypatch.setattr(
        "joker.runtime.live_paper_runner.WebullMarketDataProvider",
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
    # Bridge should be shut down after early return.
    assert runner.task1_bridge is None
    assert result.errors

"""Compatibility bridge delegates to SessionSupervisor runtimes."""

from __future__ import annotations

from pathlib import Path

from joker.broker.interface import PaperBroker
from joker.runtime.compatibility import (
    CompatibilityLivePaperBridge,
    ExecutionDelegatingBroker,
)
from joker.runtime.execution_runtime import ExecutionRuntime
from joker.runtime.live_paper_runner import LivePaperRunner
from joker.runtime.market_runtime import MarketRuntime
from joker.runtime.session_supervisor import SessionSupervisor


def test_bridge_starts_supervisor_market_and_execution(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker,
        db_path=tmp_path / "t1.db",
        session_id="compat-1",
    )
    bridge.start()
    try:
        assert isinstance(bridge.supervisor, SessionSupervisor)
        assert isinstance(bridge.market_runtime, MarketRuntime)
        assert isinstance(bridge.execution_runtime, ExecutionRuntime)
        assert bridge.supervisor.ledger_store is not None
        # Delegating broker routes submit through execution runtime.
        wrapped = ExecutionDelegatingBroker(inner=broker, bridge=bridge)
        assert wrapped.inner is broker
    finally:
        bridge.shutdown()


def test_live_paper_runner_exposes_task1_bridge_attr() -> None:
    # Construction only — full run requires Webull; attribute must exist.
    from joker.config.settings import AppSettings, EnvSettings

    runner = LivePaperRunner(AppSettings(), EnvSettings())
    assert runner.task1_bridge is None
    assert hasattr(runner, "_task1_bridge")

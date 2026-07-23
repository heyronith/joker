"""Phase 11 shadow mode tests."""

from __future__ import annotations

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.risk.governor import RiskGovernor
from joker.runtime.shadow import ShadowRuntime
from joker.schemas.domain import RiskConfig, RiskDecision
from tests.fixtures.domain import make_candidate


def test_shadow_never_submits() -> None:
    runtime = ShadowRuntime(mode=SafetyMode.SHADOW)
    broker = PaperBroker()
    record = runtime.record_candidate(
        make_candidate(),
        RiskDecision(candidate_id="c1", approved=True),
        broker,
    )
    assert len(broker.list_open_orders()) == 0
    assert record.simulated_entry > 0


def test_simulated_outcome() -> None:
    runtime = ShadowRuntime(mode=SafetyMode.SHADOW)
    record = runtime.record_candidate(
        make_candidate(),
        RiskDecision(candidate_id="c1", approved=True),
        PaperBroker(),
    )
    runtime.simulate_outcome(record, exit_price=2.0)
    assert record.simulated_pnl is not None


def test_shadow_mode_label() -> None:
    from joker.tui.state import DashboardState

    state = DashboardState(mode=SafetyMode.SHADOW)
    assert "SHADOW" in state.mode_label()

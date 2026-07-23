"""Phase 9 reactive engine tests."""

from __future__ import annotations

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.risk.governor import RiskGovernor
from joker.runtime.reactive_engine import ReactiveEngine, StateMachineError, TradingState
from joker.schemas.domain import Playbook, PlaybookSetup, RiskConfig
from tests.fixtures.domain import make_candidate, make_daily_state
from datetime import date


def _engine() -> ReactiveEngine:
    config = RiskConfig(
        max_daily_loss_usd=500,
        max_trades_per_day=3,
        max_open_positions=1,
        max_premium_usd=200,
        max_spread_pct=15,
        quote_max_age_seconds=300,
    )
    return ReactiveEngine(RiskGovernor(config, SafetyMode.PAPER), PaperBroker(slippage_pct=0))


def test_valid_transitions() -> None:
    engine = _engine()
    engine.transition(TradingState.WATCHING)
    assert engine.state is TradingState.WATCHING


def test_invalid_transition_rejected() -> None:
    engine = _engine()
    with pytest.raises(StateMachineError):
        engine.transition(TradingState.OPEN_POSITION)


def test_rejected_risk_does_not_create_order() -> None:
    engine = _engine()
    engine.transition(TradingState.WATCHING)
    c = make_candidate(stop_price=0)
    decision = engine.on_signal(c, make_daily_state())
    assert decision.approved is False
    assert len(engine.broker.list_open_orders()) == 0


def test_approved_risk_creates_paper_order() -> None:
    engine = _engine()
    pb = Playbook(
        trading_day=date.today(),
        title="t",
        summary="s",
        setups=[
            PlaybookSetup(
                name="s",
                direction="long_call",
                stop_rule="x",
                take_profit_rule="y",
            )
        ],
        approved=True,
    )
    engine.arm_playbook(pb)
    decision = engine.on_signal(make_candidate(), make_daily_state())
    assert decision.approved is True
    assert len(engine.broker.list_positions()) >= 1


def test_transitions_logged() -> None:
    engine = _engine()
    engine.transition(TradingState.WATCHING)
    assert len(engine.transitions) >= 1

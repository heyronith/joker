"""Kill-switch must reject LivePaperRunner entry route before broker submit."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import BrokerClient
from joker.broker.webull import WebullClient
from joker.risk.governor import RiskGovernor, RiskReasonCode
from joker.runtime.reactive_engine import ReactiveEngine, TradingState
from joker.schemas.domain import (
    BrokerOrder,
    DailyState,
    OrderIntent,
    RiskConfig,
)
from joker.storage.database import Database
from joker.storage.models import (
    FillRecord,
    OrderRecord,
    PositionRecord,
    RiskDecisionRecord,
)
from tests.fixtures.domain import make_candidate


class SpyWebullPaperBroker(WebullClient):
    """Webull paper identity with submit spying — never hits the network."""

    LIVE_CALLS_ENABLED = False

    def __init__(self) -> None:
        # Bypass WebullClient.__init__ env/network setup; keep paper identity.
        self._submit_calls: list[OrderIntent] = []
        self._orders: dict[str, BrokerOrder] = {}
        self._intent_by_order: dict[str, OrderIntent] = {}
        self._account_id = "PAPER_SPY_KILL_SWITCH"

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:  # type: ignore[override]
        self._submit_calls.append(intent)
        pytest.fail("kill-switch must not call broker.submit_order")

    def cancel_order(self, order_id: str) -> BrokerOrder:  # type: ignore[override]
        pytest.fail("kill-switch must not call broker.cancel_order")

    def get_order(self, order_id: str) -> BrokerOrder | None:  # type: ignore[override]
        return self._orders.get(order_id)

    def list_open_orders(self) -> list[BrokerOrder]:  # type: ignore[override]
        return []

    def list_positions(self) -> list:  # type: ignore[override]
        return []

    def get_account_balance(self) -> float:  # type: ignore[override]
        return 50.0

    def get_daily_pnl(self) -> float:  # type: ignore[override]
        return 0.0

    def get_daily_pnl_available(self) -> tuple[bool, float | None]:  # type: ignore[override]
        return True, 0.0


def test_kill_switch_blocks_live_paper_entry_route(tmp_path: Path) -> None:
    broker: BrokerClient = SpyWebullPaperBroker()
    assert isinstance(broker, WebullClient)

    risk = RiskConfig(
        kill_switch=True,
        policy="agent_led",
        max_daily_loss_usd=500,
        max_trades_per_day=5,
        max_open_positions=1,
        max_premium_usd=500,
        max_spread_pct=25,
        quote_max_age_seconds=900,
        allow_delayed_quotes=True,
    )
    engine = ReactiveEngine(
        RiskGovernor(risk, SafetyMode.PAPER, live_enabled=False),
        broker,
    )
    # LivePaperRunner arms IDLE→WATCHING before signal evaluation.
    engine.transition(TradingState.WATCHING)

    db = Database(tmp_path / "kill_switch.db")
    db.initialize()
    run_id = "kill-switch-webull-paper"
    candidate = make_candidate()
    daily = DailyState(
        trading_day=date.today(),
        run_id=run_id,
        mode=SafetyMode.PAPER.value,
        playbook_approved=True,
        kill_switch=False,  # config kill_switch alone must still veto
    )

    decision = engine.evaluate_signal(candidate, daily)
    assert decision.approved is False
    assert RiskReasonCode.KILL_SWITCH in decision.reason_codes
    assert getattr(broker, "_submit_calls") == []

    # Durable audit event — same persistence path LivePaperRunner uses.
    db.save(
        RiskDecisionRecord(
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            approved=decision.approved,
            reason_codes=list(decision.reason_codes),
            payload=decision.model_dump(mode="json"),
        )
    )
    saved = list(db.list_by_run(RiskDecisionRecord, run_id))
    assert len(saved) == 1
    assert saved[0].approved is False
    assert RiskReasonCode.KILL_SWITCH in saved[0].reason_codes

    assert list(db.list_by_run(OrderRecord, run_id)) == []
    assert list(db.list_by_run(FillRecord, run_id)) == []
    assert list(db.list_by_run(PositionRecord, run_id)) == []
    assert broker.list_open_orders() == []
    assert broker.list_positions() == []

    # Must not advance into ENTERING / OPEN_POSITION without approval.
    assert engine.state is TradingState.WATCHING

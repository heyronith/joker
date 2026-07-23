"""Reactive trading state machine."""

from __future__ import annotations

from enum import Enum
from typing import Callable

from joker.broker.interface import BrokerClient
from joker.risk.governor import RiskGovernor
from joker.schemas.domain import (
    BrokerOrder,
    DailyState,
    OrderIntent,
    Playbook,
    RiskDecision,
    TradeCandidate,
)


class TradingState(str, Enum):
    IDLE = "IDLE"
    WATCHING = "WATCHING"
    SETUP_ARMED = "SETUP_ARMED"
    SIGNAL_DETECTED = "SIGNAL_DETECTED"
    RISK_CHECK = "RISK_CHECK"
    ENTERING = "ENTERING"
    OPEN_POSITION = "OPEN_POSITION"
    MANAGING_EXIT = "MANAGING_EXIT"
    EXITED = "EXITED"
    COOLDOWN = "COOLDOWN"
    LOCKED = "LOCKED"


VALID_TRANSITIONS: dict[TradingState, set[TradingState]] = {
    TradingState.IDLE: {TradingState.WATCHING, TradingState.LOCKED},
    TradingState.WATCHING: {TradingState.SETUP_ARMED, TradingState.SIGNAL_DETECTED, TradingState.LOCKED},
    TradingState.SETUP_ARMED: {TradingState.SIGNAL_DETECTED, TradingState.WATCHING, TradingState.LOCKED},
    TradingState.SIGNAL_DETECTED: {TradingState.RISK_CHECK, TradingState.WATCHING},
    TradingState.RISK_CHECK: {TradingState.ENTERING, TradingState.WATCHING, TradingState.COOLDOWN},
    TradingState.ENTERING: {TradingState.OPEN_POSITION, TradingState.WATCHING},
    TradingState.OPEN_POSITION: {TradingState.MANAGING_EXIT, TradingState.EXITED},
    TradingState.MANAGING_EXIT: {TradingState.EXITED, TradingState.OPEN_POSITION},
    TradingState.EXITED: {TradingState.COOLDOWN, TradingState.WATCHING},
    TradingState.COOLDOWN: {TradingState.WATCHING, TradingState.IDLE},
    TradingState.LOCKED: {TradingState.IDLE},
}


class StateMachineError(Exception):
    pass


class ReactiveEngine:
    """Live watcher reacting to market events using active playbook."""

    def __init__(
        self,
        risk_governor: RiskGovernor,
        broker: BrokerClient,
        on_transition: Callable[[TradingState, TradingState, dict], None] | None = None,
        cooldown_seconds: int = 60,
    ) -> None:
        self.risk_governor = risk_governor
        self.broker = broker
        self.on_transition = on_transition
        self.cooldown_seconds = cooldown_seconds
        self.state = TradingState.IDLE
        self.active_playbook: Playbook | None = None
        self._transition_log: list[dict] = []
        self.last_order: BrokerOrder | None = None

    def transition(self, new_state: TradingState, payload: dict | None = None) -> None:
        allowed = VALID_TRANSITIONS.get(self.state, set())
        if new_state not in allowed and self.state != new_state:
            raise StateMachineError(
                f"Invalid transition {self.state.value} -> {new_state.value}"
            )
        old = self.state
        self.state = new_state
        entry = {"from": old.value, "to": new_state.value, "payload": payload or {}}
        self._transition_log.append(entry)
        if self.on_transition:
            self.on_transition(old, new_state, entry)

    def arm_playbook(self, playbook: Playbook) -> None:
        if not playbook.approved:
            raise StateMachineError("Cannot arm unapproved playbook")
        self.active_playbook = playbook
        self.transition(TradingState.WATCHING)

    def evaluate_signal(
        self, candidate: TradeCandidate, daily_state: DailyState
    ) -> RiskDecision:
        self.transition(TradingState.SIGNAL_DETECTED, {"candidate_id": candidate.candidate_id})
        self.transition(TradingState.RISK_CHECK)
        has_unresolved = len(self.broker.list_open_orders()) > 0
        decision = self.risk_governor.evaluate(
            candidate,
            daily_state,
            has_unresolved_order=has_unresolved,
        )
        if not decision.approved:
            self.transition(TradingState.WATCHING, {"rejected": decision.reason_codes})
        return decision

    def submit_entry(self, candidate: TradeCandidate) -> BrokerOrder:
        self.transition(TradingState.ENTERING)
        intent = OrderIntent(
            candidate_id=candidate.candidate_id,
            contract=candidate.contract,
            side="buy",
            order_type="limit",
            quantity=max(1, int(candidate.quantity)),
            limit_price=candidate.entry_limit_price,
        )
        order = self.broker.submit_order(intent)
        self.last_order = order
        if order.status == "rejected":
            self.transition(
                TradingState.WATCHING,
                {"order_rejected": order.order_id},
            )
            return order
        self.transition(
            TradingState.OPEN_POSITION,
            {
                "order_id": order.order_id,
                "order_status": order.status,
                "pending_fill": order.status in {"open", "pending"},
            },
        )
        return order

    def on_signal(self, candidate: TradeCandidate, daily_state: DailyState) -> RiskDecision:
        decision = self.evaluate_signal(candidate, daily_state)
        if not decision.approved:
            return decision
        self.submit_entry(candidate)
        return decision

    @property
    def transitions(self) -> list[dict]:
        return list(self._transition_log)

"""Communicator replay state integration tests."""

from __future__ import annotations

from joker.agents.communicator import CommunicatorAgent


def _replay_state() -> dict:
    return {
        "mode": "PAPER",
        "trade_state": "WATCHING",
        "council_status": "complete",
        "playbook_title": "Daily SPY Plan",
        "playbook_status": "approved",
        "active_setup": "VWAP reclaim call",
        "market_price": 553.25,
        "last_risk_decision": "rejected: WIDE_SPREAD",
        "last_exit_reason": "stop_loss",
        "agent_mode": "openai",
        "replay_mode": True,
        "replay_is_synthetic": True,
    }


def test_communicator_system_doing_from_replay_state() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("what is the system doing?", _replay_state())
    assert "PAPER" in reply
    assert "553.25" in reply


def test_communicator_reject_reason() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("why did it reject the last trade?", _replay_state())
    assert "WIDE_SPREAD" in reply or "rejected" in reply.lower()


def test_communicator_active_setup() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("what setup is active?", _replay_state())
    assert "VWAP reclaim call" in reply


def test_communicator_last_exit() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("what was the last exit reason?", _replay_state())
    assert "stop_loss" in reply


def test_communicator_agent_mode() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("are we using mock agents or OpenAI agents?", _replay_state())
    assert "openai" in reply.lower()


def test_communicator_synthetic_data() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("is this real data or synthetic data?", _replay_state())
    assert "synthetic" in reply.lower()


def test_communicator_unavailable_state() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("what was the last exit reason?", {})
    assert "unavailable" in reply.lower() or "no exit" in reply.lower()


def test_communicator_refuses_buy_advice() -> None:
    agent = CommunicatorAgent()
    reply = agent.answer("should I buy calls now?", _replay_state())
    assert "cannot provide trade recommendations" in reply.lower()


def test_communicator_no_invented_price() -> None:
    agent = CommunicatorAgent()
    state = {**_replay_state(), "market_price": None}
    reply = agent.answer("what is the system doing?", state)
    assert "unavailable" in reply.lower()
    assert "$6" not in reply  # no invented price

"""Communicator agent — local state only, no invented market data."""

from __future__ import annotations

from typing import Any

from joker.agents.llm_client import LLMClient
from joker.agents.prompts import COMMUNICATOR_ROLE, agent_system_prompt
from joker.agents.security import (
    PromptInjectionDetected,
    check_user_input,
    filter_injection_from_response,
    sanitize_untrusted_input,
)
from joker.schemas.domain import CommunicatorResponse


class CommunicatorAgent:
    name = "CommunicatorAgent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client

    def summarize_system_state(self, state: dict[str, Any]) -> str:
        mode = state.get("mode", "PAPER")
        trade_state = state.get("trade_state", "IDLE")
        council = state.get("council_status", "idle")
        playbook = state.get("playbook_title") or state.get("playbook_status")
        risk = state.get("last_risk_decision")
        parts = [
            f"The system is running in {mode} mode.",
            f"Trade state is {trade_state}.",
            f"Agent council status is {council}.",
        ]
        if playbook:
            parts.append(f"Active playbook: {playbook}.")
        else:
            parts.append("No active playbook data is available.")
        if state.get("market_price") is None:
            parts.append("Current market price is unavailable.")
        else:
            parts.append(f"Last known SPY price: ${state['market_price']:,.2f}.")
        if risk:
            parts.append(f"Last risk decision: {risk}.")
        elif state.get("kill_switch"):
            parts.append("Kill switch is ON — all new orders are blocked.")
        return " ".join(parts)

    def _answer_local(self, question: str, state: dict[str, Any]) -> CommunicatorResponse:
        q = question.lower().strip()
        if any(p in q for p in ("buy", "sell", "should i trade", "you should")):
            return CommunicatorResponse(
                answer=(
                    "I cannot provide trade recommendations. I can explain what the "
                    "system is watching, playbook status, and risk governor decisions."
                ),
                data_available=True,
                refused_advice=True,
            )
        if "risk" in q and ("reject" in q or "decision" in q or "governor" in q):
            risk = state.get("last_risk_decision")
            if not risk:
                return CommunicatorResponse(
                    answer="No risk governor decision is recorded in current state.",
                    data_available=False,
                )
            return CommunicatorResponse(
                answer=f"Latest risk governor decision: {risk}",
                data_available=True,
            )
        if "what is the system doing" in q or "what are you doing" in q:
            return CommunicatorResponse(
                answer=self.summarize_system_state(state),
                data_available=True,
            )
        if ("playbook" in q and "active" in q) or "what setup" in q:
            setup = state.get("active_setup")
            if setup:
                return CommunicatorResponse(
                    answer=f"Active setup: {setup}.",
                    data_available=True,
                )
            title = state.get("playbook_title")
            if title:
                return CommunicatorResponse(
                    answer=f"Playbook '{title}' is active but no specific setup is flagged in state.",
                    data_available=True,
                )
            return CommunicatorResponse(
                answer="No active setup information is available.",
                data_available=False,
            )
        if "playbook" in q:
            title = state.get("playbook_title")
            status = state.get("playbook_status")
            if not title and not status:
                return CommunicatorResponse(
                    answer="Playbook data is unavailable in current state.",
                    data_available=False,
                )
            return CommunicatorResponse(
                answer=f"Playbook '{title or 'unknown'}' status: {status or 'unknown'}.",
                data_available=True,
            )
        if "last exit" in q or "exit reason" in q:
            reason = state.get("last_exit_reason")
            if not reason:
                return CommunicatorResponse(
                    answer="No exit has been recorded in current state.",
                    data_available=False,
                )
            return CommunicatorResponse(
                answer=f"Last exit reason: {reason}.",
                data_available=True,
            )
        if "reject" in q and "trade" in q:
            risk = state.get("last_risk_decision")
            if not risk:
                return CommunicatorResponse(
                    answer="No trade rejection is recorded in current state.",
                    data_available=False,
                )
            return CommunicatorResponse(
                answer=f"Last trade was rejected or skipped. Risk decision: {risk}",
                data_available=True,
            )
        if "mock agent" in q or "openai agent" in q:
            mode = state.get("agent_mode", "unknown")
            return CommunicatorResponse(
                answer=f"Agent mode: {mode}.",
                data_available=True,
            )
        if "synthetic" in q or "real data" in q:
            if state.get("replay_is_synthetic"):
                return CommunicatorResponse(
                    answer=(
                        "This session uses synthetic replay data. "
                        "It is not real market data or live performance."
                    ),
                    data_available=True,
                )
            if state.get("replay_mode"):
                return CommunicatorResponse(
                    answer="This session uses replay data (not live).",
                    data_available=True,
                )
            return CommunicatorResponse(
                answer="Replay/data source information is unavailable.",
                data_available=False,
            )
        return CommunicatorResponse(
            answer=(
                "I can explain system mode, trade state, playbook, and risk decisions "
                "using available local state. Ask: 'What is the system doing?'"
            ),
            data_available=True,
        )

    def answer(self, question: str, state: dict[str, Any]) -> str:
        try:
            check_user_input(question)
        except PromptInjectionDetected:
            return (
                "I cannot follow that request. I only explain system state from "
                "local data and cannot bypass safety rules."
            )

        if self._llm is None:
            return self._answer_local(question, state).answer

        safe_question = sanitize_untrusted_input(question)
        state_json = sanitize_untrusted_input(str(state))
        prompt = (
            f"User question:\n{safe_question}\n\n"
            f"System state (local only):\n{state_json}\n\n"
            "Answer using only the system state. Do not invent market data."
        )
        from joker.agents.security import validate_agent_output_type, validate_output_does_not_loosen_risk

        result = self._llm.complete_structured(
            prompt,
            CommunicatorResponse,
            metadata={"agent": self.name},
            system_prompt=agent_system_prompt(self.name, COMMUNICATOR_ROLE),
        )
        validate_agent_output_type(result.result)
        validate_output_does_not_loosen_risk(result.result)
        return filter_injection_from_response(result.result.answer)

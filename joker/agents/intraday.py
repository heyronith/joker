"""Intraday and postmarket agent helpers for the agentic paper loop."""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from joker.agents.llm_client import LLMClient, LLMClientError
from joker.agents.prompts import agent_system_prompt
from joker.agents.security import validate_agent_output_type, validate_output_does_not_loosen_risk
from joker.compliance.openai_audit import audit_and_sanitize_openai_context
from joker.config.settings import AgentSettings
from joker.memory import memory_prompt_dict
from joker.schemas.domain import (
    DayMemoryBundle,
    IntradayCouncilResult,
    Playbook,
    PlaybookPatch,
    SessionLesson,
    TechnicalFeatures,
    TradeProposal,
)


INTRADAY_ROLE = (
    "You are the IntradayCouncil for SPY 0DTE paper research. "
    "Given live features, the active playbook, and day memory, you may: "
    "(1) propose a PlaybookPatch to enable/disable existing setups only, and/or "
    "(2) set propose_entry=true on a TradeProposal for an existing setup_id. "
    "You cannot place orders, change risk limits, or invent contracts. "
    "If no action is warranted, set propose_entry=false and leave patch null. "
    "Return IntradayCouncilResult."
)

POSTMARKET_ROLE = (
    "You are the PostmarketLearner. Summarize today's session into SessionLesson "
    "for tomorrow's memory. Use only provided stats — no invented prices or OPRA. "
    "Focus on what worked, what failed, risk rejects, and next_day_hints. "
    "Return SessionLesson."
)


class IntradayAgentError(Exception):
    pass


def run_intraday_council(
    llm: LLMClient,
    settings: AgentSettings,
    *,
    run_id: str,
    playbook: Playbook,
    features: TechnicalFeatures,
    memory: DayMemoryBundle | None,
    open_position: bool,
    trades_entered: int,
    max_trades: int,
) -> IntradayCouncilResult:
    context: dict[str, Any] = {
        "run_id": run_id,
        "features": features.model_dump(mode="json"),
        "playbook": playbook.model_dump(mode="json"),
        "open_position": open_position,
        "trades_entered": trades_entered,
        "max_trades_remaining": max(0, max_trades - trades_entered),
        "setups": [
            {
                "setup_id": s.setup_id,
                "name": s.name,
                "direction": s.direction,
                "enabled": s.enabled,
            }
            for s in playbook.setups
        ],
    }
    if memory is not None:
        context["day_memory"] = memory_prompt_dict(memory)

    safe_context, _ = audit_and_sanitize_openai_context(context, prompt_type="IntradayCouncil")
    prompt = (
        f"Context JSON:\n{json.dumps(safe_context, default=str)}\n\n"
        "Return IntradayCouncilResult."
    )
    try:
        result = llm.complete_structured(
            prompt,
            IntradayCouncilResult,
            metadata={"agent": "IntradayCouncil"},
            system_prompt=agent_system_prompt("IntradayCouncil", INTRADAY_ROLE),
            timeout_seconds=float(settings.council_timeout_seconds),
        )
    except LLMClientError as exc:
        raise IntradayAgentError(f"IntradayCouncil failed: {exc}") from exc

    out = result.result
    validate_agent_output_type(out)
    validate_output_does_not_loosen_risk(out)
    if out.patch is not None:
        validate_agent_output_type(out.patch)
        validate_output_does_not_loosen_risk(out.patch)
        if out.patch.playbook_id != playbook.playbook_id:
            out = out.model_copy(
                update={
                    "patch": out.patch.model_copy(
                        update={"playbook_id": playbook.playbook_id}
                    )
                }
            )
    if out.proposal is not None:
        validate_agent_output_type(out.proposal)
        validate_output_does_not_loosen_risk(out.proposal)
        out = out.model_copy(
            update={
                "proposal": out.proposal.model_copy(
                    update={"run_id": run_id or out.proposal.run_id}
                )
            }
        )
    return out


def run_postmarket_learner(
    llm: LLMClient,
    settings: AgentSettings,
    *,
    trading_day: date,
    session_stats: dict[str, Any],
    memory: DayMemoryBundle | None = None,
) -> SessionLesson:
    context: dict[str, Any] = {
        "trading_day": trading_day.isoformat(),
        "session_stats": session_stats,
    }
    if memory is not None:
        context["prior_memory"] = memory_prompt_dict(memory)
    safe_context, _ = audit_and_sanitize_openai_context(
        context, prompt_type="PostmarketLearner"
    )
    prompt = (
        f"Context JSON:\n{json.dumps(safe_context, default=str)}\n\n"
        "Return SessionLesson."
    )
    try:
        result = llm.complete_structured(
            prompt,
            SessionLesson,
            metadata={"agent": "PostmarketLearner"},
            system_prompt=agent_system_prompt("PostmarketLearner", POSTMARKET_ROLE),
            timeout_seconds=float(settings.council_timeout_seconds),
        )
    except LLMClientError as exc:
        raise IntradayAgentError(f"PostmarketLearner failed: {exc}") from exc
    lesson = result.result
    validate_agent_output_type(lesson)
    validate_output_does_not_loosen_risk(lesson)
    return lesson.model_copy(update={"trading_day": trading_day})


def mock_intraday_result(
    playbook: Playbook,
    features: TechnicalFeatures,
    *,
    run_id: str,
) -> IntradayCouncilResult:
    """Deterministic intraday stub for mock_agents mode."""
    from joker.strategy.signal_rules import detect_setup_from_playbook

    setup = detect_setup_from_playbook(playbook, features)
    proposal = None
    if setup is not None:
        proposal = TradeProposal(
            run_id=run_id,
            setup_id=setup.setup_id,
            direction=setup.direction,
            propose_entry=True,
            confidence=0.7,
            rationale="Mock intraday: structured setup matched features",
            stop_pct=setup.stop_pct,
            take_profit_pct=setup.take_profit_pct,
        )
    return IntradayCouncilResult(
        summary="Mock intraday evaluation",
        patch=None,
        proposal=proposal,
    )


def mock_session_lesson(
    trading_day: date,
    session_stats: dict[str, Any],
) -> SessionLesson:
    return SessionLesson(
        trading_day=trading_day,
        summary=f"Mock postmarket for {trading_day.isoformat()}",
        what_worked=["Structured setup evaluation ran"],
        what_failed=[] if session_stats.get("trades_entered") else ["No entries"],
        risk_notes=list(session_stats.get("risk_reject_codes") or [])[:5],
        next_day_hints=["Respect VWAP and momentum thresholds"],
        final_pnl_usd=float(session_stats.get("final_pnl_usd") or 0.0),
        trades_entered=int(session_stats.get("trades_entered") or 0),
        risk_rejections=int(session_stats.get("risk_rejections") or 0),
    )

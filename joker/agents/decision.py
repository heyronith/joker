"""Realtime agent decision-maker for agent_led execution mode."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from joker.agents.decision_context import build_agent_market_context
from joker.agents.llm_client import LLMClient, LLMClientError
from joker.agents.prompts import agent_system_prompt
from joker.agents.security import validate_agent_output_type, validate_output_does_not_loosen_risk
from joker.agents.session_memory import PendingProposal, SessionMicroMemory
from joker.compliance.openai_audit import audit_and_sanitize_openai_context
from joker.config.settings import AgentSettings
from joker.memory import memory_prompt_dict
from joker.risk.capital import CapitalBudget
from joker.schemas.domain import (
    DayMemoryBundle,
    IntradayDecision,
    Playbook,
    RiskConfig,
    TechnicalFeatures,
    TradeProposal,
)


DECISION_ROLE = (
    "You are the realtime DecisionAgent for SPY 0DTE paper/sandbox trading. "
    "You are the primary trading authority for this session. Soft risk caps in context "
    "are advisory only. "
    "Optimize toward the session capital goal in context.capital (target profit on "
    "authorized capital) while never exceeding available_usd or aggression_cap. "
    "On propose/confirm you MUST set win_probability (0-1), expected_r (reward:risk), "
    "and expected_value_usd (positive only when edge is real). Code rejects EV<=0 or "
    "low win_probability. "
    "When proposing/confirming, set allocation_style and optionally capital_fraction "
    "(0-1 of available) or target_contracts. Use aggressive only for high-EV edges; "
    "prefer split/conservative to leave dry powder. "
    "Use a TWO-STEP entry process: "
    "(1) action='propose' when you see an emerging edge — do not enter yet; "
    "(2) on a later tick, action='confirm' only if the edge still holds vs the pending "
    "proposal and fresh quotes; otherwise action='abandon' or 'hold'. "
    "Never confirm without a pending proposal. Prefer hold when data is incoherent. "
    "If goal_met or stance=defend_pause, hold. Use session_expectancy to avoid repeating "
    "bad timing. You cannot place broker orders, change kill_switch, or enable live money."
)


class DecisionAgentError(Exception):
    pass


def run_decision_agent(
    llm: LLMClient,
    settings: AgentSettings,
    *,
    run_id: str,
    playbook: Playbook | None,
    features: TechnicalFeatures,
    risk: RiskConfig,
    memory: DayMemoryBundle | None,
    session_memory: SessionMicroMemory | None,
    capital_budget: CapitalBudget | None,
    open_position: bool,
    trades_entered: int,
    spy_price: float | None,
    option_context: dict[str, Any] | None = None,
) -> IntradayDecision:
    market = build_agent_market_context(
        features=features,
        spy_price=spy_price,
        option_context=option_context or {},
        memory=session_memory,
        now=features.as_of,
    )
    context: dict[str, Any] = {
        "run_id": run_id,
        "execution_mode": "agent_led",
        "open_position": open_position,
        "trades_entered": trades_entered,
        "capital": (
            capital_budget.prompt_dict(minutes_to_close=features.minutes_to_close)
            if capital_budget is not None
            else {}
        ),
        "advisory_risk_caps": {
            "daily_loss_cap_usd": risk.max_daily_loss_usd,
            "trades_cap_per_day": risk.max_trades_per_day,
            "premium_cap_usd": risk.max_premium_usd,
            "spread_cap_pct": risk.max_spread_pct,
            "note": "Advisory only under agent_led — capital.authorized_usd is the hard ceiling",
        },
        "playbook": playbook.model_dump(mode="json") if playbook else None,
        **market,
    }
    if memory is not None:
        context["day_memory"] = memory_prompt_dict(memory)

    safe_context, _ = audit_and_sanitize_openai_context(context, prompt_type="DecisionAgent")
    has_pending = bool(session_memory and session_memory.pending)
    prompt = (
        f"Context JSON:\n{json.dumps(safe_context, default=str)}\n\n"
        f"Pending proposal active: {has_pending}. "
        "Decide now. Return IntradayDecision with action hold|propose|confirm|abandon "
        "and EV fields on propose/confirm."
    )
    try:
        result = llm.complete_structured(
            prompt,
            IntradayDecision,
            metadata={"agent": "DecisionAgent"},
            system_prompt=agent_system_prompt("DecisionAgent", DECISION_ROLE),
            timeout_seconds=float(settings.council_timeout_seconds),
        )
    except LLMClientError as exc:
        raise DecisionAgentError(f"DecisionAgent failed: {exc}") from exc

    out = result.result
    validate_agent_output_type(out)
    validate_output_does_not_loosen_risk(out)
    if out.action in ("propose", "confirm", "enter"):
        if out.direction not in ("long_call", "long_put"):
            raise DecisionAgentError(f"{out.action} requires direction long_call or long_put")
        if out.confidence < 0:
            raise DecisionAgentError("invalid confidence")
        # Soft fill EV defaults so schema always has numbers for downstream gates
        if out.win_probability is None:
            out = out.model_copy(update={"win_probability": out.confidence})
        if out.expected_r is None:
            # Rough R from stop/TP percents
            stop = max(out.stop_pct, 0.05)
            out = out.model_copy(update={"expected_r": out.take_profit_pct / stop})
        if out.expected_value_usd is None:
            # Unit EV proxy in R terms; sizing uses sign + p_win
            p = float(out.win_probability or 0.0)
            er = float(out.expected_r or 0.0)
            unit = p * er - (1.0 - p)
            out = out.model_copy(update={"expected_value_usd": unit})
    return out


def decision_to_proposal(
    decision: IntradayDecision,
    *,
    run_id: str,
    playbook: Playbook | None,
) -> TradeProposal | None:
    """Map an enter/confirm decision to TradeProposal for existing entry machinery."""
    if decision.action not in ("enter", "confirm") or decision.direction is None:
        return None
    setup_id = decision.setup_id
    if not setup_id and playbook is not None:
        for setup in playbook.setups:
            if setup.enabled and setup.direction == decision.direction:
                setup_id = setup.setup_id
                break
    if not setup_id:
        setup_id = f"agent_{decision.direction}"
    return TradeProposal(
        run_id=run_id,
        setup_id=setup_id,
        direction=decision.direction,
        propose_entry=True,
        confidence=decision.confidence,
        rationale=decision.rationale or decision.summary,
        stop_pct=decision.stop_pct,
        take_profit_pct=decision.take_profit_pct,
    )


def pending_from_decision(
    decision: IntradayDecision,
    *,
    spy_price: float,
    option_context: dict[str, Any],
) -> PendingProposal:
    call = option_context.get("atm_call") if isinstance(option_context.get("atm_call"), dict) else {}
    put = option_context.get("atm_put") if isinstance(option_context.get("atm_put"), dict) else {}
    assert decision.direction in ("long_call", "long_put")
    return PendingProposal(
        direction=decision.direction,
        setup_id=decision.setup_id,
        confidence=decision.confidence,
        stop_pct=decision.stop_pct,
        take_profit_pct=decision.take_profit_pct,
        rationale=decision.rationale or decision.summary,
        spy_price=float(spy_price),
        proposed_at=datetime.now(timezone.utc),
        atm_call_mid=float(call["mid"]) if call.get("mid") is not None else None,
        atm_put_mid=float(put["mid"]) if put.get("mid") is not None else None,
        summary=decision.summary or "",
        capital_fraction=decision.capital_fraction,
        target_contracts=decision.target_contracts,
        allocation_style=decision.allocation_style,
        win_probability=decision.win_probability,
        expected_r=decision.expected_r,
        expected_value_usd=decision.expected_value_usd,
    )


def confirm_gate(
    pending: PendingProposal,
    *,
    spy_price: float,
    option_context: dict[str, Any],
    now: datetime | None = None,
    ttl_seconds: float = 120.0,
    max_spy_drift_pct: float = 0.20,
    max_option_mid_worsen_pct: float = 15.0,
) -> tuple[bool, str]:
    """Deterministic freshness checks before an agent confirm can execute."""
    now = now or datetime.now(timezone.utc)
    age = (now - pending.proposed_at).total_seconds()
    if age > ttl_seconds:
        return False, f"proposal_expired age_s={age:.0f}"
    if pending.spy_price > 0:
        drift = abs(spy_price - pending.spy_price) / pending.spy_price * 100.0
        if drift > max_spy_drift_pct:
            return False, f"spy_drift_pct={drift:.3f}"
    key = "atm_call" if pending.direction == "long_call" else "atm_put"
    block = option_context.get(key) if isinstance(option_context.get(key), dict) else {}
    prior_mid = pending.atm_call_mid if pending.direction == "long_call" else pending.atm_put_mid
    mid = block.get("mid")
    if prior_mid is not None and mid is not None and prior_mid > 0:
        # For long options, higher mid = more expensive entry (worse)
        worsen = ((float(mid) - float(prior_mid)) / float(prior_mid)) * 100.0
        if worsen > max_option_mid_worsen_pct:
            return False, f"option_mid_worsened_pct={worsen:.1f}"
    return True, "ok"


def decision_from_pending(
    pending: PendingProposal,
    *,
    confidence: float | None = None,
    capital_fraction: float | None = None,
    target_contracts: int | None = None,
    allocation_style: str | None = None,
    win_probability: float | None = None,
    expected_r: float | None = None,
    expected_value_usd: float | None = None,
) -> IntradayDecision:
    return IntradayDecision(
        action="confirm",
        direction=pending.direction,
        setup_id=pending.setup_id,
        confidence=confidence if confidence is not None else pending.confidence,
        rationale=pending.rationale,
        stop_pct=pending.stop_pct,
        take_profit_pct=pending.take_profit_pct,
        summary=pending.summary or "confirm pending proposal",
        capital_fraction=capital_fraction if capital_fraction is not None else pending.capital_fraction,
        target_contracts=target_contracts if target_contracts is not None else pending.target_contracts,
        allocation_style=(
            allocation_style or pending.allocation_style or "auto"  # type: ignore[arg-type]
        ),
        win_probability=win_probability if win_probability is not None else pending.win_probability,
        expected_r=expected_r if expected_r is not None else pending.expected_r,
        expected_value_usd=(
            expected_value_usd if expected_value_usd is not None else pending.expected_value_usd
        ),
    )


def ev_entry_allowed(
    decision: IntradayDecision,
    *,
    min_win_probability: float = 0.45,
) -> tuple[bool, str]:
    """Deterministic reckless-aggression veto before sizing/order."""
    if decision.action not in ("propose", "confirm", "enter"):
        return True, "n/a"
    if decision.expected_value_usd is not None and decision.expected_value_usd <= 0:
        return False, "ev_non_positive"
    if decision.win_probability is not None and decision.win_probability < min_win_probability:
        return False, f"win_probability_low:{decision.win_probability:.2f}"
    return True, "ok"


def mock_decision(
    features: TechnicalFeatures,
    playbook: Playbook | None,
    *,
    force_enter: bool = False,
    session_memory: SessionMicroMemory | None = None,
) -> IntradayDecision:
    """Offline deterministic decision for tests — respects propose/confirm when memory given."""
    from joker.strategy.signal_rules import detect_setup_from_playbook

    if session_memory is not None and session_memory.pending is not None:
        return decision_from_pending(session_memory.pending, confidence=0.75)

    if playbook is not None:
        setup = detect_setup_from_playbook(playbook, features)
        if setup is not None or force_enter:
            chosen = setup or next((s for s in playbook.setups if s.enabled), None)
            if chosen is not None:
                return IntradayDecision(
                    action="propose" if session_memory is not None else "enter",
                    direction=chosen.direction,
                    setup_id=chosen.setup_id,
                    confidence=0.7,
                    rationale="Mock decision: structured setup match",
                    stop_pct=chosen.stop_pct,
                    take_profit_pct=chosen.take_profit_pct,
                    summary="mock propose" if session_memory is not None else "mock enter",
                    win_probability=0.62,
                    expected_r=chosen.take_profit_pct / max(chosen.stop_pct, 0.05),
                    expected_value_usd=0.5,
                )
    return IntradayDecision(action="hold", summary="mock hold", confidence=0.0)

"""CLI-facing event stream helpers for live paper sessions."""

from __future__ import annotations

from typing import Any

# High-signal events printed immediately to the terminal.
LIVE_CLI_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "live_paper.started",
        "webull.auth.result",
        "memory.loaded",
        "playbook.trimmed",
        "playbook.validation",
        "playbook.approved",
        "playbook.patched",
        "playbook.patch_rejected",
        "signal.detected",
        "risk.decision",
        "order.submitted",
        "order.pending_fill",
        "order.filled",
        "order.failed",
        "exit.executed",
        "option.selected",
        "option.rejected",
        "options.quotes_loaded",
        "options.unavailable",
        "intraday.council",
        "intraday.failed",
        "execution.mode",
        "agent.decision",
        "agent.decision_failed",
        "agent.decision_invalid",
        "agent.decision_low_confidence",
        "agent.execute",
        "agent.propose",
        "agent.propose_expired",
        "agent.propose_abandoned",
        "agent.confirm_rejected",
        "agent.confirm_executed",
        "agent.outcome",
        "agent.prefilter_skip",
        "capital.reserved",
        "capital.rejected",
        "capital.sized",
        "capital.goal_met_pause",
        "option.advisory",
        "proposal.rejected_low_confidence",
        "proposal.unknown_setup",
        "proposal.direction_mismatch",
        "provider.error",
        "provider.poll_error",
        "live_paper.failure",
        "memory.lesson_saved",
        "memory.lesson_failed",
        "live_paper.completed",
    }
)


def should_stream_event(event_type: str) -> bool:
    return event_type in LIVE_CLI_EVENT_TYPES or event_type.startswith("proposal.")


def format_live_event(event_type: str, payload: dict[str, Any] | None = None) -> str:
    """One-line human summary for terminal streaming."""
    p = payload or {}
    if event_type == "signal.detected":
        return (
            f"SIGNAL {p.get('direction', '')} setup={p.get('setup_id', '')[:8]} "
            f"limit={p.get('limit_price')} source={p.get('source')}"
        )
    if event_type == "risk.decision":
        approved = p.get("approved")
        mark = "APPROVED" if approved else "REJECTED"
        reasons = ",".join(p.get("reason_codes") or []) or p.get("message", "")
        return f"RISK {mark} {reasons}"
    if event_type == "order.submitted":
        return (
            f"ORDER {str(p.get('side', '')).upper()} id={p.get('order_id')} "
            f"status={p.get('status')} limit={p.get('limit_price')}"
        )
    if event_type == "order.filled":
        return f"FILL entry={p.get('entry_price')} pos={p.get('position_id')}"
    if event_type == "order.pending_fill":
        return f"PENDING FILL order={p.get('order_id')} status={p.get('status')}"
    if event_type == "order.failed":
        return f"ORDER FAILED id={p.get('order_id')} status={p.get('status')}"
    if event_type == "exit.executed":
        return f"EXIT reason={p.get('reason') or p.get('exit_reason')}"
    if event_type == "intraday.council":
        return (
            f"INTRADAY summary={str(p.get('summary', ''))[:80]} "
            f"propose={p.get('propose_entry')} patch={p.get('has_patch')}"
        )
    if event_type == "execution.mode":
        return (
            f"MODE execution={p.get('execution_mode')} "
            f"risk_policy={p.get('risk_policy')} "
            f"rules_auto_entry={p.get('rules_auto_entry')}"
        )
    if event_type == "agent.decision":
        return (
            f"AI {str(p.get('action', '')).upper()} "
            f"dir={p.get('direction')} conf={p.get('confidence')} "
            f"p_win={p.get('win_probability')} ev={p.get('expected_value_usd')} "
            f"agg={p.get('aggression_cap')} gap={p.get('goal_gap_pct')} "
            f"pending={p.get('pending')} "
            f"{str(p.get('summary', ''))[:80]}"
        )
    if event_type == "agent.propose":
        return (
            f"AI PROPOSE {p.get('direction')} conf={p.get('confidence')} "
            f"p_win={p.get('win_probability')} ev={p.get('expected_value_usd')} "
            f"gap={p.get('goal_gap_pct')} via={p.get('via', '')}"
        )
    if event_type == "agent.confirm_executed":
        return (
            f"AI CONFIRM {p.get('direction')} conf={p.get('confidence')} "
            f"p_win={p.get('win_probability')} ev={p.get('expected_value_usd')} "
            f"gap={p.get('goal_gap_pct')}"
        )
    if event_type == "agent.confirm_rejected":
        return f"AI CONFIRM REJECTED {p.get('reason')}"
    if event_type == "agent.prefilter_skip":
        return f"AI PREFILTER SKIP {p.get('reason')}"
    if event_type == "agent.outcome":
        return f"AI OUTCOME {p.get('quality_note')}"
    if event_type == "capital.sized":
        return (
            f"SIZE qty={p.get('quantity')} notional={p.get('notional_usd')} "
            f"agg={p.get('aggression_cap')} ev_gate={p.get('ev_gate')}"
        )
    if event_type == "option.advisory":
        return f"OPTION ADVISORY {p.get('advisories') or p.get('reason')}"
    if event_type == "agent.execute":
        return (
            f"AI EXECUTE {p.get('direction')} conf={p.get('confidence')} "
            f"setup={p.get('setup_id')}"
        )
    if event_type == "playbook.validation":
        return f"PLAYBOOK validation approved={p.get('approved')} reasons={p.get('reason_codes')}"
    if event_type == "live_paper.started":
        return (
            f"SESSION START broker={p.get('broker')} auto_orders={p.get('auto_orders')} "
            f"duration_s={p.get('duration_seconds')}"
        )
    if event_type == "live_paper.completed":
        return (
            f"SESSION DONE events={p.get('events_processed')} "
            f"entered={p.get('trades_entered')} exited={p.get('trades_exited')} "
            f"pnl={p.get('final_pnl_usd')}"
        )
    # Generic fallback
    keys = list(p.keys())[:4]
    brief = " ".join(f"{k}={p.get(k)}" for k in keys)
    return f"{event_type} {brief}".strip()

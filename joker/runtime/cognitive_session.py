"""Stable cognitive session identity helpers for recoverable Task 2 cycles."""

from __future__ import annotations

from datetime import date


def stable_cognitive_session_id(
    *,
    trading_date: date,
    broker_account_id: str,
    mode: str = "paper",
) -> str:
    """Derive a durable cognitive session id independent of per-process run_id.

    Recovery registries and LangGraph checkpoints key off this identity so a
    normal CLI restart can resume unfinished cycles from the same trading day
    and paper account.
    """
    account = (broker_account_id or "paper").strip().lower() or "paper"
    mode_key = (mode or "paper").strip().lower() or "paper"
    return f"cog:{mode_key}:{account}:{trading_date.isoformat()}"

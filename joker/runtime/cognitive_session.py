"""Stable cognitive session identity helpers for recoverable Task 2 cycles."""

from __future__ import annotations

import hashlib
from datetime import date

from joker.config.settings import EnvSettings
from joker.time.calendar import MarketCalendar
from joker.time.clock import SystemExchangeClock

LOCAL_PAPER_ACCOUNT_IDENTITY = "local_paper"


def hash_paper_account_id(account_id: str) -> str:
    """Stable non-reversible fingerprint of a Webull paper account id."""
    raw = (account_id or "").strip()
    if not raw:
        raise ValueError("paper account id is required for hashing")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def exchange_trading_date(*, calendar: MarketCalendar | None = None) -> date:
    """Current XNYS exchange trading date (not host ``date.today()``)."""
    return SystemExchangeClock(calendar=calendar or MarketCalendar()).trading_date()


def paper_account_identity(
    *,
    broker_kind: str,
    env: EnvSettings,
) -> str:
    """Resolve the durable paper-account identity used in cognitive session keys.

    * ``webull_paper`` → hashed configured ``WEBULL_PAPER_ACCOUNT_ID``
    * ``local_paper`` (and other non-Webull kinds) → explicit local-paper identity
    """
    kind = (broker_kind or "").strip().lower()
    if kind == "webull_paper":
        account = (env.webull_paper_account_id or "").strip()
        if not account:
            raise ValueError(
                "webull_paper cognitive sessions require WEBULL_PAPER_ACCOUNT_ID"
            )
        return f"webull:{hash_paper_account_id(account)}"
    return LOCAL_PAPER_ACCOUNT_IDENTITY


def stable_cognitive_session_id(
    *,
    trading_date: date,
    account_identity: str,
    mode: str = "paper",
) -> str:
    """Derive a durable cognitive session id independent of per-process run_id.

    Recovery registries and LangGraph checkpoints key off this identity so a
    normal CLI restart can resume unfinished cycles for the same paper account
    and exchange trading date only.
    """
    identity = (account_identity or "").strip().lower() or LOCAL_PAPER_ACCOUNT_IDENTITY
    mode_key = (mode or "paper").strip().lower() or "paper"
    return f"cog:{mode_key}:{identity}:{trading_date.isoformat()}"


def live_paper_cognitive_session_id(
    *,
    broker_kind: str,
    env: EnvSettings,
    trading_date: date | None = None,
    mode: str = "paper",
) -> str:
    """Build the live-paper cognitive session id from account + exchange date."""
    return stable_cognitive_session_id(
        trading_date=trading_date or exchange_trading_date(),
        account_identity=paper_account_identity(broker_kind=broker_kind, env=env),
        mode=mode,
    )


def live_gated_cognitive_session_id(
    *,
    account_id_hash: str,
    trading_date: date | None = None,
    scope: str = "live",
) -> str:
    """Stable LIVE_GATED session identity — account hash + exchange trading date.

    ``run_id`` may remain process-unique; ``session_id`` must survive restart so
    Task-1 projection and checkpoints recover the prior day/account session.
    """
    identity = (account_id_hash or "").strip().lower()
    if not identity:
        raise ValueError("live_gated_cognitive_session_id requires account_id_hash")
    return stable_cognitive_session_id(
        trading_date=trading_date or exchange_trading_date(),
        account_identity=f"live:{identity}",
        mode=scope or "live",
    )

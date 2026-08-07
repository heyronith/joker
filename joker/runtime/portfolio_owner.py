"""Persisted portfolio-execution ownership helpers.

Recovery must never invent today's exchange clock trading date for durable
portfolio ownership. Owner date comes only from persisted sources:
session-id embedded date, explicit recovery candidate date, or an explicit
operator/CLI owner date. Conflicting sources fail closed.
"""

from __future__ import annotations

from datetime import date

from joker.persistence.cognitive_execution_provenance import PortfolioExecutionOwner
from joker.runtime.cognitive_session import stable_cognitive_session_trading_date


class PortfolioOwnerDateConflictError(ValueError):
    """Raised when durable owner trading-date sources disagree."""


def _normalize_trading_date(value: str | date | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    # Validate ISO shape.
    return date.fromisoformat(text).isoformat()


def resolve_persisted_portfolio_owner(
    *,
    session_id: str,
    broker_account_identity: str,
    explicit_trading_date: str | date | None = None,
    candidate_trading_date: str | date | None = None,
) -> PortfolioExecutionOwner:
    """Build a durable portfolio owner without consulting the current clock."""
    embedded = stable_cognitive_session_trading_date(session_id)
    sources = {
        "session_id_embedded": _normalize_trading_date(embedded),
        "explicit_recovery_owner": _normalize_trading_date(explicit_trading_date),
        "candidate_trading_date": _normalize_trading_date(candidate_trading_date),
    }
    present = {name: value for name, value in sources.items() if value is not None}
    if not present:
        raise ValueError(
            "persisted portfolio owner trading_date unavailable; "
            "recovery cannot invent the current exchange clock date"
        )
    unique = sorted(set(present.values()))
    if len(unique) > 1:
        raise PortfolioOwnerDateConflictError(
            "portfolio owner trading_date conflict: "
            + ", ".join(f"{name}={value}" for name, value in sorted(present.items()))
        )
    return PortfolioExecutionOwner(
        session_id=session_id,
        broker_account_identity=broker_account_identity,
        trading_date=unique[0],
    )

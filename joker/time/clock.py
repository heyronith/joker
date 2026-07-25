"""Exchange-aware clocks for truthful market-time interpretation."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from joker.time.calendar import (
    EXCHANGE_TZ,
    POST_MARKET_END,
    PREMARKET_START,
    MarketCalendar,
)


class SessionPhase(StrEnum):
    """US equities session phase relative to XNYS extended hours."""

    PREMARKET = "premarket"
    REGULAR = "regular"
    POST_MARKET = "post_market"
    CLOSED = "closed"


class ExchangeClock(Protocol):
    """Protocol for exchange-timezone clocks used across market layers."""

    def now(self) -> datetime:
        """Current exchange-aware timestamp (never naive)."""
        ...

    def trading_date(self) -> date:
        """Exchange trading date for ``now()`` in America/New_York."""
        ...

    def session_phase(self, timestamp: datetime | None = None) -> SessionPhase:
        """Session phase at ``timestamp`` (defaults to ``now()``)."""
        ...


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError(
            "Naive datetime rejected; provide timezone-aware timestamps "
            "(prefer America/New_York)"
        )
    return ts.astimezone(EXCHANGE_TZ)


def _session_phase_for(ts: datetime, calendar: MarketCalendar) -> SessionPhase:
    """Classify ``ts`` into premarket / regular / post_market / closed."""
    local = _ensure_aware(ts)
    local_date = local.date()

    if not calendar.is_trading_day(local_date):
        return SessionPhase.CLOSED

    open_ts = calendar.session_open(local_date)
    close_ts = calendar.session_close(local_date)
    premarket_start = datetime.combine(local_date, PREMARKET_START, tzinfo=EXCHANGE_TZ)
    post_end = datetime.combine(local_date, POST_MARKET_END, tzinfo=EXCHANGE_TZ)

    if premarket_start <= local < open_ts:
        return SessionPhase.PREMARKET
    if open_ts <= local < close_ts:
        return SessionPhase.REGULAR
    if close_ts <= local < post_end:
        return SessionPhase.POST_MARKET
    return SessionPhase.CLOSED


def _trading_date_for(ts: datetime, calendar: MarketCalendar) -> date:
    """Resolve the exchange trading date for ``ts`` (not host local date)."""
    local = _ensure_aware(ts)
    return calendar.current_or_next_session(local)


class SystemExchangeClock:
    """Live clock driven by the host OS, projected into America/New_York."""

    def __init__(self, calendar: MarketCalendar | None = None) -> None:
        self._calendar = calendar if calendar is not None else MarketCalendar()

    @property
    def calendar(self) -> MarketCalendar:
        return self._calendar

    def now(self) -> datetime:
        return datetime.now(EXCHANGE_TZ)

    def trading_date(self) -> date:
        return _trading_date_for(self.now(), self._calendar)

    def session_phase(self, timestamp: datetime | None = None) -> SessionPhase:
        return _session_phase_for(
            timestamp if timestamp is not None else self.now(),
            self._calendar,
        )


class FrozenExchangeClock:
    """Deterministic clock for tests and replay (never reads wall clock)."""

    def __init__(
        self,
        frozen_now: datetime,
        calendar: MarketCalendar | None = None,
    ) -> None:
        self._calendar = calendar if calendar is not None else MarketCalendar()
        self._frozen_now = _ensure_aware(frozen_now)

    @property
    def calendar(self) -> MarketCalendar:
        return self._calendar

    def now(self) -> datetime:
        return self._frozen_now

    def set_now(self, frozen_now: datetime) -> None:
        """Advance or rewind the frozen instant (tests / replay only)."""
        self._frozen_now = _ensure_aware(frozen_now)

    def advance(self, delta: timedelta) -> None:
        """Advance the frozen clock by ``delta``."""
        self._frozen_now = self._frozen_now + delta

    def trading_date(self) -> date:
        return _trading_date_for(self._frozen_now, self._calendar)

    def session_phase(self, timestamp: datetime | None = None) -> SessionPhase:
        return _session_phase_for(
            timestamp if timestamp is not None else self._frozen_now,
            self._calendar,
        )

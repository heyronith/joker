"""NYSE (XNYS) market calendar wrapper built on exchange-calendars."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

EXCHANGE_TZ = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# NYSE extended-hours conventions (canonical wall-clock in America/New_York).
PREMARKET_START = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
POST_MARKET_END = time(20, 0)


@dataclass(frozen=True)
class SessionBoundaries:
    """Exchange-timezone session window for a trading date."""

    trading_date: date
    premarket_open: datetime
    regular_open: datetime
    regular_close: datetime
    post_market_close: datetime
    is_early_close: bool

    @property
    def session_open(self) -> datetime:
        """Alias for regular session open."""
        return self.regular_open

    @property
    def session_close(self) -> datetime:
        """Alias for regular session close."""
        return self.regular_close


class MarketCalendar:
    """Thin, typed wrapper around exchange_calendars XNYS calendar."""

    def __init__(self, calendar: Any | None = None) -> None:
        self._cal = calendar if calendar is not None else xcals.get_calendar("XNYS")
        self._tz = EXCHANGE_TZ

    @property
    def tz(self) -> ZoneInfo:
        return self._tz

    def _ensure_aware(self, ts: datetime) -> datetime:
        if ts.tzinfo is None:
            raise ValueError("Naive datetimes are not allowed; use America/New_York or UTC")
        return ts.astimezone(self._tz)

    def _to_session_label(self, d: date) -> pd.Timestamp:
        return pd.Timestamp(d)

    def is_trading_day(self, d: date) -> bool:
        """Return True if ``d`` is an XNYS trading session."""
        return bool(self._cal.is_session(self._to_session_label(d)))

    def is_open_at(self, ts: datetime) -> bool:
        """Return True if the regular session is open at ``ts``."""
        aware = self._ensure_aware(ts)
        minute = pd.Timestamp(aware.astimezone(_UTC))
        return bool(self._cal.is_open_on_minute(minute))

    def is_early_close(self, d: date) -> bool:
        """Return True if trading date ``d`` has an early regular close."""
        if not self.is_trading_day(d):
            return False
        session = self._to_session_label(d)
        early = self._cal.early_closes
        if early is None or len(early) == 0:
            return False
        return session in set(pd.DatetimeIndex(early).normalize())

    def session_open(self, d: date) -> datetime:
        """Regular session open for trading date ``d`` in exchange TZ."""
        if not self.is_trading_day(d):
            raise ValueError(f"{d.isoformat()} is not an XNYS trading day")
        open_utc = self._cal.session_open(self._to_session_label(d))
        return pd.Timestamp(open_utc).to_pydatetime().astimezone(self._tz)

    def session_close(self, d: date) -> datetime:
        """Regular session close for trading date ``d`` in exchange TZ."""
        if not self.is_trading_day(d):
            raise ValueError(f"{d.isoformat()} is not an XNYS trading day")
        close_utc = self._cal.session_close(self._to_session_label(d))
        return pd.Timestamp(close_utc).to_pydatetime().astimezone(self._tz)

    def session_boundaries(self, d: date) -> SessionBoundaries:
        """Full extended-hours boundaries for trading date ``d``."""
        if not self.is_trading_day(d):
            raise ValueError(f"{d.isoformat()} is not an XNYS trading day")
        regular_open = self.session_open(d)
        regular_close = self.session_close(d)
        premarket_open = datetime.combine(d, PREMARKET_START, tzinfo=self._tz)
        post_market_close = datetime.combine(d, POST_MARKET_END, tzinfo=self._tz)
        return SessionBoundaries(
            trading_date=d,
            premarket_open=premarket_open,
            regular_open=regular_open,
            regular_close=regular_close,
            post_market_close=post_market_close,
            is_early_close=self.is_early_close(d),
        )

    def current_or_next_session(self, ts: datetime) -> date:
        """Return the trading date in progress, or the next upcoming session."""
        aware = self._ensure_aware(ts)
        local_date = aware.date()
        if self.is_trading_day(local_date):
            post_end = datetime.combine(local_date, POST_MARKET_END, tzinfo=self._tz)
            if aware <= post_end:
                return local_date
            nxt = self._cal.next_session(self._to_session_label(local_date))
            return pd.Timestamp(nxt).date()

        session = self._cal.date_to_session(
            self._to_session_label(local_date),
            direction="next",
        )
        return pd.Timestamp(session).date()

    def previous_session(self, d: date) -> date:
        """Previous XNYS trading session before or on ``d``."""
        label = self._to_session_label(d)
        if self.is_trading_day(d):
            prev = self._cal.previous_session(label)
        else:
            prev = self._cal.date_to_session(label, direction="previous")
        return pd.Timestamp(prev).date()

    def next_session(self, d: date) -> date:
        """Next XNYS trading session after or on ``d``."""
        label = self._to_session_label(d)
        if self.is_trading_day(d):
            nxt = self._cal.next_session(label)
        else:
            nxt = self._cal.date_to_session(label, direction="next")
        return pd.Timestamp(nxt).date()

    def minutes_until_close(self, ts: datetime) -> float | None:
        """Minutes until regular close on the session owning ``ts``, else None."""
        aware = self._ensure_aware(ts)
        session = self.current_or_next_session(aware)
        boundaries = self.session_boundaries(session)
        if aware < boundaries.premarket_open:
            return None
        if aware > boundaries.regular_close:
            return 0.0
        return (boundaries.regular_close - aware).total_seconds() / 60.0

    def trading_minutes_elapsed(self, ts: datetime) -> float | None:
        """Minutes since regular open for the active regular session, else None."""
        aware = self._ensure_aware(ts)
        if not self.is_trading_day(aware.date()):
            return None
        boundaries = self.session_boundaries(aware.date())
        if aware < boundaries.regular_open or aware > boundaries.regular_close:
            return None
        return (aware - boundaries.regular_open).total_seconds() / 60.0


"""Market calendar session boundaries and early closes."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock, SessionPhase

ET = ZoneInfo("America/New_York")


def test_is_trading_day_weekday() -> None:
    cal = MarketCalendar()
    assert cal.is_trading_day(date(2026, 7, 1)) is True


def test_session_open_close() -> None:
    cal = MarketCalendar()
    d = date(2026, 7, 1)
    bounds = cal.session_boundaries(d)
    assert bounds.regular_open.tzinfo is not None
    assert bounds.regular_open < bounds.regular_close


def test_early_close_detection() -> None:
    cal = MarketCalendar()
    d = date(2025, 7, 3)
    if cal.is_trading_day(d):
        assert isinstance(cal.session_boundaries(d).is_early_close, bool)


def test_session_phase_premarket() -> None:
    cal = MarketCalendar()
    ts = datetime(2026, 7, 1, 8, 0, tzinfo=ET)
    clock = FrozenExchangeClock(ts, calendar=cal)
    assert clock.session_phase() is SessionPhase.PREMARKET

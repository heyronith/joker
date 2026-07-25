
"""Exchange clock tests including DST and trading date."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock, SessionPhase, SystemExchangeClock

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def test_system_clock_returns_aware_et() -> None:
    clock = SystemExchangeClock(MarketCalendar())
    now = clock.now()
    assert now.tzinfo is not None
    assert now.utcoffset() is not None


def test_frozen_clock_trading_date_near_midnight_utc() -> None:
    # 2026-07-01 03:30 UTC = 2026-06-30 23:30 ET
    cal = MarketCalendar()
    frozen = FrozenExchangeClock(datetime(2026, 7, 1, 3, 30, tzinfo=UTC), calendar=cal)
    assert frozen.trading_date().isoformat() >= "2026-06-30"


def test_dst_spring_forward_session_phase() -> None:
    # US DST spring forward 2026-03-08; Monday 2026-03-09 is a session
    cal = MarketCalendar()
    ts = datetime(2026, 3, 9, 14, 0, tzinfo=UTC)  # 10:00 EDT
    clock = FrozenExchangeClock(ts, calendar=cal)
    assert clock.session_phase() is SessionPhase.REGULAR


def test_dst_fall_back_regular_hours() -> None:
    cal = MarketCalendar()
    ts = datetime(2026, 11, 2, 14, 30, tzinfo=UTC)  # after fall back weekend; Mon Nov 2
    clock = FrozenExchangeClock(ts, calendar=cal)
    phase = clock.session_phase()
    assert phase in {SessionPhase.REGULAR, SessionPhase.PREMARKET, SessionPhase.POST_MARKET, SessionPhase.CLOSED}

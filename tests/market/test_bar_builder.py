
"""1m/5m aggregation, late observations, volume rules."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from joker.market.bars import BarBuilder, BarTimeframe
from joker.market.observations import QuoteObservation, TradeObservation, UnderlyingObservation
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

ET = ZoneInfo("America/New_York")


def _clock(ts: datetime) -> FrozenExchangeClock:
    return FrozenExchangeClock(ts, calendar=MarketCalendar())


def test_quote_does_not_add_volume() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    b.ingest_quote(
        QuoteObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=5),
            received_timestamp=start + timedelta(seconds=5),
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            source="test",
        )
    )
    bars = b.close_ready_bars(start + timedelta(minutes=2))
    m1 = [x for x in bars if x.timeframe is BarTimeframe.M1]
    assert m1
    assert m1[0].volume == 0


def test_trade_adds_volume_and_emits_once() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=10),
            received_timestamp=start + timedelta(seconds=10),
            price=Decimal("500.5"),
            size=100,
            source="test",
        )
    )
    first = b.close_ready_bars(start + timedelta(minutes=2))
    second = b.close_ready_bars(start + timedelta(minutes=2))
    m1 = [x for x in first if x.timeframe is BarTimeframe.M1]
    assert m1 and m1[0].volume == 100
    assert not any(x.timeframe is BarTimeframe.M1 and x.start == m1[0].start for x in second)


def test_5m_aggregation() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=6))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    for i in range(3):
        b.ingest_trade(
            TradeObservation(
                symbol="SPY",
                source_timestamp=start + timedelta(minutes=i, seconds=5),
                received_timestamp=start + timedelta(minutes=i, seconds=5),
                price=Decimal("500") + Decimal(i),
                size=10,
                source="test",
            )
        )
    bars = b.close_ready_bars(start + timedelta(minutes=6))
    assert any(x.timeframe is BarTimeframe.M5 for x in bars)

"""1m/5m aggregation, late observations, volume rules."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from hypothesis import given, settings
from hypothesis import strategies as st

from joker.market.bars import BarBuilder, BarTimeframe, floor_to_interval
from joker.market.observations import QuoteObservation, TradeObservation
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
    # Past end + late tolerance so bars emit.
    bars = b.close_ready_bars(start + timedelta(minutes=1, seconds=3))
    m1 = [x for x in bars if x.timeframe is BarTimeframe.M1]
    assert m1
    assert m1[0].volume == 0


def test_trade_volume_1m_and_5m_exact() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=6))
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
    bars = b.close_ready_bars(start + timedelta(minutes=5, seconds=3))
    m1 = [x for x in bars if x.timeframe is BarTimeframe.M1]
    m5 = [x for x in bars if x.timeframe is BarTimeframe.M5]
    assert len(m1) == 1
    assert m1[0].volume == 100
    assert len(m5) == 1
    assert m5[0].volume == 100


def test_cum_vol_delta_applied_once_not_double_counted_with_trade_size() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=1),
            received_timestamp=start + timedelta(seconds=1),
            price=Decimal("500"),
            size=50,
            cumulative_volume=1000,
            source="test",
        )
    )
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=20),
            received_timestamp=start + timedelta(seconds=20),
            price=Decimal("500.1"),
            size=50,  # must NOT also be counted — cum-vol delta is 75
            cumulative_volume=1075,
            source="test",
        )
    )
    bars = b.close_ready_bars(start + timedelta(minutes=1, seconds=3))
    m1 = next(x for x in bars if x.timeframe is BarTimeframe.M1)
    # First trade: no prior cum-vol → size 50; second: valid delta 75 (not +50).
    assert m1.volume == 125
    # 5m interval still open at T+1m; assert open bar has same volume.
    open_m5 = next(x for x in b.open_bars() if x.timeframe is BarTimeframe.M5)
    assert open_m5.volume == 125


def test_late_observation_accepted_within_tolerance() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    end = start + timedelta(minutes=1)
    clock = _clock(end + timedelta(seconds=1))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=10),
            received_timestamp=start + timedelta(seconds=10),
            price=Decimal("500"),
            size=10,
            source="test",
        )
    )
    # Late by 1s (within 2s tolerance) — still assigned to the 10:00 interval.
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=50),
            received_timestamp=end + timedelta(seconds=1),
            price=Decimal("501"),
            size=5,
            source="test",
        )
    )
    open_m1 = [x for x in b.open_bars() if x.timeframe is BarTimeframe.M1]
    assert open_m1
    assert open_m1[0].late_data is True
    assert "late_data" in open_m1[0].quality_flags
    assert open_m1[0].volume == 15


def test_late_observation_rejected_beyond_tolerance() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    end = start + timedelta(minutes=1)
    clock = _clock(end + timedelta(seconds=5))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=10),
            received_timestamp=start + timedelta(seconds=10),
            price=Decimal("500"),
            size=10,
            source="test",
        )
    )
    b.ingest_trade(
        TradeObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=50),
            received_timestamp=end + timedelta(seconds=5),
            price=Decimal("501"),
            size=99,
            source="test",
        )
    )
    assert any(f.code == "dropped_too_late" for f in b.findings)
    open_m1 = [x for x in b.open_bars() if x.timeframe is BarTimeframe.M1]
    assert open_m1[0].volume == 10


def test_cumulative_volume_regression_finding() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    b.ingest_quote(
        QuoteObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=1),
            received_timestamp=start + timedelta(seconds=1),
            last=Decimal("500"),
            cumulative_volume=1000,
            source="test",
        )
    )
    b.ingest_quote(
        QuoteObservation(
            symbol="SPY",
            source_timestamp=start + timedelta(seconds=2),
            received_timestamp=start + timedelta(seconds=2),
            last=Decimal("500"),
            cumulative_volume=900,
            source="test",
        )
    )
    assert any(f.code == "cumulative_volume_regression" for f in b.findings)


def test_no_overlapping_closed_intervals() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=4))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    for i in range(3):
        b.ingest_trade(
            TradeObservation(
                symbol="SPY",
                source_timestamp=start + timedelta(minutes=i, seconds=5),
                received_timestamp=start + timedelta(minutes=i, seconds=5),
                price=Decimal("500"),
                size=1,
                source="test",
            )
        )
    bars = b.close_ready_bars(start + timedelta(minutes=3, seconds=3))
    m1 = sorted([x for x in bars if x.timeframe is BarTimeframe.M1], key=lambda x: x.start)
    for prev, cur in zip(m1, m1[1:]):
        assert prev.end <= cur.start
        assert prev.start < cur.start


def test_observation_assigned_at_most_once_per_timeframe() -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    trade = TradeObservation(
        symbol="SPY",
        source_timestamp=start + timedelta(seconds=10),
        received_timestamp=start + timedelta(seconds=10),
        price=Decimal("500"),
        size=7,
        source="test",
    )
    b.ingest_trade(trade)
    b.ingest_trade(trade)  # same observation id — second assign skipped
    bars = b.close_ready_bars(start + timedelta(minutes=1, seconds=3))
    m1 = next(x for x in bars if x.timeframe is BarTimeframe.M1)
    assert m1.observation_ids.count(trade.observation_id) == 1
    assert m1.volume == 7


def test_emits_once() -> None:
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
    first = b.close_ready_bars(start + timedelta(minutes=1, seconds=3))
    second = b.close_ready_bars(start + timedelta(minutes=1, seconds=3))
    m1 = [x for x in first if x.timeframe is BarTimeframe.M1]
    assert m1 and m1[0].volume == 100
    assert not any(x.timeframe is BarTimeframe.M1 and x.start == m1[0].start for x in second)


@given(
    sizes=st.lists(st.integers(min_value=1, max_value=50), min_size=1, max_size=8),
)
@settings(max_examples=40, deadline=None)
def test_property_no_overlapping_1m_bars(sizes: list[int]) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = _clock(start + timedelta(minutes=len(sizes) + 2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    for i, size in enumerate(sizes):
        ts = start + timedelta(minutes=i, seconds=5)
        b.ingest_trade(
            TradeObservation(
                symbol="SPY",
                source_timestamp=ts,
                received_timestamp=ts,
                price=Decimal("500"),
                size=size,
                source="test",
            )
        )
    bars = b.close_ready_bars(start + timedelta(minutes=len(sizes), seconds=3))
    m1 = sorted([x for x in bars if x.timeframe is BarTimeframe.M1], key=lambda x: x.start)
    for prev, cur in zip(m1, m1[1:]):
        assert prev.end <= cur.start


@given(offset_seconds=st.integers(min_value=0, max_value=59))
@settings(max_examples=30, deadline=None)
def test_property_one_obs_one_interval_per_tf(offset_seconds: int) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    ts = start + timedelta(seconds=offset_seconds)
    clock = _clock(start + timedelta(minutes=2))
    b = BarBuilder(clock, late_tolerance_seconds=2)
    trade = TradeObservation(
        symbol="SPY",
        source_timestamp=ts,
        received_timestamp=ts,
        price=Decimal("500"),
        size=3,
        source="test",
    )
    b.ingest_trade(trade)
    expected_start = floor_to_interval(ts, BarTimeframe.M1)
    open_m1 = [x for x in b.open_bars() if x.timeframe is BarTimeframe.M1]
    assert len(open_m1) == 1
    assert open_m1[0].start == expected_start
    assert open_m1[0].observation_ids == (trade.observation_id,)

"""Exchange-aligned bar aggregation from typed observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from joker.market.exceptions import FeatureFrameError, FeatureTimeframeError
from joker.market.observations import (
    QuoteObservation,
    TradeObservation,
    UnderlyingObservation,
)
from joker.time.calendar import EXCHANGE_TZ
from joker.time.clock import ExchangeClock

__all__ = [
    "BarBuilder",
    "BarTimeframe",
    "FeatureFrameError",
    "FeatureTimeframeError",
    "MarketBar",
    "floor_to_interval",
    "require_timeframe",
]


class BarTimeframe(StrEnum):
    """Supported exchange-aligned bar intervals."""

    M1 = "1m"
    M5 = "5m"

    @property
    def minutes(self) -> int:
        if self is BarTimeframe.M1:
            return 1
        if self is BarTimeframe.M5:
            return 5
        raise FeatureTimeframeError(f"Unsupported timeframe: {self}")


class MarketBar(BaseModel):
    """Immutable OHLCV bar for a single exchange-aligned interval."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    timeframe: BarTimeframe
    start: datetime
    end: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = 0
    quality_flags: tuple[str, ...] = ()
    incomplete: bool = False
    late_data: bool = False
    observation_ids: tuple[UUID, ...] = ()

    @field_validator("start", "end")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("MarketBar timestamps must be timezone-aware")
        return value

    @field_validator("volume")
    @classmethod
    def _non_negative_volume(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Bar volume cannot be negative")
        return value


def require_timeframe(
    bars: list[MarketBar] | tuple[MarketBar, ...],
    expected: BarTimeframe,
) -> tuple[MarketBar, ...]:
    """Validate every bar matches ``expected``; raise FeatureTimeframeError otherwise."""
    out = tuple(bars)
    for bar in out:
        if bar.timeframe != expected:
            raise FeatureTimeframeError(
                f"Expected timeframe {expected.value}, got {bar.timeframe.value} "
                f"for {bar.symbol} @ {bar.start.isoformat()}"
            )
    return out


def _ensure_aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Naive datetimes are not allowed in bar aggregation")
    return ts.astimezone(EXCHANGE_TZ)


def floor_to_interval(ts: datetime, timeframe: BarTimeframe) -> datetime:
    """Floor ``ts`` to the exchange-aligned interval start in America/New_York."""
    local = _ensure_aware(ts)
    minutes = timeframe.minutes
    truncated = local.replace(second=0, microsecond=0)
    floored_minute = (truncated.minute // minutes) * minutes
    return truncated.replace(minute=floored_minute)


def interval_end(start: datetime, timeframe: BarTimeframe) -> datetime:
    return start + timedelta(minutes=timeframe.minutes)


@dataclass
class _OpenBar:
    symbol: str
    timeframe: BarTimeframe
    start: datetime
    end: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int = 0
    late_data: bool = False
    incomplete: bool = False
    quality_flags: set[str] = field(default_factory=set)
    observation_ids: list[UUID] = field(default_factory=list)
    tick_count: int = 0

    def apply_price(self, price: Decimal, observation_id: UUID, *, late: bool) -> None:
        if self.tick_count == 0:
            self.open = price
            self.high = price
            self.low = price
            self.close = price
        else:
            if price > self.high:
                self.high = price
            if price < self.low:
                self.low = price
            self.close = price
        self.tick_count += 1
        self.observation_ids.append(observation_id)
        if late:
            self.late_data = True
            self.quality_flags.add("late_data")

    def add_volume(self, size: int, *, questionable: bool = False) -> None:
        if size < 0:
            self.quality_flags.add("negative_volume_ignored")
            return
        self.volume += size
        if questionable:
            self.quality_flags.add("questionable_volume")

    def to_market_bar(self, *, incomplete: bool | None = None) -> MarketBar:
        flags = sorted(self.quality_flags)
        is_incomplete = self.incomplete if incomplete is None else incomplete
        if is_incomplete and "incomplete" not in flags:
            flags.append("incomplete")
        if self.tick_count == 0:
            if "empty_interval" not in flags:
                flags.append("empty_interval")
            is_incomplete = True
        return MarketBar(
            symbol=self.symbol,
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            quality_flags=tuple(flags),
            incomplete=is_incomplete,
            late_data=self.late_data,
            observation_ids=tuple(self.observation_ids),
        )


@dataclass
class BarBuilder:
    """
    Aggregate quotes/trades/underlying ticks into exchange-aligned 1m/5m bars.

    Quotes never contribute volume by themselves. Trades contribute ``size``.
    Positive cumulative-volume deltas contribute volume. Completed bars are
    emitted exactly once via ``close_ready_bars``.
    """

    clock: ExchangeClock
    late_tolerance_seconds: int = 2
    _open: dict[tuple[str, BarTimeframe, datetime], _OpenBar] = field(
        default_factory=dict, init=False
    )
    _emitted: set[tuple[str, BarTimeframe, datetime]] = field(default_factory=set, init=False)
    _last_cum_vol: dict[str, int] = field(default_factory=dict, init=False)
    _closed_bars: list[MarketBar] = field(default_factory=list, init=False)

    def ingest_quote(self, obs: QuoteObservation) -> None:
        """Update OHLC from quote mid/last; never add volume from the quote itself."""
        price = self._quote_price(obs.bid, obs.ask, obs.last)
        if price is None:
            return
        for tf in (BarTimeframe.M1, BarTimeframe.M5):
            bar = self._bar_for(obs.symbol, tf, obs.source_timestamp)
            if bar is None:
                continue
            late = self._is_late(bar, obs.source_timestamp)
            bar.apply_price(price, obs.observation_id, late=late)
            self._apply_cum_vol_delta(bar, obs.symbol, obs.cumulative_volume)

    def ingest_trade(self, obs: TradeObservation) -> None:
        """Update OHLC and add trade size to volume."""
        for tf in (BarTimeframe.M1, BarTimeframe.M5):
            bar = self._bar_for(obs.symbol, tf, obs.source_timestamp)
            if bar is None:
                continue
            late = self._is_late(bar, obs.source_timestamp)
            bar.apply_price(obs.price, obs.observation_id, late=late)
            bar.add_volume(obs.size)
            self._apply_cum_vol_delta(bar, obs.symbol, obs.cumulative_volume)

    def ingest_underlying(self, obs: UnderlyingObservation) -> None:
        """Update OHLC from underlying last/mid; volume only via cum-vol delta."""
        price = self._quote_price(obs.bid, obs.ask, obs.last)
        if price is None:
            return
        for tf in (BarTimeframe.M1, BarTimeframe.M5):
            bar = self._bar_for(obs.symbol, tf, obs.source_timestamp)
            if bar is None:
                continue
            late = self._is_late(bar, obs.source_timestamp)
            bar.apply_price(price, obs.observation_id, late=late)
            self._apply_cum_vol_delta(bar, obs.symbol, obs.cumulative_volume)

    def close_ready_bars(self, now: datetime | None = None) -> list[MarketBar]:
        """Emit completed bars whose interval end is at or before ``now`` (once each)."""
        reference = _ensure_aware(now if now is not None else self.clock.now())
        ready: list[MarketBar] = []
        to_remove: list[tuple[str, BarTimeframe, datetime]] = []

        for key, open_bar in self._open.items():
            emit_key = (open_bar.symbol, open_bar.timeframe, open_bar.start)
            if emit_key in self._emitted:
                to_remove.append(key)
                continue
            if reference < open_bar.end:
                continue
            incomplete = open_bar.tick_count == 0 or open_bar.incomplete
            if open_bar.late_data:
                open_bar.quality_flags.add("includes_late_ticks")
            market_bar = open_bar.to_market_bar(incomplete=incomplete)
            self._emitted.add(emit_key)
            self._closed_bars.append(market_bar)
            ready.append(market_bar)
            to_remove.append(key)

        for key in to_remove:
            self._open.pop(key, None)

        ready.sort(key=lambda b: (b.start, b.timeframe.value, b.symbol))
        return ready

    def open_bars(self) -> tuple[MarketBar, ...]:
        """Snapshot of currently open (incomplete) bars."""
        return tuple(
            b.to_market_bar(incomplete=True)
            for b in sorted(self._open.values(), key=lambda x: (x.start, x.timeframe.value))
        )

    def closed_bars(self) -> tuple[MarketBar, ...]:
        return tuple(self._closed_bars)

    def _bar_for(
        self,
        symbol: str,
        timeframe: BarTimeframe,
        ts: datetime,
    ) -> _OpenBar | None:
        start = floor_to_interval(ts, timeframe)
        end = interval_end(start, timeframe)
        key = (symbol, timeframe, start)
        emit_key = (symbol, timeframe, start)

        if emit_key in self._emitted:
            # Already closed — accept only within late tolerance.
            local = _ensure_aware(ts)
            if local <= end + timedelta(seconds=self.late_tolerance_seconds):
                # Re-open transiently is not allowed; late updates after emit are dropped.
                return None
            return None

        existing = self._open.get(key)
        if existing is not None:
            local = _ensure_aware(ts)
            if local >= end:
                if local <= end + timedelta(seconds=self.late_tolerance_seconds):
                    existing.late_data = True
                    existing.quality_flags.add("late_data")
                    return existing
                existing.quality_flags.add("dropped_too_late")
                return None
            return existing

        # Detect gap vs prior open interval for same symbol/timeframe.
        prior_keys = [
            k for k in self._open if k[0] == symbol and k[1] == timeframe and k[2] < start
        ]
        for pk in prior_keys:
            prior = self._open[pk]
            prior.incomplete = True
            prior.quality_flags.add("gap_before_next_interval")

        bar = _OpenBar(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            open=Decimal("0"),
            high=Decimal("0"),
            low=Decimal("0"),
            close=Decimal("0"),
        )
        self._open[key] = bar
        return bar

    def _is_late(self, bar: _OpenBar, ts: datetime) -> bool:
        return _ensure_aware(ts) >= bar.end

    def _apply_cum_vol_delta(
        self,
        bar: _OpenBar,
        symbol: str,
        cumulative_volume: int | None,
    ) -> None:
        if cumulative_volume is None:
            return
        prev = self._last_cum_vol.get(symbol)
        self._last_cum_vol[symbol] = cumulative_volume
        if prev is None:
            return
        delta = cumulative_volume - prev
        if delta < 0:
            bar.quality_flags.add("cumulative_volume_regression")
            return
        if delta == 0:
            return
        bar.add_volume(delta)

    @staticmethod
    def _quote_price(
        bid: Decimal | None,
        ask: Decimal | None,
        last: Decimal | None,
    ) -> Decimal | None:
        if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
            return (bid + ask) / Decimal("2")
        if last is not None and last > 0:
            return last
        if bid is not None and bid > 0:
            return bid
        if ask is not None and ask > 0:
            return ask
        return None

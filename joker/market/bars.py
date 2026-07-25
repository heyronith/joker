"""Exchange-aligned bar aggregation from typed observations.

Volume policy (exactly one volume contribution per observation):
1. If ``cumulative_volume`` is present and the delta vs the prior value for the
   symbol is non-negative, use that delta.
2. Otherwise, for trade observations only, use the explicit ``size``.
3. Never add both trade size and cumulative-volume delta for the same observation.
4. Quote / underlying observations never contribute volume except via a valid
   cumulative-volume delta.

Interval assignment uses ``source_timestamp``. Late acceptance uses
``received_timestamp`` (or ingestion time). Intervals stay open until
``interval_end + late_tolerance`` so in-tolerance late ticks can still apply.
"""

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
    "BarIngestFinding",
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


@dataclass(frozen=True)
class BarIngestFinding:
    """Observable quality finding produced during bar ingestion."""

    code: str
    symbol: str
    timeframe: str | None
    message: str
    observation_id: UUID | None = None


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
        if observation_id not in self.observation_ids:
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
    """Aggregate quotes/trades into exchange-aligned 1m/5m bars.

    See module docstring for the volume policy. Completed bars emit exactly once
    via ``close_ready_bars`` after ``interval_end + late_tolerance``.
    """

    clock: ExchangeClock
    late_tolerance_seconds: int = 2
    _open: dict[tuple[str, BarTimeframe, datetime], _OpenBar] = field(
        default_factory=dict, init=False
    )
    _emitted: set[tuple[str, BarTimeframe, datetime]] = field(default_factory=set, init=False)
    _last_cum_vol: dict[str, int] = field(default_factory=dict, init=False)
    _closed_bars: list[MarketBar] = field(default_factory=list, init=False)
    _findings: list[BarIngestFinding] = field(default_factory=list, init=False)
    _assigned_obs: set[tuple[UUID, BarTimeframe]] = field(default_factory=set, init=False)

    @property
    def findings(self) -> tuple[BarIngestFinding, ...]:
        """Observable ingestion findings (drops, regressions, etc.)."""
        return tuple(self._findings)

    def clear_findings(self) -> None:
        self._findings.clear()

    def ingest_quote(self, obs: QuoteObservation) -> None:
        """Update OHLC from quote mid/last; volume only via valid cum-vol delta."""
        price = self._quote_price(obs.bid, obs.ask, obs.last)
        volume = self._resolve_volume(
            symbol=obs.symbol,
            cumulative_volume=obs.cumulative_volume,
            trade_size=None,
        )
        self._apply_observation(
            symbol=obs.symbol,
            observation_id=obs.observation_id,
            source_timestamp=obs.source_timestamp,
            received_timestamp=obs.received_timestamp,
            price=price,
            volume=volume,
        )

    def ingest_trade(self, obs: TradeObservation) -> None:
        """Update OHLC; volume from cum-vol delta when valid else trade size."""
        volume = self._resolve_volume(
            symbol=obs.symbol,
            cumulative_volume=obs.cumulative_volume,
            trade_size=obs.size,
        )
        self._apply_observation(
            symbol=obs.symbol,
            observation_id=obs.observation_id,
            source_timestamp=obs.source_timestamp,
            received_timestamp=obs.received_timestamp,
            price=obs.price,
            volume=volume,
        )

    def ingest_underlying(self, obs: UnderlyingObservation) -> None:
        """Update OHLC from underlying; volume only via valid cum-vol delta."""
        price = self._quote_price(obs.bid, obs.ask, obs.last)
        volume = self._resolve_volume(
            symbol=obs.symbol,
            cumulative_volume=obs.cumulative_volume,
            trade_size=None,
        )
        self._apply_observation(
            symbol=obs.symbol,
            observation_id=obs.observation_id,
            source_timestamp=obs.source_timestamp,
            received_timestamp=obs.received_timestamp,
            price=price,
            volume=volume,
        )

    def close_ready_bars(self, now: datetime | None = None) -> list[MarketBar]:
        """Emit completed bars after ``end + late_tolerance`` (once each)."""
        reference = _ensure_aware(now if now is not None else self.clock.now())
        tolerance = timedelta(seconds=self.late_tolerance_seconds)
        ready: list[MarketBar] = []
        to_remove: list[tuple[str, BarTimeframe, datetime]] = []

        for key, open_bar in self._open.items():
            emit_key = (open_bar.symbol, open_bar.timeframe, open_bar.start)
            if emit_key in self._emitted:
                to_remove.append(key)
                continue
            # Hold open through late tolerance window.
            if reference < open_bar.end + tolerance:
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

    def _resolve_volume(
        self,
        *,
        symbol: str,
        cumulative_volume: int | None,
        trade_size: int | None,
    ) -> int:
        """Compute volume once per observation; update last cum-vol for symbol."""
        if cumulative_volume is not None:
            prev = self._last_cum_vol.get(symbol)
            self._last_cum_vol[symbol] = cumulative_volume
            if prev is not None:
                delta = cumulative_volume - prev
                if delta >= 0:
                    # Valid cum-vol delta (including zero) wins over trade size.
                    return delta
                self._findings.append(
                    BarIngestFinding(
                        code="cumulative_volume_regression",
                        symbol=symbol,
                        timeframe=None,
                        message=f"Cumulative volume fell from {prev} to {cumulative_volume}",
                    )
                )
                # Invalid delta → fall through to explicit trade size.
        if trade_size is not None and trade_size > 0:
            return trade_size
        return 0

    def _apply_observation(
        self,
        *,
        symbol: str,
        observation_id: UUID,
        source_timestamp: datetime,
        received_timestamp: datetime,
        price: Decimal | None,
        volume: int,
    ) -> None:
        if price is None and volume <= 0:
            return
        for tf in (BarTimeframe.M1, BarTimeframe.M5):
            bar = self._bar_for(
                symbol,
                tf,
                source_timestamp=source_timestamp,
                received_timestamp=received_timestamp,
                observation_id=observation_id,
            )
            if bar is None:
                continue
            assign_key = (observation_id, tf)
            if assign_key in self._assigned_obs:
                continue
            self._assigned_obs.add(assign_key)
            late = _ensure_aware(received_timestamp) >= bar.end
            if price is not None:
                bar.apply_price(price, observation_id, late=late)
            elif late:
                bar.late_data = True
                bar.quality_flags.add("late_data")
            if volume > 0:
                bar.add_volume(volume)

    def _bar_for(
        self,
        symbol: str,
        timeframe: BarTimeframe,
        *,
        source_timestamp: datetime,
        received_timestamp: datetime,
        observation_id: UUID,
    ) -> _OpenBar | None:
        start = floor_to_interval(source_timestamp, timeframe)
        end = interval_end(start, timeframe)
        key = (symbol, timeframe, start)
        emit_key = (symbol, timeframe, start)
        received = _ensure_aware(received_timestamp)
        tolerance = timedelta(seconds=self.late_tolerance_seconds)

        if emit_key in self._emitted:
            self._findings.append(
                BarIngestFinding(
                    code="dropped_after_close",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    message="Observation arrived after bar was already emitted",
                    observation_id=observation_id,
                )
            )
            return None

        if received > end + tolerance:
            self._findings.append(
                BarIngestFinding(
                    code="dropped_too_late",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    message=(
                        f"Observation received after tolerance "
                        f"({self.late_tolerance_seconds}s past {end.isoformat()})"
                    ),
                    observation_id=observation_id,
                )
            )
            return None

        existing = self._open.get(key)
        if existing is not None:
            if received >= end:
                existing.late_data = True
                existing.quality_flags.add("late_data")
            return existing

        # Mark prior open intervals incomplete when a newer interval starts.
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
        if received >= end:
            bar.late_data = True
            bar.quality_flags.add("late_data")
        self._open[key] = bar
        return bar

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

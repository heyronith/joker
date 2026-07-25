"""Technical feature calculation from market data."""

from __future__ import annotations

from datetime import datetime, time, timezone
from statistics import pstdev
from zoneinfo import ZoneInfo

from joker.schemas.domain import Candle, MarketSnapshot, TechnicalFeatures

_ET = ZoneInfo("America/New_York")


def calculate_vwap(
    candles: list[Candle],
    *,
    allow_equal_weight_fallback: bool = True,
) -> float | None:
    """Volume-weighted typical price; equal-weight fallback when volume is all zero.

    Quote-derived candles (Webull bars unavailable) often have volume=0. Without a
    fallback, distance_from_vwap stays None and EdgePrefilter blocks the LLM forever.
    """
    if not candles:
        return None
    total_pv = 0.0
    total_volume = 0.0
    typicals: list[float] = []
    for c in candles:
        typical = (c.high + c.low + c.close) / 3.0
        typicals.append(typical)
        total_pv += typical * c.volume
        total_volume += c.volume
    if total_volume > 0:
        return total_pv / total_volume
    if not allow_equal_weight_fallback or not typicals:
        return None
    return sum(typicals) / len(typicals)


def previous_day_levels(candles: list[Candle]) -> tuple[float | None, float | None]:
    if not candles:
        return None, None
    return max(c.high for c in candles), min(c.low for c in candles)


def session_levels(candles: list[Candle]) -> tuple[float | None, float | None]:
    if not candles:
        return None, None
    return max(c.high for c in candles), min(c.low for c in candles)


def momentum(candles: list[Candle], periods: int = 5) -> float | None:
    if len(candles) < periods + 1:
        return None
    start = candles[-(periods + 1)].close
    end = candles[-1].close
    if start == 0:
        return None
    return ((end - start) / start) * 100.0


def range_pct(candles: list[Candle], periods: int = 15) -> float | None:
    if len(candles) < 2:
        return None
    window = candles[-min(periods, len(candles)) :]
    hi = max(c.high for c in window)
    lo = min(c.low for c in window)
    mid = (hi + lo) / 2.0
    if mid <= 0:
        return None
    return ((hi - lo) / mid) * 100.0


def distance_from_vwap_pct(price: float, vwap: float | None) -> float | None:
    if vwap is None or vwap == 0:
        return None
    return ((price - vwap) / vwap) * 100.0


def pct_from_level(price: float, level: float | None) -> float | None:
    if level is None or level == 0:
        return None
    return ((price - level) / level) * 100.0


def trend_label(momentum_val: float | None) -> str:
    if momentum_val is None:
        return "unknown"
    if momentum_val > 0.15:
        return "trend_up"
    if momentum_val < -0.15:
        return "trend_down"
    return "chop"


def extension_label(dist_vwap: float | None) -> str:
    if dist_vwap is None:
        return "unknown"
    if dist_vwap >= 0.25:
        return "extended_up"
    if dist_vwap <= -0.25:
        return "extended_down"
    return "near_vwap"


def opening_range(candles: list[Candle], bars: int = 5) -> tuple[float | None, float | None]:
    if not candles:
        return None, None
    window = candles[: min(bars, len(candles))]
    return max(c.high for c in window), min(c.low for c in window)


def vwap_bands(
    candles: list[Candle], vwap: float | None, band_mult: float = 1.0
) -> tuple[float | None, float | None]:
    if vwap is None or len(candles) < 3:
        return None, None
    typicals = [(c.high + c.low + c.close) / 3.0 for c in candles]
    try:
        sigma = pstdev(typicals)
    except Exception:
        return None, None
    if sigma <= 0:
        return vwap, vwap
    return vwap + band_mult * sigma, vwap - band_mult * sigma


def session_minutes(as_of: datetime) -> tuple[float | None, float | None, str]:
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    et = as_of.astimezone(_ET)
    open_m = et.hour * 60 + et.minute - (9 * 60 + 30)
    close_m = (16 * 60) - (et.hour * 60 + et.minute)
    in_rth = 0 <= open_m <= (6 * 60 + 30)
    if not in_rth:
        return float(open_m), float(close_m), "outside_rth"
    if open_m < 30:
        part = "open_drive"
    elif close_m < 45:
        part = "power_hour"
    elif open_m < 120:
        part = "morning"
    elif close_m < 150:
        part = "afternoon"
    else:
        part = "midday"
    return float(open_m), float(close_m), part


def split_session_candles(
    candles: list[Candle],
) -> tuple[list[Candle], list[Candle], list[Candle]]:
    """Split candles into prior_day / premarket / RTH by America/New_York clock.

    Heuristic: bars before 09:30 ET today = premarket; bars from previous calendar
    day(s) = prior_day; rest = intraday RTH (caller still uses snapshot.candles for VWAP).
    """
    if not candles:
        return [], [], []
    last = candles[-1].timestamp
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    et_last = last.astimezone(_ET)
    today = et_last.date()
    prior: list[Candle] = []
    premarket: list[Candle] = []
    rth: list[Candle] = []
    for c in candles:
        ts = c.timestamp if c.timestamp.tzinfo else c.timestamp.replace(tzinfo=timezone.utc)
        et = ts.astimezone(_ET)
        if et.date() < today:
            prior.append(c)
        elif et.time() < time(9, 30):
            premarket.append(c)
        else:
            rth.append(c)
    return prior, premarket, rth


def is_stale(as_of: datetime, max_age_seconds: int = 60, reference_time: datetime | None = None) -> bool:
    now = reference_time or datetime.now(timezone.utc)
    ts = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    ref = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return (ref - ts).total_seconds() > max_age_seconds


class FeatureEngine:
    """Calculate typed technical features from snapshots and candles."""

    def __init__(self, max_age_seconds: int = 60) -> None:
        self.max_age_seconds = max_age_seconds

    def compute_from_bars(
        self,
        *,
        symbol: str,
        price: float,
        bars,
        timeframe,
        as_of: datetime,
        reference_time: datetime | None = None,
        prior_day_candles: list[Candle] | None = None,
        premarket_candles: list[Candle] | None = None,
    ) -> TechnicalFeatures:
        """Compute features from explicitly timeframe-tagged bars.

        Raises FeatureTimeframeError when timeframe is missing/invalid.
        Does not infer timeframe from list length.
        """
        from joker.market.bars import MarketBar, require_timeframe
        from joker.market.exceptions import FeatureTimeframeError

        tf = require_timeframe(timeframe)
        if not isinstance(bars, (list, tuple)):
            raise FeatureTimeframeError("bars must be a sequence of MarketBar")
        for b in bars:
            if not isinstance(b, MarketBar):
                raise FeatureTimeframeError("all bars must be MarketBar instances")
            if b.timeframe != tf:
                raise FeatureTimeframeError(
                    f"bar timeframe {b.timeframe.value} does not match requested {tf.value}"
                )
        candles = [
            Candle(
                symbol=symbol,
                timestamp=b.start,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=float(b.volume),
            )
            for b in bars
        ]
        snap = MarketSnapshot(
            symbol=symbol,
            timestamp=as_of,
            price=price,
            candles=candles,
        )
        return self.compute(
            snap,
            prior_day_candles=prior_day_candles,
            premarket_candles=premarket_candles,
            as_of=as_of,
            reference_time=reference_time,
        )

    def compute(
        self,
        snapshot: MarketSnapshot,
        prior_day_candles: list[Candle] | None = None,
        premarket_candles: list[Candle] | None = None,
        as_of: datetime | None = None,
        reference_time: datetime | None = None,
    ) -> TechnicalFeatures:
        """Legacy path: uses snapshot.candles as 1m-equivalent history.

        Prefer compute_from_bars(..., timeframe=BarTimeframe.M1) for Task 1 callers.
        """
        as_of = as_of or snapshot.timestamp
        intraday = snapshot.candles
        vwap = calculate_vwap(intraday)
        prev_high, prev_low = previous_day_levels(prior_day_candles or [])
        pm_high, pm_low = session_levels(premarket_candles or [])
        id_high, id_low = session_levels(intraday)
        mom = momentum(intraday)
        mom15 = momentum(intraday, periods=15)
        dist = distance_from_vwap_pct(snapshot.price, vwap)
        or_high, or_low = opening_range(intraday, bars=5)
        upper, lower = vwap_bands(intraday, vwap)
        mins_open, mins_close, day_part = session_minutes(as_of)
        volume_confirmed: bool | None
        if not intraday:
            volume_confirmed = None
        else:
            volume_confirmed = any(c.volume > 0 for c in intraday)

        return TechnicalFeatures(
            symbol=snapshot.symbol,
            as_of=as_of,
            vwap=vwap,
            previous_high=prev_high,
            previous_low=prev_low,
            premarket_high=pm_high,
            premarket_low=pm_low,
            intraday_high=id_high,
            intraday_low=id_low,
            momentum_5m=mom,
            distance_from_vwap_pct=dist,
            trend_label=trend_label(mom),
            volume_confirmed=volume_confirmed,
            is_stale=is_stale(snapshot.timestamp, self.max_age_seconds, reference_time=reference_time),
            candle_count=len(intraday),
            opening_range_high=or_high,
            opening_range_low=or_low,
            distance_from_or_high_pct=pct_from_level(snapshot.price, or_high),
            distance_from_or_low_pct=pct_from_level(snapshot.price, or_low),
            distance_from_prev_high_pct=pct_from_level(snapshot.price, prev_high),
            distance_from_prev_low_pct=pct_from_level(snapshot.price, prev_low),
            vwap_upper_band=upper,
            vwap_lower_band=lower,
            momentum_15m=mom15,
            range_15m_pct=range_pct(intraday, periods=15),
            extension_label=extension_label(dist),
            minutes_from_open=mins_open,
            minutes_to_close=mins_close,
            day_part=day_part,
        )

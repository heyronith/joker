"""Phase 6 feature engine tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from joker.features.engine import FeatureEngine, calculate_vwap
from joker.schemas.domain import Candle
from tests.fixtures.domain import make_snapshot


def _candle(ts: datetime, close: float, volume: float = 100) -> Candle:
    return Candle(
        symbol="SPY",
        timestamp=ts,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=volume,
    )


def test_vwap_calculation() -> None:
    base = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    candles = [
        _candle(base, 100, 100),
        _candle(base + timedelta(minutes=1), 102, 200),
    ]
    vwap = calculate_vwap(candles)
    assert vwap is not None
    assert 100 < vwap < 102


def test_vwap_equal_weight_fallback_when_volume_zero() -> None:
    base = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    candles = [
        _candle(base, 100, 0),
        _candle(base + timedelta(minutes=1), 102, 0),
    ]
    vwap = calculate_vwap(candles)
    assert vwap is not None
    # Equal-weight typical prices: both typical ≈ close when OHLC near close
    assert 100 < vwap < 102


def test_features_set_volume_confirmed_false_for_quote_candles() -> None:
    engine = FeatureEngine(max_age_seconds=999999)
    base = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    candles = [_candle(base + timedelta(minutes=i), 550 + i, volume=0) for i in range(6)]
    snap = make_snapshot(candles=candles, timestamp=base + timedelta(minutes=5), price=555.0)
    features = engine.compute(snap, as_of=base + timedelta(minutes=5))
    assert features.vwap is not None
    assert features.distance_from_vwap_pct is not None
    assert features.momentum_5m is not None
    assert features.volume_confirmed is False
    assert features.candle_count == 6


def test_stale_snapshot_detected() -> None:
    engine = FeatureEngine(max_age_seconds=30)
    snap = make_snapshot(
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=120)
    )
    features = engine.compute(snap)
    assert features.is_stale is True


def test_missing_candles_handled() -> None:
    engine = FeatureEngine()
    snap = make_snapshot(candles=[])
    features = engine.compute(snap)
    assert features.vwap is None


def test_features_deterministic() -> None:
    engine = FeatureEngine(max_age_seconds=999999)
    base = datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc)
    candles = [_candle(base + timedelta(minutes=i), 550 + i) for i in range(6)]
    snap = make_snapshot(candles=candles, timestamp=base + timedelta(minutes=5))
    f1 = engine.compute(snap, as_of=base + timedelta(minutes=5))
    f2 = engine.compute(snap, as_of=base + timedelta(minutes=5))
    assert f1.model_dump() == f2.model_dump()

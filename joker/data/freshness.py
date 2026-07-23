"""Quote freshness policy for real-time and delayed market data.

Webull OpenAPI quotes may be exchange-delayed. Fail-closed rules:

- Real-time quotes: reject when exchange timestamp age exceeds quote_max_age_seconds.
- Delayed quotes (when allowed): reject when poll silence exceeds feed_max_silence_seconds,
  or when exchange age exceeds delayed_quote_max_age_seconds (sanity ceiling).
- Delayed quotes when not allowed: reject as DELAYED_NOT_ALLOWED.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class FreshnessConfig:
    quote_max_age_seconds: int = 30
    feed_max_silence_seconds: int = 60
    delayed_quote_max_age_seconds: int = 900
    allow_delayed_quotes: bool = True


@dataclass(frozen=True)
class FreshnessVerdict:
    ok: bool
    reason: str | None = None
    exchange_age_seconds: float | None = None
    receive_age_seconds: float | None = None
    delayed: bool = False


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def evaluate_quote_freshness(
    *,
    quote_timestamp: datetime,
    reference_time: datetime,
    delayed: bool = False,
    received_at: datetime | None = None,
    config: FreshnessConfig | None = None,
) -> FreshnessVerdict:
    """Evaluate whether a quote is fresh enough to trade against."""
    cfg = config or FreshnessConfig()
    ref = _aware(reference_time)
    qts = _aware(quote_timestamp)
    exchange_age = (ref - qts).total_seconds()
    recv = _aware(received_at) if received_at is not None else None
    receive_age = (ref - recv).total_seconds() if recv is not None else None

    if delayed and not cfg.allow_delayed_quotes:
        return FreshnessVerdict(
            ok=False,
            reason="DELAYED_NOT_ALLOWED",
            exchange_age_seconds=exchange_age,
            receive_age_seconds=receive_age,
            delayed=True,
        )

    if delayed and cfg.allow_delayed_quotes:
        if receive_age is not None and receive_age > cfg.feed_max_silence_seconds:
            return FreshnessVerdict(
                ok=False,
                reason="FEED_SILENT",
                exchange_age_seconds=exchange_age,
                receive_age_seconds=receive_age,
                delayed=True,
            )
        if exchange_age > cfg.delayed_quote_max_age_seconds:
            return FreshnessVerdict(
                ok=False,
                reason="STALE_QUOTE",
                exchange_age_seconds=exchange_age,
                receive_age_seconds=receive_age,
                delayed=True,
            )
        return FreshnessVerdict(
            ok=True,
            exchange_age_seconds=exchange_age,
            receive_age_seconds=receive_age,
            delayed=True,
        )

    if exchange_age > cfg.quote_max_age_seconds:
        return FreshnessVerdict(
            ok=False,
            reason="STALE_QUOTE",
            exchange_age_seconds=exchange_age,
            receive_age_seconds=receive_age,
            delayed=False,
        )
    return FreshnessVerdict(
        ok=True,
        exchange_age_seconds=exchange_age,
        receive_age_seconds=receive_age,
        delayed=False,
    )


def feed_health_from_received(
    *,
    last_received_at: datetime | None,
    now: datetime,
    feed_max_silence_seconds: int,
    last_error: bool = False,
) -> str:
    """OK / STALE / ERROR based on poll receive time, not exchange timestamp."""
    if last_error:
        return "ERROR"
    if last_received_at is None:
        return "ERROR"
    age = (_aware(now) - _aware(last_received_at)).total_seconds()
    if age > feed_max_silence_seconds:
        return "STALE"
    return "OK"

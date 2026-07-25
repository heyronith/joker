
"""Data quality findings."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

from joker.market.quality import evaluate_data_quality
from joker.market.snapshots import UnderlyingSnapshot

ET = ZoneInfo("America/New_York")


def test_stale_underlying() -> None:
    now = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    report = evaluate_data_quality(
        now=now,
        underlying=UnderlyingSnapshot(
            symbol="SPY",
            exchange_time=now - timedelta(seconds=120),
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
        ),
    )
    codes = {f.code for f in report.findings}
    assert any("stale" in c for c in codes)

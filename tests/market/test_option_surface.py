
"""Option surface sorting, mid, spread, missing greeks."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from joker.market.option_surface import OptionSurfaceBuilder

ET = ZoneInfo("America/New_York")


def test_surface_sort_and_mid() -> None:
    now = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    surface = OptionSurfaceBuilder.from_provider_rows(
        underlying_symbol="SPY",
        exchange_time=now,
        trading_date=date(2026, 7, 1),
        rows=[
            {
                "contract_id": "p1",
                "expiry": date(2026, 7, 1),
                "strike": "501",
                "option_type": "put",
                "bid": "1.0",
                "ask": "1.2",
                "quote_timestamp": now,
            },
            {
                "contract_id": "c1",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "2.0",
                "ask": "2.2",
                "quote_timestamp": now,
            },
        ],
    )
    assert len(surface.contracts) == 2
    assert surface.contracts[0].option_type == "call"
    assert surface.contracts[0].mid == Decimal("2.1")
    assert any("delta" in f or "missing" in f for f in surface.contracts[0].quality_flags)

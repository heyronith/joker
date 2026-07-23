"""Mock market data for development and testing."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.schemas.domain import MarketSnapshot


def mock_spy_snapshot(price: float = 550.0) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="SPY",
        timestamp=datetime.now(timezone.utc),
        price=price,
        bid=price - 0.01,
        ask=price + 0.01,
        candles=[],
    )

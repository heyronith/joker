"""MarketRuntime ingest → snapshot + data-quality persistence."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from joker.events.bus import InProcessAsyncEventBus
from joker.market.bars import BarBuilder
from joker.market.option_surface import OptionSurfaceRepository
from joker.market.snapshots import SnapshotRepository
from joker.persistence.aiosqlite_lifecycle import wait_for_no_aiosqlite_workers
from joker.runtime.market_runtime import MarketRuntime, MarketRuntimeConfig
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_market_runtime_ingest_and_tick_persists_snapshot(tmp_path) -> None:
    start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
    clock = FrozenExchangeClock(start, calendar=MarketCalendar())
    bus = InProcessAsyncEventBus()
    snap_repo = SnapshotRepository(tmp_path / "m.db")
    await snap_repo.initialize()
    surface_repo = OptionSurfaceRepository(tmp_path / "m.db")
    await surface_repo.initialize()
    bars = BarBuilder(clock, late_tolerance_seconds=2)
    rt = MarketRuntime(
        clock=clock,
        bar_builder=bars,
        event_bus=bus,
        snapshot_repo=snap_repo,
        surface_repo=surface_repo,
        session_id="sess-mkt",
        config=MarketRuntimeConfig(min_option_contracts=1, underlying_stale_seconds=3600),
    )

    try:
        await rt.ingest_underlying_quote(
            symbol="SPY",
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            last=Decimal("500"),
            bid_size=10,
            ask_size=12,
            cumulative_volume=1000,
            source_timestamp=start + timedelta(seconds=5),
            received_timestamp=start + timedelta(seconds=5),
        )
        await rt.ingest_trade(
            price=Decimal("500.05"),
            size=25,
            cumulative_volume=1025,
            source_timestamp=start + timedelta(seconds=20),
            received_timestamp=start + timedelta(seconds=20),
        )
        await rt.ingest_option_quotes(
            [
                {
                    "contract_id": "SPY250701C00500000",
                    "symbol": "SPY250701C00500000",
                    "expiry": date(2026, 7, 1),
                    "strike": "500",
                    "option_type": "call",
                    "bid": "1.00",
                    "ask": "1.10",
                    "quote_timestamp": start + timedelta(seconds=20),
                }
            ]
        )

        # Advance clock past 1m end + late tolerance to close bars.
        clock.set_now(start + timedelta(minutes=1, seconds=3))
        result = await rt.tick(now=start + timedelta(minutes=1, seconds=3))
        await bus.drain()

        assert result.snapshot is not None
        assert result.quality is not None
        assert result.surface is not None
        loaded = await snap_repo.get_by_id(result.snapshot.snapshot_id)
        assert loaded is not None
        assert loaded.underlying.symbol == "SPY"
        assert loaded.underlying.exchange_time is not None
        assert loaded.underlying.mid == Decimal("500.0")
        assert loaded.data_quality_id == result.quality.report_id
        assert loaded.option_surface_id == result.surface.surface_id
        assert any(b.volume == 25 for b in loaded.bars_1m) or any(
            b.volume == 25 for b in result.closed_bars
        )
    finally:
        await bus.close()
        await wait_for_no_aiosqlite_workers(timeout_seconds=5.0)

"""Immutable snapshot repository."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from joker.market.snapshots import MarketSnapshot, SnapshotRepository, UnderlyingSnapshot
from joker.persistence.aiosqlite_lifecycle import (
    iter_aiosqlite_worker_threads,
    wait_for_no_aiosqlite_workers,
)

ET = ZoneInfo("America/New_York")


@pytest.mark.asyncio
async def test_snapshot_roundtrip(tmp_path) -> None:
    repo = SnapshotRepository(tmp_path / "snap.db")
    try:
        await repo.initialize()
        now = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        snap = MarketSnapshot(
            exchange_time=now,
            trading_date=date(2026, 7, 1),
            underlying=UnderlyingSnapshot(
                symbol="SPY",
                exchange_time=now,
                last=Decimal("500"),
            ),
            data_quality_id=uuid4(),
        )
        await repo.save(snap)
        loaded = await repo.get_by_id(snap.snapshot_id)
        assert loaded is not None
        assert loaded.snapshot_id == snap.snapshot_id
        rows = await repo.list_by_trading_date(date(2026, 7, 1))
        assert len(rows) == 1
    finally:
        await wait_for_no_aiosqlite_workers(timeout_seconds=5.0)
        assert not iter_aiosqlite_worker_threads()

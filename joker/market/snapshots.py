"""Immutable, schema-versioned market snapshots and append-only persistence."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field, field_validator

from joker.market.bars import MarketBar
from joker.market.exceptions import SnapshotError

SCHEMA_VERSION = "1"


def _reject_naive(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Naive datetimes are not allowed in snapshots")
    return ts


class UnderlyingSnapshot(BaseModel):
    """Point-in-time underlying quote state."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    exchange_time: datetime
    last: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    mid: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    cumulative_volume: int | None = None
    source: str = "unknown"

    @field_validator("exchange_time")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


class FeatureSnapshot(BaseModel):
    """Schema-versioned feature vector derived from bars/snapshots."""

    model_config = ConfigDict(frozen=True)

    feature_snapshot_id: UUID = Field(default_factory=uuid4)
    schema_version: str = SCHEMA_VERSION
    exchange_time: datetime
    trading_date: date
    symbol: str
    features: dict[str, float | str | None] = Field(default_factory=dict)

    @field_validator("exchange_time")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


class DataQualitySnapshot(BaseModel):
    """Compact quality summary referenced by MarketSnapshot.data_quality_id."""

    model_config = ConfigDict(frozen=True)

    data_quality_id: UUID = Field(default_factory=uuid4)
    severity: str
    finding_codes: tuple[str, ...] = ()
    usable_for_reasoning: bool = True
    usable_for_execution: bool = True


class MarketSnapshot(BaseModel):
    """Immutable market state used by downstream reasoning and execution."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: UUID = Field(default_factory=uuid4)
    schema_version: str = SCHEMA_VERSION
    exchange_time: datetime
    trading_date: date

    underlying: UnderlyingSnapshot
    bars_1m: tuple[MarketBar, ...] = ()
    bars_5m: tuple[MarketBar, ...] = ()

    feature_snapshot_id: UUID | None = None
    option_surface_id: UUID | None = None
    data_quality_id: UUID

    source_event_ids: tuple[UUID, ...] = ()

    @field_validator("exchange_time")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    trading_date TEXT NOT NULL,
    exchange_time TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_trading_date
    ON market_snapshots(trading_date);
"""


class SnapshotRepository:
    """Append-only aiosqlite repository for MarketSnapshot rows."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        """Create schema if missing."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await db.commit()
        self._initialized = True

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def save(self, snapshot: MarketSnapshot) -> UUID:
        """Append a snapshot. Duplicate IDs are rejected (append-only)."""
        await self._ensure_initialized()
        payload = snapshot.model_dump_json()
        created_at = datetime.now(tz=snapshot.exchange_time.tzinfo).isoformat()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute(
                    """
                    INSERT INTO market_snapshots
                        (snapshot_id, trading_date, exchange_time, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(snapshot.snapshot_id),
                        snapshot.trading_date.isoformat(),
                        snapshot.exchange_time.isoformat(),
                        payload,
                        created_at,
                    ),
                )
                await db.commit()
        except aiosqlite.IntegrityError as exc:
            raise SnapshotError(
                f"Snapshot {snapshot.snapshot_id} already exists (append-only)"
            ) from exc
        except aiosqlite.Error as exc:
            raise SnapshotError(f"Failed to save snapshot {snapshot.snapshot_id}") from exc
        return snapshot.snapshot_id

    async def get_by_id(self, snapshot_id: UUID) -> MarketSnapshot | None:
        """Load a snapshot by ID, or None if missing."""
        await self._ensure_initialized()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    "SELECT payload_json FROM market_snapshots WHERE snapshot_id = ?",
                    (str(snapshot_id),),
                ) as cursor:
                    row = await cursor.fetchone()
        except aiosqlite.Error as exc:
            raise SnapshotError(f"Failed to load snapshot {snapshot_id}") from exc
        if row is None:
            return None
        return MarketSnapshot.model_validate_json(row[0])

    async def list_by_trading_date(self, trading_date: date) -> list[MarketSnapshot]:
        """Return all snapshots for a trading date, ordered by exchange_time."""
        await self._ensure_initialized()
        try:
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """
                    SELECT payload_json FROM market_snapshots
                    WHERE trading_date = ?
                    ORDER BY exchange_time ASC
                    """,
                    (trading_date.isoformat(),),
                ) as cursor:
                    rows = await cursor.fetchall()
        except aiosqlite.Error as exc:
            raise SnapshotError(
                f"Failed to list snapshots for {trading_date.isoformat()}"
            ) from exc
        return [MarketSnapshot.model_validate_json(r[0]) for r in rows]

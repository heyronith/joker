"""Forward-only durable index of in-process domain events for replay horizons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from joker.evolution.migrations import apply_task3_migrations


@dataclass(frozen=True)
class SessionEventIndexRecord:
    """One indexed domain event within a trading session."""

    event_id: str
    session_id: str
    event_type: str
    exchange_timestamp: datetime
    sequence: int | None = None
    correlation_id: str | None = None
    cycle_id: str | None = None
    snapshot_id: str | None = None
    data_quality_id: str | None = None
    option_surface_id: str | None = None
    client_order_id: str | None = None
    contract_id: str | None = None
    position_lifecycle_id: str | None = None
    payload_json: str = "{}"
    created_at: datetime | None = None


class SessionEventIndexRepository:
    """Append-only session event horizon index (Task 3)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        apply_task3_migrations(self._db_path)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout = 30000")
        self._initialized = True

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def record(self, record: SessionEventIndexRecord) -> bool:
        """Insert event; ignore duplicate event_id (idempotent). Returns True if inserted.

        When ``sequence`` is omitted, assign the next session-local sequence so
        horizon verification can use a globally_contiguous policy.
        """
        await self._ensure()
        created = (record.created_at or datetime.now(timezone.utc)).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA busy_timeout = 30000")
            try:
                await db.execute("BEGIN IMMEDIATE")
                await db.execute(
                    """
                    INSERT INTO session_event_index (
                        event_id, session_id, event_type, exchange_timestamp,
                        sequence, correlation_id, cycle_id, snapshot_id,
                        data_quality_id, option_surface_id, client_order_id,
                        contract_id, position_lifecycle_id, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.event_id,
                        record.session_id,
                        record.event_type,
                        record.exchange_timestamp.isoformat(),
                        record.sequence,
                        record.correlation_id,
                        record.cycle_id,
                        record.snapshot_id,
                        record.data_quality_id,
                        record.option_surface_id,
                        record.client_order_id,
                        record.contract_id,
                        record.position_lifecycle_id,
                        record.payload_json or "{}",
                        created,
                    ),
                )
                # Keep session sequences authoritative in exchange-time order so
                # horizon verification can require timestamp + sequence monotonicity.
                if record.sequence is None:
                    cur = await db.execute(
                        """
                        SELECT event_id FROM session_event_index
                        WHERE session_id = ?
                        ORDER BY exchange_timestamp ASC, event_id ASC
                        """,
                        (record.session_id,),
                    )
                    ordered = await cur.fetchall()
                    for seq_num, (eid,) in enumerate(ordered, start=1):
                        await db.execute(
                            """
                            UPDATE session_event_index
                            SET sequence = ?
                            WHERE event_id = ?
                            """,
                            (seq_num, eid),
                        )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                await db.rollback()
                return False
            except Exception:
                await db.rollback()
                raise

    async def list_horizon(
        self,
        session_id: str,
        *,
        start_timestamp: datetime,
        end_timestamp: datetime,
    ) -> list[SessionEventIndexRecord]:
        """Return events in [start, end] ordered by sequence, exchange_ts, event_id."""
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM session_event_index
                WHERE session_id = ?
                  AND exchange_timestamp >= ?
                  AND exchange_timestamp <= ?
                ORDER BY
                    CASE WHEN sequence IS NULL THEN 1 ELSE 0 END,
                    sequence ASC,
                    exchange_timestamp ASC,
                    event_id ASC
                """,
                (
                    session_id,
                    start_timestamp.isoformat(),
                    end_timestamp.isoformat(),
                ),
            )
            rows = await cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def list_sequence_range(
        self,
        session_id: str,
        *,
        start_sequence: int,
        end_sequence: int,
    ) -> list[SessionEventIndexRecord]:
        """Return events with sequence in [start_sequence, end_sequence]."""
        await self._ensure()
        lo, hi = sorted((int(start_sequence), int(end_sequence)))
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM session_event_index
                WHERE session_id = ?
                  AND sequence IS NOT NULL
                  AND sequence >= ?
                  AND sequence <= ?
                ORDER BY sequence ASC, exchange_timestamp ASC, event_id ASC
                """,
                (session_id, lo, hi),
            )
            rows = await cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def get_by_event_id(self, event_id: str) -> SessionEventIndexRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM session_event_index WHERE event_id = ?",
                (event_id,),
            )
            row = await cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> SessionEventIndexRecord:
        payload = row["payload_json"] or "{}"
        extra: dict[str, Any] = {}
        try:
            extra = json.loads(payload)
        except Exception:
            pass
        ts = datetime.fromisoformat(row["exchange_timestamp"])
        created_raw = row["created_at"]
        created = (
            datetime.fromisoformat(created_raw) if created_raw else None
        )
        return SessionEventIndexRecord(
            event_id=row["event_id"],
            session_id=row["session_id"],
            event_type=row["event_type"],
            exchange_timestamp=ts,
            sequence=row["sequence"],
            correlation_id=row["correlation_id"],
            cycle_id=row["cycle_id"] or extra.get("cycle_id"),
            snapshot_id=row["snapshot_id"] or extra.get("snapshot_id"),
            data_quality_id=row["data_quality_id"] or extra.get("data_quality_id"),
            option_surface_id=row["option_surface_id"] or extra.get("option_surface_id"),
            client_order_id=row["client_order_id"] or extra.get("client_order_id"),
            contract_id=row["contract_id"] or extra.get("contract_id"),
            position_lifecycle_id=row["position_lifecycle_id"]
            or extra.get("position_lifecycle_id"),
            payload_json=payload,
            created_at=created,
        )

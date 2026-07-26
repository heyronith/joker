"""Durable shadow ledger for assignments, fills, positions, and evidence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import aiosqlite
from pydantic import BaseModel, ConfigDict, Field


_SHADOW_DDL = """
CREATE TABLE IF NOT EXISTS shadow_cycles (
    shadow_cycle_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_cycles_assignment
    ON shadow_cycles (assignment_id, created_at);

CREATE TABLE IF NOT EXISTS shadow_orders (
    client_order_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    position_lifecycle_id TEXT,
    contract_id TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_fills (
    fill_id TEXT PRIMARY KEY NOT NULL,
    client_order_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    fee TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_fills_order
    ON shadow_fills (client_order_id, quantity, price);

CREATE TABLE IF NOT EXISTS shadow_positions (
    position_key TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    position_lifecycle_id TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    quantity TEXT NOT NULL,
    average_price TEXT NOT NULL,
    realised_pnl TEXT NOT NULL,
    status TEXT NOT NULL,
    last_snapshot_id TEXT,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_position_events (
    event_id TEXT PRIMARY KEY NOT NULL,
    position_lifecycle_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_runtime_checkpoints (
    assignment_id TEXT PRIMARY KEY NOT NULL,
    last_snapshot_id TEXT,
    cursor_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_evidence_summaries (
    shadow_evidence_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    champion_version_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class ShadowEvidenceSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    shadow_evidence_id: UUID = Field(default_factory=uuid4)
    assignment_id: UUID
    challenger_version_id: UUID
    champion_version_id: UUID
    observed_cycle_count: int = 0
    traded_cycle_count: int = 0
    completed_position_count: int = 0
    open_position_count: int = 0
    regime_tags: tuple[str, ...] = ()
    safety_findings: tuple[str, ...] = ()
    integrity_findings: tuple[str, ...] = ()
    challenger_pnl: Decimal = Decimal("0")
    champion_reference_pnl: Decimal | None = None
    mean_latency_ms: Decimal | None = None
    total_cost_gbp: Decimal | None = None
    minimum_requirements_met: bool = False
    rejection_codes: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ShadowLedger:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SHADOW_DDL)
            await db.commit()

    async def record_cycle(
        self,
        *,
        assignment_id: UUID,
        challenger_version_id: UUID,
        snapshot_id: str,
        status: str,
        payload: dict[str, Any],
    ) -> str:
        await self.initialize()
        cycle_id = str(uuid4())
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO shadow_cycles (
                    shadow_cycle_id, assignment_id, challenger_version_id,
                    snapshot_id, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    str(assignment_id),
                    str(challenger_version_id),
                    snapshot_id,
                    status,
                    json.dumps(payload, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
        return cycle_id

    async def upsert_position(
        self,
        *,
        assignment_id: UUID,
        challenger_version_id: UUID,
        position_lifecycle_id: str,
        contract_id: str,
        configuration_version_id: UUID,
        quantity: Decimal,
        average_price: Decimal,
        realised_pnl: Decimal,
        status: str,
        last_snapshot_id: str | None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self.initialize()
        key = f"{assignment_id}:{position_lifecycle_id}"
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO shadow_positions (
                    position_key, assignment_id, challenger_version_id,
                    position_lifecycle_id, contract_id, configuration_version_id,
                    quantity, average_price, realised_pnl, status,
                    last_snapshot_id, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    str(assignment_id),
                    str(challenger_version_id),
                    position_lifecycle_id,
                    contract_id,
                    str(configuration_version_id),
                    str(quantity),
                    str(average_price),
                    str(realised_pnl),
                    status,
                    last_snapshot_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(payload or {}, default=str),
                ),
            )
            await db.commit()

    async def record_fill(
        self,
        *,
        fill_id: str,
        client_order_id: str,
        assignment_id: UUID,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Return False if fill already existed (idempotent)."""
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO shadow_fills (
                        fill_id, client_order_id, assignment_id,
                        quantity, price, fee, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fill_id,
                        client_order_id,
                        str(assignment_id),
                        str(quantity),
                        str(price),
                        str(fee),
                        json.dumps(payload or {}, default=str),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def save_checkpoint(
        self, assignment_id: UUID, *, last_snapshot_id: str | None, cursor: dict[str, Any]
    ) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO shadow_runtime_checkpoints (
                    assignment_id, last_snapshot_id, cursor_json, updated_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(assignment_id),
                    last_snapshot_id,
                    json.dumps(cursor, default=str),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()

    async def load_checkpoint(self, assignment_id: UUID) -> dict[str, Any] | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM shadow_runtime_checkpoints WHERE assignment_id = ?",
                (str(assignment_id),),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return {
            "last_snapshot_id": row["last_snapshot_id"],
            "cursor": json.loads(row["cursor_json"]),
            "updated_at": row["updated_at"],
        }

    async def list_open_positions(self, assignment_id: UUID | None = None) -> list[dict[str, Any]]:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            if assignment_id is None:
                cur = await db.execute(
                    "SELECT * FROM shadow_positions WHERE status = 'open'"
                )
            else:
                cur = await db.execute(
                    """
                    SELECT * FROM shadow_positions
                    WHERE status = 'open' AND assignment_id = ?
                    """,
                    (str(assignment_id),),
                )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def count_cycles(self, assignment_id: UUID) -> int:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT COUNT(*) FROM shadow_cycles WHERE assignment_id = ?",
                (str(assignment_id),),
            )
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def count_traded_cycles(self, assignment_id: UUID) -> int:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                SELECT COUNT(*) FROM shadow_cycles
                WHERE assignment_id = ? AND status = 'traded'
                """,
                (str(assignment_id),),
            )
            row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def save_evidence_summary(self, summary: ShadowEvidenceSummary) -> None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO shadow_evidence_summaries (
                    shadow_evidence_id, assignment_id, challenger_version_id,
                    champion_version_id, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(summary.shadow_evidence_id),
                    str(summary.assignment_id),
                    str(summary.challenger_version_id),
                    str(summary.champion_version_id),
                    summary.model_dump_json(),
                    summary.created_at.isoformat(),
                ),
            )
            await db.commit()

    async def get_latest_evidence(
        self, assignment_id: UUID
    ) -> ShadowEvidenceSummary | None:
        await self.initialize()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                SELECT payload_json FROM shadow_evidence_summaries
                WHERE assignment_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (str(assignment_id),),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return ShadowEvidenceSummary.model_validate_json(row[0])

"""Durable cognitive-cycle registry for automatic runtime recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid5, NAMESPACE_URL

import aiosqlite

CycleStatus = Literal["pending", "running", "completed", "abandoned"]
GraphKind = Literal["decision", "position"]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cognitive_cycle_registry (
    cycle_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    graph_kind TEXT NOT NULL,
    trigger_event_id TEXT,
    snapshot_id TEXT,
    status TEXT NOT NULL,
    checkpoint_thread_id TEXT NOT NULL,
    last_completed_node TEXT,
    parent_entry_cycle_id TEXT,
    original_strategy_id TEXT,
    original_proposal_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, graph_kind, cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_cog_cycle_status
    ON cognitive_cycle_registry (session_id, status);
"""


def stable_position_cycle_id(
    session_id: str,
    event_id: str,
    *,
    purpose: str = "position_reassessment",
) -> str:
    """Derive a stable position reassessment cycle id from a Task 1 event."""
    return str(uuid5(NAMESPACE_URL, f"{session_id}:{event_id}:{purpose}"))


@dataclass(frozen=True)
class CognitiveCycleRecord:
    session_id: str
    graph_kind: GraphKind
    cycle_id: str
    trigger_event_id: str | None
    snapshot_id: str | None
    status: CycleStatus
    checkpoint_thread_id: str
    last_completed_node: str | None = None
    parent_entry_cycle_id: str | None = None
    original_strategy_id: str | None = None
    original_proposal_id: str | None = None
    payload: dict[str, Any] | None = None
    updated_at: str | None = None


class CognitiveCycleRegistry:
    """Persist and resume unfinished decision/position cycles."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_CREATE_SQL)
            await db.commit()
        self._initialized = True

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def upsert(self, record: CognitiveCycleRecord) -> None:
        await self._ensure()
        updated = record.updated_at or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cognitive_cycle_registry (
                    cycle_id, session_id, graph_kind, trigger_event_id, snapshot_id,
                    status, checkpoint_thread_id, last_completed_node,
                    parent_entry_cycle_id, original_strategy_id, original_proposal_id,
                    payload_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.cycle_id,
                    record.session_id,
                    record.graph_kind,
                    record.trigger_event_id,
                    record.snapshot_id,
                    record.status,
                    record.checkpoint_thread_id,
                    record.last_completed_node,
                    record.parent_entry_cycle_id,
                    record.original_strategy_id,
                    record.original_proposal_id,
                    json.dumps(record.payload or {}),
                    updated,
                ),
            )
            await db.commit()

    async def list_resumable(self, session_id: str) -> list[CognitiveCycleRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM cognitive_cycle_registry
                WHERE session_id = ? AND status IN ('pending', 'running')
                ORDER BY updated_at ASC
                """,
                (session_id,),
            )
            rows = await cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def get(
        self, session_id: str, graph_kind: str, cycle_id: str
    ) -> CognitiveCycleRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM cognitive_cycle_registry
                WHERE session_id = ? AND graph_kind = ? AND cycle_id = ?
                """,
                (session_id, graph_kind, cycle_id),
            )
            row = await cur.fetchone()
        return self._row_to_record(row) if row is not None else None

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> CognitiveCycleRecord:
        return CognitiveCycleRecord(
            session_id=row["session_id"],
            graph_kind=row["graph_kind"],  # type: ignore[arg-type]
            cycle_id=row["cycle_id"],
            trigger_event_id=row["trigger_event_id"],
            snapshot_id=row["snapshot_id"],
            status=row["status"],  # type: ignore[arg-type]
            checkpoint_thread_id=row["checkpoint_thread_id"],
            last_completed_node=row["last_completed_node"],
            parent_entry_cycle_id=row["parent_entry_cycle_id"],
            original_strategy_id=row["original_strategy_id"],
            original_proposal_id=row["original_proposal_id"],
            payload=json.loads(row["payload_json"] or "{}"),
            updated_at=row["updated_at"],
        )

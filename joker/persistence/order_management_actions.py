"""Durable order-management action idempotency keys."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS order_management_action_keys (
    action_key TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    source_order_id TEXT NOT NULL,
    source_order_state TEXT,
    trigger_event_id TEXT,
    decision_id TEXT,
    action TEXT NOT NULL,
    replacement_client_order_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_om_action_session
    ON order_management_action_keys (session_id, source_order_id);
"""


def make_order_management_action_key(
    *,
    source_order_id: str,
    source_order_state: str,
    trigger_event_id: str,
    decision_id: str,
    action: str,
) -> str:
    raw = "|".join(
        [
            source_order_id,
            source_order_state,
            trigger_event_id,
            decision_id,
            action,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OrderManagementActionRecord:
    action_key: str
    session_id: str
    source_order_id: str
    action: str
    source_order_state: str | None = None
    trigger_event_id: str | None = None
    decision_id: str | None = None
    replacement_client_order_id: str | None = None
    created_at: str | None = None


class OrderManagementActionRepository:
    """Durable store preventing duplicate cancel/replace after restart."""

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

    async def has_key(self, action_key: str) -> bool:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                "SELECT 1 FROM order_management_action_keys WHERE action_key = ?",
                (action_key,),
            )
            return await cur.fetchone() is not None

    async def record(self, record: OrderManagementActionRecord) -> bool:
        """Insert action key. Returns False when the key already existed."""
        await self._ensure()
        created = record.created_at or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO order_management_action_keys (
                        action_key, session_id, source_order_id, source_order_state,
                        trigger_event_id, decision_id, action,
                        replacement_client_order_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.action_key,
                        record.session_id,
                        record.source_order_id,
                        record.source_order_state,
                        record.trigger_event_id,
                        record.decision_id,
                        record.action,
                        record.replacement_client_order_id,
                        created,
                    ),
                )
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

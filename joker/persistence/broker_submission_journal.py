"""Durable broker submission journal — idempotent live/paper order identity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiosqlite

SubmissionStatus = Literal[
    "prepared",
    "previewed",
    "submission_started",
    "submission_unknown",
    "accepted",
    "partially_filled",
    "filled",
    "cancel_pending",
    "cancelled",
    "rejected",
    "reconciled",
]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS broker_submission_journal (
    account_id_hash TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    broker_mode TEXT NOT NULL,
    session_id TEXT,
    cycle_id TEXT,
    proposal_id TEXT,
    decision_id TEXT,
    strategy_id TEXT,
    position_lifecycle_id TEXT,
    contract_id TEXT,
    side TEXT,
    position_intent TEXT,
    quantity INTEGER,
    limit_price TEXT,
    payload_hash TEXT,
    preview_hash TEXT,
    status TEXT NOT NULL,
    broker_order_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_error_code TEXT,
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (account_id_hash, client_order_id)
);
CREATE INDEX IF NOT EXISTS idx_broker_sub_journal_status
    ON broker_submission_journal (status, updated_at);
CREATE INDEX IF NOT EXISTS idx_broker_sub_journal_session
    ON broker_submission_journal (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_broker_sub_journal_lifecycle
    ON broker_submission_journal (position_lifecycle_id, created_at);
"""


def payload_hash(payload: dict[str, Any] | None) -> str:
    canonical = json.dumps(payload or {}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BrokerSubmissionRecord:
    client_order_id: str
    broker_mode: str
    account_id_hash: str
    status: SubmissionStatus
    session_id: str | None = None
    cycle_id: str | None = None
    proposal_id: str | None = None
    decision_id: str | None = None
    strategy_id: str | None = None
    position_lifecycle_id: str | None = None
    contract_id: str | None = None
    side: str | None = None
    position_intent: str | None = None
    quantity: int | None = None
    limit_price: str | None = None
    payload_hash: str | None = None
    preview_hash: str | None = None
    broker_order_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    last_error_code: str | None = None
    extra: dict[str, Any] | None = None


class DuplicateSubmissionError(Exception):
    pass


class BrokerSubmissionJournal:
    """Append-friendly status transitions with unique (account, client_order_id)."""

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

    async def prepare(self, record: BrokerSubmissionRecord) -> BrokerSubmissionRecord:
        """Insert prepared row. Rejects duplicates — never INSERT OR REPLACE."""
        await self._ensure()
        now = record.created_at or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO broker_submission_journal (
                        account_id_hash, client_order_id, broker_mode, session_id,
                        cycle_id, proposal_id, decision_id, strategy_id,
                        position_lifecycle_id, contract_id, side, position_intent,
                        quantity, limit_price, payload_hash, preview_hash, status,
                        broker_order_id, created_at, updated_at, last_error_code,
                        extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.account_id_hash,
                        record.client_order_id,
                        record.broker_mode,
                        record.session_id,
                        record.cycle_id,
                        record.proposal_id,
                        record.decision_id,
                        record.strategy_id,
                        record.position_lifecycle_id,
                        record.contract_id,
                        record.side,
                        record.position_intent,
                        record.quantity,
                        record.limit_price,
                        record.payload_hash,
                        record.preview_hash,
                        "prepared",
                        record.broker_order_id,
                        now,
                        now,
                        record.last_error_code,
                        json.dumps(record.extra or {}),
                    ),
                )
                await db.commit()
            except aiosqlite.IntegrityError as exc:
                raise DuplicateSubmissionError(
                    f"duplicate client_order_id for account hash "
                    f"{record.account_id_hash[:8]}…"
                ) from exc
        stored = await self.get(record.account_id_hash, record.client_order_id)
        assert stored is not None
        return stored

    async def transition(
        self,
        *,
        account_id_hash: str,
        client_order_id: str,
        status: SubmissionStatus,
        broker_order_id: str | None = None,
        preview_hash: str | None = None,
        payload_hash_value: str | None = None,
        last_error_code: str | None = None,
        extra_update: dict[str, Any] | None = None,
    ) -> BrokerSubmissionRecord:
        await self._ensure()
        existing = await self.get(account_id_hash, client_order_id)
        if existing is None:
            raise KeyError(f"submission not found: {client_order_id}")
        now = datetime.now(timezone.utc).isoformat()
        extra = dict(existing.extra or {})
        if extra_update:
            extra.update(extra_update)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE broker_submission_journal
                SET status = ?,
                    broker_order_id = COALESCE(?, broker_order_id),
                    preview_hash = COALESCE(?, preview_hash),
                    payload_hash = COALESCE(?, payload_hash),
                    last_error_code = COALESCE(?, last_error_code),
                    updated_at = ?,
                    extra_json = ?
                WHERE account_id_hash = ? AND client_order_id = ?
                """,
                (
                    status,
                    broker_order_id,
                    preview_hash,
                    payload_hash_value,
                    last_error_code,
                    now,
                    json.dumps(extra),
                    account_id_hash,
                    client_order_id,
                ),
            )
            await db.commit()
        stored = await self.get(account_id_hash, client_order_id)
        assert stored is not None
        return stored

    async def get(
        self, account_id_hash: str, client_order_id: str
    ) -> BrokerSubmissionRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM broker_submission_journal
                WHERE account_id_hash = ? AND client_order_id = ?
                """,
                (account_id_hash, client_order_id),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    async def list_by_status(
        self, status: SubmissionStatus, *, account_id_hash: str | None = None
    ) -> list[BrokerSubmissionRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            if account_id_hash:
                cursor = await db.execute(
                    """
                    SELECT * FROM broker_submission_journal
                    WHERE status = ? AND account_id_hash = ?
                    ORDER BY updated_at ASC
                    """,
                    (status, account_id_hash),
                )
            else:
                cursor = await db.execute(
                    """
                    SELECT * FROM broker_submission_journal
                    WHERE status = ?
                    ORDER BY updated_at ASC
                    """,
                    (status,),
                )
            rows = await cursor.fetchall()
        return [_row_to_record(r) for r in rows]

    async def list_unknown_or_open(
        self, *, account_id_hash: str
    ) -> list[BrokerSubmissionRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT * FROM broker_submission_journal
                WHERE account_id_hash = ?
                  AND status IN (
                    'prepared', 'previewed', 'submission_started',
                    'submission_unknown', 'accepted', 'partially_filled',
                    'cancel_pending'
                  )
                ORDER BY created_at ASC
                """,
                (account_id_hash,),
            )
            rows = await cursor.fetchall()
        return [_row_to_record(r) for r in rows]


def _row_to_record(row: aiosqlite.Row) -> BrokerSubmissionRecord:
    extra_raw = row["extra_json"] if "extra_json" in row.keys() else "{}"
    try:
        extra = json.loads(extra_raw or "{}")
    except json.JSONDecodeError:
        extra = {}
    return BrokerSubmissionRecord(
        client_order_id=row["client_order_id"],
        broker_mode=row["broker_mode"],
        account_id_hash=row["account_id_hash"],
        status=row["status"],  # type: ignore[arg-type]
        session_id=row["session_id"],
        cycle_id=row["cycle_id"],
        proposal_id=row["proposal_id"],
        decision_id=row["decision_id"],
        strategy_id=row["strategy_id"],
        position_lifecycle_id=row["position_lifecycle_id"],
        contract_id=row["contract_id"],
        side=row["side"],
        position_intent=row["position_intent"],
        quantity=row["quantity"],
        limit_price=row["limit_price"],
        payload_hash=row["payload_hash"],
        preview_hash=row["preview_hash"],
        broker_order_id=row["broker_order_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        last_error_code=row["last_error_code"],
        extra=extra if isinstance(extra, dict) else {},
    )


def apply_broker_submission_journal_migration(db_path: str | Path) -> None:
    """Sync DDL for Task-1 migration path."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        conn.executescript(_CREATE_SQL)
        conn.commit()
    finally:
        conn.close()


class SyncBrokerSubmissionJournal:
    """Synchronous journal used by BrokerClient.submit_order (sync interface)."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        apply_broker_submission_journal_migration(self._db_path)

    def prepare(self, record: BrokerSubmissionRecord) -> BrokerSubmissionRecord:
        import sqlite3

        now = record.created_at or datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self._db_path)
        try:
            try:
                conn.execute(
                    """
                    INSERT INTO broker_submission_journal (
                        account_id_hash, client_order_id, broker_mode, session_id,
                        cycle_id, proposal_id, decision_id, strategy_id,
                        position_lifecycle_id, contract_id, side, position_intent,
                        quantity, limit_price, payload_hash, preview_hash, status,
                        broker_order_id, created_at, updated_at, last_error_code,
                        extra_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.account_id_hash,
                        record.client_order_id,
                        record.broker_mode,
                        record.session_id,
                        record.cycle_id,
                        record.proposal_id,
                        record.decision_id,
                        record.strategy_id,
                        record.position_lifecycle_id,
                        record.contract_id,
                        record.side,
                        record.position_intent,
                        record.quantity,
                        record.limit_price,
                        record.payload_hash,
                        record.preview_hash,
                        "prepared",
                        record.broker_order_id,
                        now,
                        now,
                        record.last_error_code,
                        json.dumps(record.extra or {}),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise DuplicateSubmissionError(
                    f"duplicate client_order_id for account hash "
                    f"{record.account_id_hash[:8]}…"
                ) from exc
        finally:
            conn.close()
        stored = self.get(record.account_id_hash, record.client_order_id)
        assert stored is not None
        return stored

    def transition(
        self,
        *,
        account_id_hash: str,
        client_order_id: str,
        status: SubmissionStatus,
        broker_order_id: str | None = None,
        preview_hash: str | None = None,
        payload_hash_value: str | None = None,
        last_error_code: str | None = None,
        extra_update: dict[str, Any] | None = None,
    ) -> BrokerSubmissionRecord:
        import sqlite3

        existing = self.get(account_id_hash, client_order_id)
        if existing is None:
            raise KeyError(f"submission not found: {client_order_id}")
        now = datetime.now(timezone.utc).isoformat()
        extra = dict(existing.extra or {})
        if extra_update:
            extra.update(extra_update)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                UPDATE broker_submission_journal
                SET status = ?,
                    broker_order_id = COALESCE(?, broker_order_id),
                    preview_hash = COALESCE(?, preview_hash),
                    payload_hash = COALESCE(?, payload_hash),
                    last_error_code = COALESCE(?, last_error_code),
                    updated_at = ?,
                    extra_json = ?
                WHERE account_id_hash = ? AND client_order_id = ?
                """,
                (
                    status,
                    broker_order_id,
                    preview_hash,
                    payload_hash_value,
                    last_error_code,
                    now,
                    json.dumps(extra),
                    account_id_hash,
                    client_order_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
        stored = self.get(account_id_hash, client_order_id)
        assert stored is not None
        return stored

    def get(
        self, account_id_hash: str, client_order_id: str
    ) -> BrokerSubmissionRecord | None:
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT * FROM broker_submission_journal
                WHERE account_id_hash = ? AND client_order_id = ?
                """,
                (account_id_hash, client_order_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _row_to_record(row)

    def list_by_status(
        self, status: SubmissionStatus, *, account_id_hash: str | None = None
    ) -> list[BrokerSubmissionRecord]:
        import sqlite3

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            if account_id_hash:
                rows = conn.execute(
                    """
                    SELECT * FROM broker_submission_journal
                    WHERE status = ? AND account_id_hash = ?
                    ORDER BY updated_at ASC
                    """,
                    (status, account_id_hash),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM broker_submission_journal
                    WHERE status = ?
                    ORDER BY updated_at ASC
                    """,
                    (status,),
                ).fetchall()
        finally:
            conn.close()
        return [_row_to_record(r) for r in rows]

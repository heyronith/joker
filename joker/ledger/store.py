"""Async SQLite append-only ledger store."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from joker.ledger.exceptions import IdempotencyConflict, LedgerError
from joker.ledger.schemas import LedgerEvent

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_events (
    ledger_event_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    position_id TEXT,
    exchange_timestamp TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_session
    ON ledger_events (session_id, exchange_timestamp, ledger_event_id);
CREATE INDEX IF NOT EXISTS idx_ledger_order
    ON ledger_events (client_order_id, exchange_timestamp, ledger_event_id);
CREATE INDEX IF NOT EXISTS idx_ledger_contract
    ON ledger_events (contract_id, exchange_timestamp, ledger_event_id);
CREATE INDEX IF NOT EXISTS idx_ledger_position
    ON ledger_events (position_id, exchange_timestamp, ledger_event_id);
"""


class SqliteLedgerStore:
    """Async append-only SQLite ledger with unique idempotency keys."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection and create tables/indexes if missing."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            from joker.persistence.aiosqlite_lifecycle import close_aiosqlite_connection

            conn = self._conn
            self._conn = None
            await close_aiosqlite_connection(conn)

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise LedgerError("SqliteLedgerStore is not initialized; call initialize() first")
        return self._conn

    async def append(self, event: LedgerEvent) -> bool:
        """Append a ledger event transactionally.

        Returns True if inserted. Returns False if the same idempotency_key already
        exists with an identical event payload (idempotent replay).

        Raises IdempotencyConflict if the key exists with a different payload.
        """
        conn = self._require_conn()
        existing = await self._get_by_idempotency_key(event.idempotency_key)
        if existing is not None:
            if self._events_equivalent(existing, event):
                return False
            raise IdempotencyConflict(
                f"idempotency_key={event.idempotency_key!r} already used by "
                f"ledger_event_id={existing.ledger_event_id}"
            )

        try:
            await conn.execute(
                """
                INSERT INTO ledger_events (
                    ledger_event_id, idempotency_key, session_id, client_order_id,
                    contract_id, position_id, exchange_timestamp, created_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.ledger_event_id),
                    event.idempotency_key,
                    event.session_id,
                    event.client_order_id,
                    event.contract_id,
                    event.position_id,
                    event.exchange_timestamp.isoformat(),
                    event.created_at.isoformat(),
                    event.model_dump_json(),
                ),
            )
            await conn.commit()
        except aiosqlite.IntegrityError as exc:
            await conn.rollback()
            existing = await self._get_by_idempotency_key(event.idempotency_key)
            if existing is not None and self._events_equivalent(existing, event):
                return False
            raise IdempotencyConflict(
                f"unique constraint failed for idempotency_key={event.idempotency_key!r}"
            ) from exc
        except Exception as exc:
            await conn.rollback()
            raise LedgerError(f"failed to append ledger event: {exc}") from exc
        return True

    async def _get_by_idempotency_key(self, key: str) -> LedgerEvent | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            "SELECT payload FROM ledger_events WHERE idempotency_key = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return LedgerEvent.model_validate_json(row["payload"])

    @staticmethod
    def _events_equivalent(left: LedgerEvent, right: LedgerEvent) -> bool:
        """Compare business fields ignoring created_at / ledger_event_id / exchange_timestamp.

        ``exchange_timestamp`` is excluded so re-polling an already-recorded fill
        (same idempotency_key, new wall-clock stamp) is treated as an idempotent
        replay rather than a conflict.
        """
        return (
            left.broker_account_id == right.broker_account_id
            and left.client_order_id == right.client_order_id
            and left.broker_order_id == right.broker_order_id
            and left.contract_id == right.contract_id
            and left.side == right.side
            and left.quantity == right.quantity
            and left.price == right.price
            and left.source_event_id == right.source_event_id
            and left.idempotency_key == right.idempotency_key
            and left.event_type == right.event_type
            and left.fees == right.fees
            and left.metadata == right.metadata
            and left.session_id == right.session_id
            and left.position_id == right.position_id
        )

    async def get_by_session(self, session_id: str) -> list[LedgerEvent]:
        return await self._query(
            "SELECT payload FROM ledger_events WHERE session_id = ? "
            "ORDER BY exchange_timestamp ASC, created_at ASC, ledger_event_id ASC",
            (session_id,),
        )

    async def get_by_order(self, client_order_id: str) -> list[LedgerEvent]:
        return await self._query(
            "SELECT payload FROM ledger_events WHERE client_order_id = ? "
            "ORDER BY exchange_timestamp ASC, created_at ASC, ledger_event_id ASC",
            (client_order_id,),
        )

    async def get_by_contract(self, contract_id: str) -> list[LedgerEvent]:
        return await self._query(
            "SELECT payload FROM ledger_events WHERE contract_id = ? "
            "ORDER BY exchange_timestamp ASC, created_at ASC, ledger_event_id ASC",
            (contract_id,),
        )

    async def get_by_position(self, position_id: str) -> list[LedgerEvent]:
        return await self._query(
            "SELECT payload FROM ledger_events WHERE position_id = ? "
            "ORDER BY exchange_timestamp ASC, created_at ASC, ledger_event_id ASC",
            (position_id,),
        )

    async def all_events(self) -> list[LedgerEvent]:
        return await self._query(
            "SELECT payload FROM ledger_events "
            "ORDER BY exchange_timestamp ASC, created_at ASC, ledger_event_id ASC",
            (),
        )

    async def _query(self, sql: str, params: tuple) -> list[LedgerEvent]:
        conn = self._require_conn()
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [LedgerEvent.model_validate_json(row["payload"]) for row in rows]

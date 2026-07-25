"""SQLite-backed graph checkpoint store for JokerGraphState.

LangGraph's ``AsyncSqliteSaver`` expects full Checkpoint channel tuples, not an
arbitrary TypedDict. Task 1 therefore uses a thin aiosqlite JSON table that is
still SQLite-backed and smoke-test friendly. Sync helpers wrap the async API.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

import aiosqlite

from joker.graph.state import JokerGraphState

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS graph_checkpoints (
    checkpoint_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    state_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_graph_checkpoints_session
    ON graph_checkpoints (session_id, created_at DESC);
"""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_state(state: JokerGraphState | dict[str, Any]) -> str:
    """JSON-encode state, converting datetimes to ISO-8601."""

    def default(obj: object) -> str:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

    return json.dumps(dict(state), default=default, sort_keys=True)


def _deserialize_state(raw: str) -> JokerGraphState:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("checkpoint state_json must be an object")
    exchange_time = data.get("exchange_time")
    if isinstance(exchange_time, str):
        data["exchange_time"] = datetime.fromisoformat(exchange_time)
    return data  # type: ignore[return-value]


class CheckpointRecord:
    """Loaded checkpoint metadata + state."""

    __slots__ = ("checkpoint_id", "session_id", "created_at", "state")

    def __init__(
        self,
        *,
        checkpoint_id: str,
        session_id: str,
        created_at: datetime,
        state: JokerGraphState,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.session_id = session_id
        self.created_at = created_at
        self.state = state


@runtime_checkable
class CheckpointStore(Protocol):
    """Protocol for persisting and restoring JokerGraphState."""

    async def save(
        self,
        state: JokerGraphState,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> str:
        """Persist state; return checkpoint_id."""
        ...

    async def load_latest(self, session_id: str) -> CheckpointRecord | None:
        """Load the newest checkpoint for a session, if any."""
        ...

    async def load(self, checkpoint_id: str) -> CheckpointRecord | None:
        """Load a checkpoint by id."""
        ...

    async def list_by_session(self, session_id: str) -> list[CheckpointRecord]:
        """List checkpoints for a session, newest first."""
        ...


class SqliteCheckpointStore:
    """Simple aiosqlite JSON checkpoint table (SQLite-backed Task 1 store)."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open connection and ensure schema exists."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError(
                "SqliteCheckpointStore is not initialized; call initialize() first"
            )
        return self._conn

    async def save(
        self,
        state: JokerGraphState,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> str:
        if not session_id.strip():
            raise ValueError("session_id is required")
        cid = checkpoint_id or str(uuid4())
        created = _utc_now().isoformat()
        payload = _serialize_state(state)
        conn = self._require_conn()
        await conn.execute(
            """
            INSERT INTO graph_checkpoints (checkpoint_id, session_id, created_at, state_json)
            VALUES (?, ?, ?, ?)
            """,
            (cid, session_id, created, payload),
        )
        await conn.commit()
        return cid

    async def load_latest(self, session_id: str) -> CheckpointRecord | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT checkpoint_id, session_id, created_at, state_json
            FROM graph_checkpoints
            WHERE session_id = ?
            ORDER BY created_at DESC, checkpoint_id DESC
            LIMIT 1
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def load(self, checkpoint_id: str) -> CheckpointRecord | None:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT checkpoint_id, session_id, created_at, state_json
            FROM graph_checkpoints
            WHERE checkpoint_id = ?
            """,
            (checkpoint_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def list_by_session(self, session_id: str) -> list[CheckpointRecord]:
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT checkpoint_id, session_id, created_at, state_json
            FROM graph_checkpoints
            WHERE session_id = ?
            ORDER BY created_at DESC, checkpoint_id DESC
            """,
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> CheckpointRecord:
        return CheckpointRecord(
            checkpoint_id=row["checkpoint_id"],
            session_id=row["session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            state=_deserialize_state(row["state_json"]),
        )

    def save_sync(
        self,
        state: JokerGraphState,
        session_id: str,
        checkpoint_id: str | None = None,
    ) -> str:
        """Synchronous wrapper around ``save``."""
        return _run_sync(self._ensure_save(state, session_id, checkpoint_id))

    async def _ensure_save(
        self,
        state: JokerGraphState,
        session_id: str,
        checkpoint_id: str | None,
    ) -> str:
        if self._conn is None:
            await self.initialize()
        return await self.save(state, session_id, checkpoint_id)

    def load_latest_sync(self, session_id: str) -> CheckpointRecord | None:
        """Synchronous wrapper around ``load_latest``."""
        return _run_sync(self._ensure_load_latest(session_id))

    async def _ensure_load_latest(self, session_id: str) -> CheckpointRecord | None:
        if self._conn is None:
            await self.initialize()
        return await self.load_latest(session_id)

    def load_sync(self, checkpoint_id: str) -> CheckpointRecord | None:
        """Synchronous wrapper around ``load``."""
        return _run_sync(self._ensure_load(checkpoint_id))

    async def _ensure_load(self, checkpoint_id: str) -> CheckpointRecord | None:
        if self._conn is None:
            await self.initialize()
        return await self.load(checkpoint_id)


def _run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code; nest-safe when a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "sync checkpoint helpers cannot be called from a running event loop; "
        "use the async methods instead"
    )

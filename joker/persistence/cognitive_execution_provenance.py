"""Durable cognitive execution provenance — maps Task 1 order/position events."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS cognitive_execution_provenance (
    client_order_id TEXT PRIMARY KEY NOT NULL,
    proposal_id TEXT,
    decision_id TEXT,
    strategy_id TEXT,
    cycle_id TEXT,
    snapshot_id TEXT,
    contract_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL DEFAULT 'entry',
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cog_exec_prov_contract
    ON cognitive_execution_provenance (contract_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cog_exec_prov_session
    ON cognitive_execution_provenance (session_id, created_at);
"""


@dataclass(frozen=True)
class ExecutionProvenanceRecord:
    """Mapping from a Task 1 client_order_id to cognitive artefact IDs."""

    client_order_id: str
    proposal_id: str | None = None
    decision_id: str | None = None
    strategy_id: str | None = None
    cycle_id: str | None = None
    snapshot_id: str | None = None
    contract_id: str | None = None
    session_id: str | None = None
    kind: str = "entry"
    created_at: str | None = None
    extra: dict[str, Any] | None = None


class CognitiveExecutionProvenanceRegistry:
    """Append-only registry resolving Task 1 events back to cognitive provenance."""

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

    async def record(self, record: ExecutionProvenanceRecord) -> None:
        await self._ensure()
        created = record.created_at or datetime.now(timezone.utc).isoformat()
        payload = asdict(record)
        payload["created_at"] = created
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cognitive_execution_provenance (
                    client_order_id, proposal_id, decision_id, strategy_id,
                    cycle_id, snapshot_id, contract_id, session_id, kind,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.client_order_id,
                    record.proposal_id,
                    record.decision_id,
                    record.strategy_id,
                    record.cycle_id,
                    record.snapshot_id,
                    record.contract_id,
                    record.session_id,
                    record.kind,
                    created,
                    json.dumps(record.extra or {}),
                ),
            )
            await db.commit()

    async def get_by_client_order_id(
        self, client_order_id: str
    ) -> ExecutionProvenanceRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM cognitive_execution_provenance WHERE client_order_id = ?",
                (client_order_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return ExecutionProvenanceRecord(
            client_order_id=row["client_order_id"],
            proposal_id=row["proposal_id"],
            decision_id=row["decision_id"],
            strategy_id=row["strategy_id"],
            cycle_id=row["cycle_id"],
            snapshot_id=row["snapshot_id"],
            contract_id=row["contract_id"],
            session_id=row["session_id"],
            kind=row["kind"] or "entry",
            created_at=row["created_at"],
            extra=json.loads(row["payload_json"] or "{}"),
        )

    async def get_latest_by_contract_id(
        self, contract_id: str
    ) -> ExecutionProvenanceRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM cognitive_execution_provenance
                WHERE contract_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (contract_id,),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return ExecutionProvenanceRecord(
            client_order_id=row["client_order_id"],
            proposal_id=row["proposal_id"],
            decision_id=row["decision_id"],
            strategy_id=row["strategy_id"],
            cycle_id=row["cycle_id"],
            snapshot_id=row["snapshot_id"],
            contract_id=row["contract_id"],
            session_id=row["session_id"],
            kind=row["kind"] or "entry",
            created_at=row["created_at"],
            extra=json.loads(row["payload_json"] or "{}"),
        )

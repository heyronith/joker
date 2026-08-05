"""Durable cognitive execution provenance — maps Task 1 order/position events."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    position_lifecycle_id TEXT,
    originating_entry_client_order_id TEXT,
    parent_client_order_id TEXT,
    causation_event_id TEXT,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_cog_exec_prov_contract
    ON cognitive_execution_provenance (contract_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cog_exec_prov_session
    ON cognitive_execution_provenance (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cog_exec_prov_lifecycle
    ON cognitive_execution_provenance (position_lifecycle_id, created_at);
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
    position_lifecycle_id: str | None = None
    originating_entry_client_order_id: str | None = None
    parent_client_order_id: str | None = None
    causation_event_id: str | None = None
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
            for col, typ in (
                ("position_lifecycle_id", "TEXT"),
                ("originating_entry_client_order_id", "TEXT"),
                ("parent_client_order_id", "TEXT"),
                ("causation_event_id", "TEXT"),
            ):
                try:
                    await db.execute(
                        f"ALTER TABLE cognitive_execution_provenance "
                        f"ADD COLUMN {col} {typ}"
                    )
                except Exception:
                    pass
            await db.commit()
        self._initialized = True

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def record(self, record: ExecutionProvenanceRecord) -> None:
        await self._ensure()
        created = record.created_at or datetime.now(timezone.utc).isoformat()
        extra = dict(record.extra or {})
        if record.position_lifecycle_id:
            extra.setdefault("position_lifecycle_id", record.position_lifecycle_id)
        if record.originating_entry_client_order_id:
            extra.setdefault(
                "originating_entry_client_order_id",
                record.originating_entry_client_order_id,
            )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO cognitive_execution_provenance (
                    client_order_id, proposal_id, decision_id, strategy_id,
                    cycle_id, snapshot_id, contract_id, session_id, kind,
                    position_lifecycle_id, originating_entry_client_order_id,
                    parent_client_order_id, causation_event_id,
                    created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.position_lifecycle_id,
                    record.originating_entry_client_order_id,
                    record.parent_client_order_id,
                    record.causation_event_id,
                    created,
                    json.dumps(extra),
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
        return self._row_to_record(row)

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
        return self._row_to_record(row)

    async def list_by_target_portfolio_decision_id(
        self, target_portfolio_decision_id: str
    ) -> list[ExecutionProvenanceRecord]:
        """Return submitted portfolio components in deterministic component order."""
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM cognitive_execution_provenance
                WHERE json_extract(
                    payload_json, '$.target_portfolio_decision_id'
                ) = ?
                ORDER BY CAST(
                    json_extract(payload_json, '$.component_index') AS INTEGER
                ) ASC
                """,
                (str(target_portfolio_decision_id),),
            )
            rows = await cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def list_by_lifecycle_id(
        self, position_lifecycle_id: str
    ) -> list[ExecutionProvenanceRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """
                SELECT * FROM cognitive_execution_provenance
                WHERE position_lifecycle_id = ?
                ORDER BY created_at ASC
                """,
                (position_lifecycle_id,),
            )
            rows = await cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> ExecutionProvenanceRecord:
        keys = set(row.keys())
        extra = json.loads(row["payload_json"] or "{}")
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
            position_lifecycle_id=(
                row["position_lifecycle_id"]
                if "position_lifecycle_id" in keys
                else extra.get("position_lifecycle_id")
            ),
            originating_entry_client_order_id=(
                row["originating_entry_client_order_id"]
                if "originating_entry_client_order_id" in keys
                else extra.get("originating_entry_client_order_id")
            ),
            parent_client_order_id=(
                row["parent_client_order_id"]
                if "parent_client_order_id" in keys
                else extra.get("parent_client_order_id")
            ),
            causation_event_id=(
                row["causation_event_id"]
                if "causation_event_id" in keys
                else extra.get("causation_event_id")
            ),
            created_at=row["created_at"],
            extra=extra,
        )

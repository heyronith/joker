"""Durable cognitive execution provenance — maps Task 1 order/position events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

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

CREATE TABLE IF NOT EXISTS portfolio_execution_components (
    target_portfolio_decision_id TEXT NOT NULL,
    selected_portfolio_id TEXT,
    authorized_position_tuple_id TEXT PRIMARY KEY NOT NULL,
    component_index INTEGER NOT NULL,
    component_count INTEGER NOT NULL,
    strategy_id TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    authorized_quantity INTEGER NOT NULL,
    capital_allocation TEXT NOT NULL,
    client_order_id TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    status TEXT NOT NULL,
    submitted_quantity INTEGER NOT NULL DEFAULT 0,
    filled_quantity INTEGER NOT NULL DEFAULT 0,
    remaining_quantity INTEGER NOT NULL,
    original_decision_snapshot_id TEXT NOT NULL,
    latest_validation_snapshot_id TEXT,
    evaluated_objective_version INTEGER NOT NULL,
    submission_objective_version INTEGER,
    evaluated_timestamp TEXT NOT NULL,
    last_validation_timestamp TEXT,
    last_reconciliation_timestamp TEXT,
    failure_reoptimization_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_component_order
    ON portfolio_execution_components (
        target_portfolio_decision_id, component_index
    );
CREATE INDEX IF NOT EXISTS idx_portfolio_component_status
    ON portfolio_execution_components (status, updated_at);
"""


class PortfolioComponentStatus(StrEnum):
    AUTHORIZED = "AUTHORIZED"
    READY = "READY"
    SUBMITTED = "SUBMITTED"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    REOPTIMIZATION_REQUIRED = "REOPTIMIZATION_REQUIRED"


_COMPONENT_TRANSITIONS: dict[PortfolioComponentStatus, frozenset[PortfolioComponentStatus]] = {
    PortfolioComponentStatus.AUTHORIZED: frozenset(
        {
            PortfolioComponentStatus.READY,
            PortfolioComponentStatus.SUBMITTED,
            PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        }
    ),
    PortfolioComponentStatus.READY: frozenset(
        {
            PortfolioComponentStatus.SUBMITTED,
            PortfolioComponentStatus.WORKING,
            PortfolioComponentStatus.PARTIALLY_FILLED,
            PortfolioComponentStatus.FILLED,
            PortfolioComponentStatus.REJECTED,
            PortfolioComponentStatus.CANCELLED,
            PortfolioComponentStatus.REOPTIMIZATION_REQUIRED,
        }
    ),
    PortfolioComponentStatus.SUBMITTED: frozenset(
        {
            PortfolioComponentStatus.WORKING,
            PortfolioComponentStatus.PARTIALLY_FILLED,
            PortfolioComponentStatus.FILLED,
            PortfolioComponentStatus.REJECTED,
            PortfolioComponentStatus.CANCELLED,
        }
    ),
    PortfolioComponentStatus.WORKING: frozenset(
        {
            PortfolioComponentStatus.PARTIALLY_FILLED,
            PortfolioComponentStatus.FILLED,
            PortfolioComponentStatus.REJECTED,
            PortfolioComponentStatus.CANCELLED,
        }
    ),
    PortfolioComponentStatus.PARTIALLY_FILLED: frozenset(
        {
            PortfolioComponentStatus.PARTIALLY_FILLED,
            PortfolioComponentStatus.FILLED,
            PortfolioComponentStatus.REJECTED,
            PortfolioComponentStatus.CANCELLED,
        }
    ),
    PortfolioComponentStatus.FILLED: frozenset(),
    PortfolioComponentStatus.REJECTED: frozenset(),
    PortfolioComponentStatus.CANCELLED: frozenset(),
    PortfolioComponentStatus.REOPTIMIZATION_REQUIRED: frozenset(),
}


def stable_portfolio_client_order_id(
    target_portfolio_decision_id: str,
    authorized_position_tuple_id: str,
) -> str:
    """Return a stable broker-safe identity for one immutable authorized tuple."""
    return uuid5(
        NAMESPACE_URL,
        "joker:portfolio-component:"
        f"{target_portfolio_decision_id}:{authorized_position_tuple_id}",
    ).hex


@dataclass(frozen=True)
class PortfolioExecutionComponentRecord:
    target_portfolio_decision_id: str
    selected_portfolio_id: str | None
    authorized_position_tuple_id: str
    component_index: int
    component_count: int
    strategy_id: str
    contract_id: str
    authorized_quantity: int
    capital_allocation: Decimal
    client_order_id: str
    status: PortfolioComponentStatus
    remaining_quantity: int
    original_decision_snapshot_id: str
    evaluated_objective_version: int
    evaluated_timestamp: str
    broker_order_id: str | None = None
    submitted_quantity: int = 0
    filled_quantity: int = 0
    latest_validation_snapshot_id: str | None = None
    submission_objective_version: int | None = None
    last_validation_timestamp: str | None = None
    last_reconciliation_timestamp: str | None = None
    failure_reoptimization_reason: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.component_count < 1:
            raise ValueError("component_count must be >= 1")
        if not 0 <= self.component_index < self.component_count:
            raise ValueError("component_index must be within component_count")
        if self.authorized_quantity < 1:
            raise ValueError("authorized_quantity must be >= 1")
        if self.capital_allocation < 0:
            raise ValueError("capital_allocation must be >= 0")
        if not 0 <= self.filled_quantity <= self.submitted_quantity <= self.authorized_quantity:
            raise ValueError("invalid submitted or filled component quantities")
        if self.remaining_quantity != self.authorized_quantity - self.filled_quantity:
            raise ValueError("remaining_quantity does not match authorized minus filled")
        if not self.client_order_id:
            raise ValueError("client_order_id is required")
        evaluated = datetime.fromisoformat(self.evaluated_timestamp)
        if evaluated.tzinfo is None:
            raise ValueError("evaluated_timestamp must be timezone-aware")

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "capital_allocation": str(self.capital_allocation),
            "status": self.status.value,
            "extra": dict(self.extra or {}),
        }


class PortfolioExecutionRepository:
    """Durable sequential component state colocated with cognitive provenance."""

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

    async def authorize(
        self, record: PortfolioExecutionComponentRecord
    ) -> PortfolioExecutionComponentRecord:
        """Insert once and reject any later mutation of immutable authority."""
        await self._ensure()
        now = record.created_at or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO portfolio_execution_components (
                    target_portfolio_decision_id, selected_portfolio_id,
                    authorized_position_tuple_id, component_index, component_count,
                    strategy_id, contract_id, authorized_quantity, capital_allocation,
                    client_order_id, broker_order_id, status, submitted_quantity,
                    filled_quantity, remaining_quantity,
                    original_decision_snapshot_id, latest_validation_snapshot_id,
                    evaluated_objective_version, submission_objective_version,
                    evaluated_timestamp, last_validation_timestamp,
                    last_reconciliation_timestamp, failure_reoptimization_reason,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.target_portfolio_decision_id,
                    record.selected_portfolio_id,
                    record.authorized_position_tuple_id,
                    record.component_index,
                    record.component_count,
                    record.strategy_id,
                    record.contract_id,
                    record.authorized_quantity,
                    str(record.capital_allocation),
                    record.client_order_id,
                    record.broker_order_id,
                    record.status.value,
                    record.submitted_quantity,
                    record.filled_quantity,
                    record.remaining_quantity,
                    record.original_decision_snapshot_id,
                    record.latest_validation_snapshot_id,
                    record.evaluated_objective_version,
                    record.submission_objective_version,
                    record.evaluated_timestamp,
                    record.last_validation_timestamp,
                    record.last_reconciliation_timestamp,
                    record.failure_reoptimization_reason,
                    now,
                    record.updated_at or now,
                    json.dumps(record.extra or {}, sort_keys=True),
                ),
            )
            await db.commit()
        stored = await self.get(record.authorized_position_tuple_id)
        assert stored is not None
        immutable = (
            "target_portfolio_decision_id",
            "selected_portfolio_id",
            "component_index",
            "component_count",
            "strategy_id",
            "contract_id",
            "authorized_quantity",
            "capital_allocation",
            "client_order_id",
            "original_decision_snapshot_id",
            "evaluated_objective_version",
            "evaluated_timestamp",
        )
        if any(getattr(stored, name) != getattr(record, name) for name in immutable):
            raise ValueError(
                "authorized portfolio component conflicts with durable authority"
            )
        return stored

    async def transition(
        self,
        authorized_position_tuple_id: str,
        *,
        status: PortfolioComponentStatus,
        broker_order_id: str | None = None,
        submitted_quantity: int | None = None,
        filled_quantity: int | None = None,
        latest_validation_snapshot_id: str | None = None,
        submission_objective_version: int | None = None,
        last_validation_timestamp: str | None = None,
        last_reconciliation_timestamp: str | None = None,
        failure_reoptimization_reason: str | None = None,
        extra_update: dict[str, Any] | None = None,
    ) -> PortfolioExecutionComponentRecord:
        await self._ensure()
        existing = await self.get(authorized_position_tuple_id)
        if existing is None:
            raise KeyError(
                f"portfolio component not found: {authorized_position_tuple_id}"
            )
        if status != existing.status and status not in _COMPONENT_TRANSITIONS[existing.status]:
            raise ValueError(
                f"invalid portfolio component transition {existing.status}->{status}"
            )
        submitted = (
            existing.submitted_quantity
            if submitted_quantity is None
            else int(submitted_quantity)
        )
        filled = (
            existing.filled_quantity
            if filled_quantity is None
            else int(filled_quantity)
        )
        if submitted < 0 or filled < 0 or filled > existing.authorized_quantity:
            raise ValueError("invalid portfolio component quantities")
        if submitted > existing.authorized_quantity:
            raise ValueError("submitted quantity exceeds authorized quantity")
        if filled > submitted:
            raise ValueError("filled quantity exceeds submitted quantity")
        remaining = max(0, existing.authorized_quantity - filled)
        extra = {**dict(existing.extra or {}), **dict(extra_update or {})}
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE portfolio_execution_components
                SET status = ?, broker_order_id = COALESCE(?, broker_order_id),
                    submitted_quantity = ?, filled_quantity = ?, remaining_quantity = ?,
                    latest_validation_snapshot_id = COALESCE(?, latest_validation_snapshot_id),
                    submission_objective_version = COALESCE(?, submission_objective_version),
                    last_validation_timestamp = COALESCE(?, last_validation_timestamp),
                    last_reconciliation_timestamp = COALESCE(?, last_reconciliation_timestamp),
                    failure_reoptimization_reason = COALESCE(?, failure_reoptimization_reason),
                    updated_at = ?, payload_json = ?
                WHERE authorized_position_tuple_id = ?
                """,
                (
                    status.value,
                    broker_order_id,
                    submitted,
                    filled,
                    remaining,
                    latest_validation_snapshot_id,
                    submission_objective_version,
                    last_validation_timestamp,
                    last_reconciliation_timestamp,
                    failure_reoptimization_reason,
                    now,
                    json.dumps(extra, sort_keys=True),
                    authorized_position_tuple_id,
                ),
            )
            await db.commit()
        stored = await self.get(authorized_position_tuple_id)
        assert stored is not None
        return stored

    async def get(
        self, authorized_position_tuple_id: str
    ) -> PortfolioExecutionComponentRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE authorized_position_tuple_id = ?""",
                (authorized_position_tuple_id,),
            )
            row = await cur.fetchone()
        return self._row(row) if row is not None else None

    async def get_by_client_order_id(
        self, client_order_id: str
    ) -> PortfolioExecutionComponentRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE client_order_id = ?""",
                (client_order_id,),
            )
            row = await cur.fetchone()
        return self._row(row) if row is not None else None

    async def list_by_decision(
        self, target_portfolio_decision_id: str
    ) -> list[PortfolioExecutionComponentRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE target_portfolio_decision_id = ?
                ORDER BY component_index ASC""",
                (target_portfolio_decision_id,),
            )
            rows = await cur.fetchall()
        return [self._row(row) for row in rows]

    async def list_resumable(self) -> list[PortfolioExecutionComponentRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE status IN ('AUTHORIZED', 'READY', 'SUBMITTED', 'WORKING', 'PARTIALLY_FILLED')
                ORDER BY target_portfolio_decision_id, component_index"""
            )
            rows = await cur.fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: aiosqlite.Row) -> PortfolioExecutionComponentRecord:
        return PortfolioExecutionComponentRecord(
            target_portfolio_decision_id=row["target_portfolio_decision_id"],
            selected_portfolio_id=row["selected_portfolio_id"],
            authorized_position_tuple_id=row["authorized_position_tuple_id"],
            component_index=int(row["component_index"]),
            component_count=int(row["component_count"]),
            strategy_id=row["strategy_id"],
            contract_id=row["contract_id"],
            authorized_quantity=int(row["authorized_quantity"]),
            capital_allocation=Decimal(row["capital_allocation"]),
            client_order_id=row["client_order_id"],
            broker_order_id=row["broker_order_id"],
            status=PortfolioComponentStatus(row["status"]),
            submitted_quantity=int(row["submitted_quantity"]),
            filled_quantity=int(row["filled_quantity"]),
            remaining_quantity=int(row["remaining_quantity"]),
            original_decision_snapshot_id=row["original_decision_snapshot_id"],
            latest_validation_snapshot_id=row["latest_validation_snapshot_id"],
            evaluated_objective_version=int(row["evaluated_objective_version"]),
            submission_objective_version=(
                int(row["submission_objective_version"])
                if row["submission_objective_version"] is not None
                else None
            ),
            evaluated_timestamp=row["evaluated_timestamp"],
            last_validation_timestamp=row["last_validation_timestamp"],
            last_reconciliation_timestamp=row["last_reconciliation_timestamp"],
            failure_reoptimization_reason=row["failure_reoptimization_reason"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            extra=json.loads(row["payload_json"] or "{}"),
        )


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
        self.portfolio_executions = PortfolioExecutionRepository(self._db_path)

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
        self.portfolio_executions._initialized = True

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

"""Durable cognitive execution provenance — maps Task 1 order/position events."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    broker_account_id TEXT NOT NULL,
    trading_date TEXT NOT NULL,
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
    post_fill_objective_version INTEGER,
    post_fill_objective_fingerprint TEXT,
    post_fill_snapshot_id TEXT,
    post_fill_exchange_time TEXT,
    reconciled_filled_quantity INTEGER,
    continuation_ready INTEGER NOT NULL DEFAULT 0,
    state_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_portfolio_component_status
    ON portfolio_execution_components (status, updated_at);
CREATE TABLE IF NOT EXISTS portfolio_reoptimization_requests (
    request_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    broker_account_id TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    original_portfolio_decision_id TEXT NOT NULL,
    already_filled_tuple_ids_json TEXT NOT NULL,
    open_positions_json TEXT NOT NULL,
    remaining_authorized_tuple_ids_json TEXT NOT NULL,
    reason_codes_json TEXT NOT NULL,
    latest_objective_state_json TEXT NOT NULL,
    latest_objective_version INTEGER NOT NULL,
    latest_snapshot_id TEXT NOT NULL,
    created_exchange_time TEXT NOT NULL,
    status TEXT NOT NULL,
    replacement_decision_id TEXT,
    replacement_action TEXT,
    state_version INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_portfolio_reopt_owner_status
    ON portfolio_reoptimization_requests (
        session_id, broker_account_id, trading_date, status, created_at
    );
"""


_PORTFOLIO_COMPONENT_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("session_id", "TEXT"),
    ("run_id", "TEXT"),
    ("broker_account_id", "TEXT"),
    ("trading_date", "TEXT"),
    ("post_fill_objective_version", "INTEGER"),
    ("post_fill_objective_fingerprint", "TEXT"),
    ("post_fill_snapshot_id", "TEXT"),
    ("post_fill_exchange_time", "TEXT"),
    ("reconciled_filled_quantity", "INTEGER"),
    ("continuation_ready", "INTEGER NOT NULL DEFAULT 0"),
    ("state_version", "INTEGER NOT NULL DEFAULT 0"),
)


def apply_portfolio_execution_migration(db_path: str | Path) -> None:
    """Forward-migrate durable portfolio execution state, idempotently.

    SQLite cannot add new ``NOT NULL`` ownership columns to populated tables
    without inventing an owner.  The columns therefore remain nullable only for
    legacy rows; those rows are immediately failed closed and never returned by
    ownership-scoped resume queries.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(_CREATE_SQL)
        columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(portfolio_execution_components)")
        }
        for name, declaration in _PORTFOLIO_COMPONENT_MIGRATION_COLUMNS:
            if name not in columns:
                db.execute(
                    f"ALTER TABLE portfolio_execution_components ADD COLUMN {name} {declaration}"
                )
        db.execute(
            """
            UPDATE portfolio_execution_components
            SET status = 'REOPTIMIZATION_REQUIRED',
                failure_reoptimization_reason = COALESCE(
                    failure_reoptimization_reason,
                    'legacy_unscoped_portfolio_component'
                ),
                state_version = state_version + 1,
                updated_at = ?
            WHERE (
                session_id IS NULL OR session_id = ''
                OR run_id IS NULL OR run_id = ''
                OR broker_account_id IS NULL OR broker_account_id = ''
                OR trading_date IS NULL OR trading_date = ''
            )
              AND (
                status != 'REOPTIMIZATION_REQUIRED'
                OR failure_reoptimization_reason IS NULL
              )
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        db.executescript(
            """
            DROP INDEX IF EXISTS idx_portfolio_component_order;
            CREATE UNIQUE INDEX idx_portfolio_component_order
                ON portfolio_execution_components (
                    session_id, run_id, broker_account_id, trading_date,
                    target_portfolio_decision_id, component_index
                );
            CREATE INDEX IF NOT EXISTS idx_portfolio_component_owner_status
                ON portfolio_execution_components (
                    session_id, broker_account_id, trading_date, status, updated_at
                );
            CREATE INDEX IF NOT EXISTS idx_portfolio_reopt_owner_status
                ON portfolio_reoptimization_requests (
                    session_id, broker_account_id, trading_date, status, created_at
                );
            """
        )
        db.commit()


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


class PortfolioTransitionConflict(RuntimeError):
    """A stale component writer lost the compare-and-swap race."""


@dataclass(frozen=True)
class PortfolioExecutionOwner:
    session_id: str
    run_id: str
    broker_account_id: str
    trading_date: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.session_id,
                self.run_id,
                self.broker_account_id,
                self.trading_date,
            )
        ):
            raise ValueError("portfolio execution ownership is incomplete")
        date.fromisoformat(self.trading_date)

    def matches(self, record: PortfolioExecutionComponentRecord) -> bool:
        return (
            record.session_id == self.session_id
            and record.run_id == self.run_id
            and record.broker_account_id == self.broker_account_id
            and record.trading_date == self.trading_date
        )


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
        f"joker:portfolio-component:{target_portfolio_decision_id}:{authorized_position_tuple_id}",
    ).hex


@dataclass(frozen=True)
class PortfolioExecutionComponentRecord:
    session_id: str | None
    run_id: str | None
    broker_account_id: str | None
    trading_date: str | None
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
    post_fill_objective_version: int | None = None
    post_fill_objective_fingerprint: str | None = None
    post_fill_snapshot_id: str | None = None
    post_fill_exchange_time: str | None = None
    reconciled_filled_quantity: int | None = None
    continuation_ready: bool = False
    state_version: int = 0
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
        if self.trading_date:
            date.fromisoformat(self.trading_date)
        if self.state_version < 0:
            raise ValueError("state_version must be non-negative")
        if self.status == PortfolioComponentStatus.FILLED and (
            self.filled_quantity != self.authorized_quantity
        ):
            raise ValueError("FILLED requires the full authorized quantity")
        if self.status == PortfolioComponentStatus.PARTIALLY_FILLED and not (
            0 < self.filled_quantity < self.submitted_quantity
        ):
            raise ValueError("PARTIALLY_FILLED requires 0 < filled quantity < submitted quantity")
        if self.continuation_ready:
            required = (
                self.post_fill_objective_version,
                self.post_fill_objective_fingerprint,
                self.post_fill_snapshot_id,
                self.post_fill_exchange_time,
                self.reconciled_filled_quantity,
            )
            if self.status != PortfolioComponentStatus.FILLED or any(
                value is None or value == "" for value in required
            ):
                raise ValueError("continuation checkpoint is incomplete")
            if self.reconciled_filled_quantity != self.authorized_quantity:
                raise ValueError("continuation checkpoint quantity is not authoritative")

    @property
    def has_scoped_owner(self) -> bool:
        return all(
            (
                self.session_id,
                self.run_id,
                self.broker_account_id,
                self.trading_date,
            )
        )

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
        apply_portfolio_execution_migration(self._db_path)
        self._initialized = True

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def authorize(
        self, record: PortfolioExecutionComponentRecord
    ) -> PortfolioExecutionComponentRecord:
        """Insert once and reject any later mutation of immutable authority."""
        await self._ensure()
        if not record.has_scoped_owner:
            raise ValueError("authorized portfolio component requires durable ownership")
        now = record.created_at or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO portfolio_execution_components (
                    session_id, run_id, broker_account_id, trading_date,
                    target_portfolio_decision_id, selected_portfolio_id,
                    authorized_position_tuple_id, component_index, component_count,
                    strategy_id, contract_id, authorized_quantity, capital_allocation,
                    client_order_id, broker_order_id, status, submitted_quantity,
                    filled_quantity, remaining_quantity,
                    original_decision_snapshot_id, latest_validation_snapshot_id,
                    evaluated_objective_version, submission_objective_version,
                    evaluated_timestamp, last_validation_timestamp,
                    last_reconciliation_timestamp, failure_reoptimization_reason,
                    post_fill_objective_version,
                    post_fill_objective_fingerprint, post_fill_snapshot_id,
                    post_fill_exchange_time, reconciled_filled_quantity,
                    continuation_ready, state_version,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.run_id,
                    record.broker_account_id,
                    record.trading_date,
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
                    record.post_fill_objective_version,
                    record.post_fill_objective_fingerprint,
                    record.post_fill_snapshot_id,
                    record.post_fill_exchange_time,
                    record.reconciled_filled_quantity,
                    int(record.continuation_ready),
                    record.state_version,
                    now,
                    record.updated_at or now,
                    json.dumps(record.extra or {}, sort_keys=True),
                ),
            )
            await db.commit()
        stored = await self.get(record.authorized_position_tuple_id)
        assert stored is not None
        immutable = (
            "session_id",
            "run_id",
            "broker_account_id",
            "trading_date",
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
            raise ValueError("authorized portfolio component conflicts with durable authority")
        return stored

    async def transition(
        self,
        authorized_position_tuple_id: str,
        *,
        status: PortfolioComponentStatus,
        owner: PortfolioExecutionOwner,
        broker_order_id: str | None = None,
        submitted_quantity: int | None = None,
        filled_quantity: int | None = None,
        latest_validation_snapshot_id: str | None = None,
        submission_objective_version: int | None = None,
        last_validation_timestamp: str | None = None,
        last_reconciliation_timestamp: str | None = None,
        failure_reoptimization_reason: str | None = None,
        post_fill_objective_version: int | None = None,
        post_fill_objective_fingerprint: str | None = None,
        post_fill_snapshot_id: str | None = None,
        post_fill_exchange_time: str | None = None,
        reconciled_filled_quantity: int | None = None,
        continuation_ready: bool | None = None,
        expected_state_version: int | None = None,
        expected_status: PortfolioComponentStatus | None = None,
        extra_update: dict[str, Any] | None = None,
    ) -> PortfolioExecutionComponentRecord:
        await self._ensure()
        existing = await self.get(authorized_position_tuple_id)
        if existing is None:
            raise KeyError(f"portfolio component not found: {authorized_position_tuple_id}")
        if not owner.matches(existing):
            raise PermissionError("portfolio component owner does not match runtime")
        if not existing.has_scoped_owner:
            raise PermissionError("legacy unscoped portfolio component is non-resumable")
        if expected_state_version is not None and (
            existing.state_version != expected_state_version
        ):
            raise PortfolioTransitionConflict("portfolio component version changed")
        if expected_status is not None and existing.status != expected_status:
            raise PortfolioTransitionConflict("portfolio component status changed")
        if status != existing.status and status not in _COMPONENT_TRANSITIONS[existing.status]:
            raise ValueError(f"invalid portfolio component transition {existing.status}->{status}")
        submitted = (
            existing.submitted_quantity if submitted_quantity is None else int(submitted_quantity)
        )
        filled = existing.filled_quantity if filled_quantity is None else int(filled_quantity)
        if submitted < existing.submitted_quantity:
            raise ValueError("submitted quantity cannot decrease")
        if filled < existing.filled_quantity:
            raise ValueError("filled quantity cannot decrease")
        if submitted < 0 or filled < 0 or filled > existing.authorized_quantity:
            raise ValueError("invalid portfolio component quantities")
        if submitted > existing.authorized_quantity:
            raise ValueError("submitted quantity exceeds authorized quantity")
        if filled > submitted:
            raise ValueError("filled quantity exceeds submitted quantity")
        if status == PortfolioComponentStatus.FILLED and (filled != existing.authorized_quantity):
            raise ValueError("FILLED requires filled quantity == authorized quantity")
        if status == PortfolioComponentStatus.PARTIALLY_FILLED and not (0 < filled < submitted):
            raise ValueError("PARTIALLY_FILLED requires 0 < filled quantity < submitted quantity")
        remaining = max(0, existing.authorized_quantity - filled)
        extra = {**dict(existing.extra or {}), **dict(extra_update or {})}
        continuation = (
            existing.continuation_ready if continuation_ready is None else bool(continuation_ready)
        )
        continuation_fields = (
            post_fill_objective_version
            if post_fill_objective_version is not None
            else existing.post_fill_objective_version,
            post_fill_objective_fingerprint
            if post_fill_objective_fingerprint is not None
            else existing.post_fill_objective_fingerprint,
            post_fill_snapshot_id
            if post_fill_snapshot_id is not None
            else existing.post_fill_snapshot_id,
            post_fill_exchange_time
            if post_fill_exchange_time is not None
            else existing.post_fill_exchange_time,
            reconciled_filled_quantity
            if reconciled_filled_quantity is not None
            else existing.reconciled_filled_quantity,
        )
        if continuation and (
            status != PortfolioComponentStatus.FILLED
            or any(value is None or value == "" for value in continuation_fields)
            or int(continuation_fields[4]) != existing.authorized_quantity
        ):
            raise ValueError("post-fill continuation checkpoint is incomplete")
        desired = (
            status,
            broker_order_id or existing.broker_order_id,
            submitted,
            filled,
            latest_validation_snapshot_id or existing.latest_validation_snapshot_id,
            submission_objective_version or existing.submission_objective_version,
            last_validation_timestamp or existing.last_validation_timestamp,
            last_reconciliation_timestamp or existing.last_reconciliation_timestamp,
            failure_reoptimization_reason or existing.failure_reoptimization_reason,
            continuation_fields,
            continuation,
            extra,
        )
        current = (
            existing.status,
            existing.broker_order_id,
            existing.submitted_quantity,
            existing.filled_quantity,
            existing.latest_validation_snapshot_id,
            existing.submission_objective_version,
            existing.last_validation_timestamp,
            existing.last_reconciliation_timestamp,
            existing.failure_reoptimization_reason,
            (
                existing.post_fill_objective_version,
                existing.post_fill_objective_fingerprint,
                existing.post_fill_snapshot_id,
                existing.post_fill_exchange_time,
                existing.reconciled_filled_quantity,
            ),
            existing.continuation_ready,
            dict(existing.extra or {}),
        )
        if desired == current:
            return existing
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE portfolio_execution_components
                SET status = ?, broker_order_id = COALESCE(?, broker_order_id),
                    submitted_quantity = ?, filled_quantity = ?, remaining_quantity = ?,
                    latest_validation_snapshot_id = COALESCE(?, latest_validation_snapshot_id),
                    submission_objective_version = COALESCE(?, submission_objective_version),
                    last_validation_timestamp = COALESCE(?, last_validation_timestamp),
                    last_reconciliation_timestamp = COALESCE(?, last_reconciliation_timestamp),
                    failure_reoptimization_reason = COALESCE(?, failure_reoptimization_reason),
                    post_fill_objective_version = COALESCE(?, post_fill_objective_version),
                    post_fill_objective_fingerprint = COALESCE(?, post_fill_objective_fingerprint),
                    post_fill_snapshot_id = COALESCE(?, post_fill_snapshot_id),
                    post_fill_exchange_time = COALESCE(?, post_fill_exchange_time),
                    reconciled_filled_quantity = COALESCE(?, reconciled_filled_quantity),
                    continuation_ready = ?, state_version = state_version + 1,
                    updated_at = ?, payload_json = ?
                WHERE authorized_position_tuple_id = ?
                  AND state_version = ? AND status = ?
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
                    post_fill_objective_version,
                    post_fill_objective_fingerprint,
                    post_fill_snapshot_id,
                    post_fill_exchange_time,
                    reconciled_filled_quantity,
                    int(continuation),
                    now,
                    json.dumps(extra, sort_keys=True),
                    authorized_position_tuple_id,
                    existing.state_version,
                    existing.status.value,
                ),
            )
            await db.commit()
        if cur.rowcount != 1:
            raise PortfolioTransitionConflict(
                "portfolio component transition lost compare-and-swap race"
            )
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
        self,
        client_order_id: str,
        *,
        owner: PortfolioExecutionOwner,
    ) -> PortfolioExecutionComponentRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE client_order_id = ?
                  AND session_id = ? AND run_id = ?
                  AND broker_account_id = ? AND trading_date = ?""",
                (
                    client_order_id,
                    owner.session_id,
                    owner.run_id,
                    owner.broker_account_id,
                    owner.trading_date,
                ),
            )
            row = await cur.fetchone()
        return self._row(row) if row is not None else None

    async def list_by_decision(
        self,
        target_portfolio_decision_id: str,
        *,
        owner: PortfolioExecutionOwner | None = None,
    ) -> list[PortfolioExecutionComponentRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            if owner is None:
                cur = await db.execute(
                    """SELECT * FROM portfolio_execution_components
                    WHERE target_portfolio_decision_id = ?
                    ORDER BY component_index ASC""",
                    (target_portfolio_decision_id,),
                )
            else:
                cur = await db.execute(
                    """SELECT * FROM portfolio_execution_components
                    WHERE target_portfolio_decision_id = ?
                      AND session_id = ? AND run_id = ?
                      AND broker_account_id = ? AND trading_date = ?
                    ORDER BY component_index ASC""",
                    (
                        target_portfolio_decision_id,
                        owner.session_id,
                        owner.run_id,
                        owner.broker_account_id,
                        owner.trading_date,
                    ),
                )
            rows = await cur.fetchall()
        return [self._row(row) for row in rows]

    async def list_resumable(
        self,
        *,
        session_id: str,
        run_id: str,
        broker_account_id: str,
        trading_date: str,
    ) -> list[PortfolioExecutionComponentRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE session_id = ? AND run_id = ?
                  AND broker_account_id = ? AND trading_date = ?
                  AND status IN (
                    'AUTHORIZED', 'READY', 'SUBMITTED', 'WORKING',
                    'PARTIALLY_FILLED', 'FILLED'
                  )
                  AND EXISTS (
                    SELECT 1 FROM portfolio_execution_components pending
                    WHERE pending.target_portfolio_decision_id =
                          portfolio_execution_components.target_portfolio_decision_id
                      AND pending.session_id = portfolio_execution_components.session_id
                      AND pending.run_id = portfolio_execution_components.run_id
                      AND pending.broker_account_id =
                          portfolio_execution_components.broker_account_id
                      AND pending.trading_date =
                          portfolio_execution_components.trading_date
                      AND pending.status IN (
                        'AUTHORIZED', 'READY', 'SUBMITTED', 'WORKING',
                        'PARTIALLY_FILLED'
                      )
                  )
                ORDER BY target_portfolio_decision_id, component_index""",
                (session_id, run_id, broker_account_id, trading_date),
            )
            rows = await cur.fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: aiosqlite.Row) -> PortfolioExecutionComponentRecord:
        return PortfolioExecutionComponentRecord(
            session_id=row["session_id"],
            run_id=row["run_id"],
            broker_account_id=row["broker_account_id"],
            trading_date=row["trading_date"],
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
            post_fill_objective_version=(
                int(row["post_fill_objective_version"])
                if row["post_fill_objective_version"] is not None
                else None
            ),
            post_fill_objective_fingerprint=row["post_fill_objective_fingerprint"],
            post_fill_snapshot_id=row["post_fill_snapshot_id"],
            post_fill_exchange_time=row["post_fill_exchange_time"],
            reconciled_filled_quantity=(
                int(row["reconciled_filled_quantity"])
                if row["reconciled_filled_quantity"] is not None
                else None
            ),
            continuation_ready=bool(row["continuation_ready"]),
            state_version=int(row["state_version"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            extra=json.loads(row["payload_json"] or "{}"),
        )


class PortfolioReoptimizationStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class PortfolioReoptimizationRequestRecord:
    request_id: str
    session_id: str
    run_id: str
    broker_account_id: str
    trading_date: str
    original_portfolio_decision_id: str
    already_filled_tuple_ids: tuple[str, ...]
    open_positions: tuple[dict[str, Any], ...]
    remaining_authorized_tuple_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    latest_objective_state: dict[str, Any]
    latest_objective_version: int
    latest_snapshot_id: str
    created_exchange_time: str
    status: PortfolioReoptimizationStatus = PortfolioReoptimizationStatus.PENDING
    replacement_decision_id: str | None = None
    replacement_action: str | None = None
    state_version: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        PortfolioExecutionOwner(
            session_id=self.session_id,
            run_id=self.run_id,
            broker_account_id=self.broker_account_id,
            trading_date=self.trading_date,
        )
        created = datetime.fromisoformat(self.created_exchange_time)
        if created.tzinfo is None:
            raise ValueError("created_exchange_time must be timezone-aware")
        if not self.original_portfolio_decision_id or not self.latest_snapshot_id:
            raise ValueError("reoptimization provenance is incomplete")
        if self.latest_objective_version < 0 or self.state_version < 0:
            raise ValueError("reoptimization versions must be non-negative")


def stable_reoptimization_request_id(
    *,
    session_id: str,
    run_id: str,
    broker_account_id: str,
    trading_date: str,
    original_portfolio_decision_id: str,
    remaining_authorized_tuple_ids: tuple[str, ...],
) -> str:
    return uuid5(
        NAMESPACE_URL,
        "joker:portfolio-reoptimization:"
        f"{session_id}:{run_id}:{broker_account_id}:{trading_date}:"
        f"{original_portfolio_decision_id}:"
        + ",".join(remaining_authorized_tuple_ids),
    ).hex


class PortfolioReoptimizationRepository:
    """Durable, ownership-scoped continuation optimization work."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._initialized = False

    async def initialize(self) -> None:
        apply_portfolio_execution_migration(self._db_path)
        self._initialized = True

    async def _ensure(self) -> None:
        if not self._initialized:
            await self.initialize()

    async def enqueue(
        self, record: PortfolioReoptimizationRequestRecord
    ) -> PortfolioReoptimizationRequestRecord:
        await self._ensure()
        now = record.created_at or datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT OR IGNORE INTO portfolio_reoptimization_requests (
                    request_id, session_id, run_id, broker_account_id, trading_date,
                    original_portfolio_decision_id,
                    already_filled_tuple_ids_json, open_positions_json,
                    remaining_authorized_tuple_ids_json, reason_codes_json,
                    latest_objective_state_json, latest_objective_version,
                    latest_snapshot_id, created_exchange_time, status,
                    replacement_decision_id, replacement_action, state_version,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.session_id,
                    record.run_id,
                    record.broker_account_id,
                    record.trading_date,
                    record.original_portfolio_decision_id,
                    json.dumps(record.already_filled_tuple_ids),
                    json.dumps(record.open_positions, sort_keys=True),
                    json.dumps(record.remaining_authorized_tuple_ids),
                    json.dumps(record.reason_codes),
                    json.dumps(record.latest_objective_state, sort_keys=True),
                    record.latest_objective_version,
                    record.latest_snapshot_id,
                    record.created_exchange_time,
                    record.status.value,
                    record.replacement_decision_id,
                    record.replacement_action,
                    record.state_version,
                    now,
                    record.updated_at or now,
                    json.dumps(record.extra or {}, sort_keys=True),
                ),
            )
            await db.commit()
        stored = await self.get(record.request_id)
        assert stored is not None
        if (
            stored.session_id != record.session_id
            or stored.run_id != record.run_id
            or stored.broker_account_id != record.broker_account_id
            or stored.trading_date != record.trading_date
            or stored.original_portfolio_decision_id != record.original_portfolio_decision_id
            or stored.remaining_authorized_tuple_ids != record.remaining_authorized_tuple_ids
        ):
            raise ValueError("reoptimization request conflicts with durable authority")
        return stored

    async def get(self, request_id: str) -> PortfolioReoptimizationRequestRecord | None:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            row = await (
                await db.execute(
                    "SELECT * FROM portfolio_reoptimization_requests WHERE request_id = ?",
                    (request_id,),
                )
            ).fetchone()
        return self._row(row) if row is not None else None

    async def list_pending(
        self,
        *,
        session_id: str,
        run_id: str,
        broker_account_id: str,
        trading_date: str,
    ) -> list[PortfolioReoptimizationRequestRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    """
                    SELECT * FROM portfolio_reoptimization_requests
                    WHERE session_id = ? AND run_id = ?
                      AND broker_account_id = ? AND trading_date = ?
                      AND status IN ('PENDING', 'RUNNING')
                    ORDER BY created_at, request_id
                    """,
                    (session_id, run_id, broker_account_id, trading_date),
                )
            ).fetchall()
        return [self._row(row) for row in rows]

    async def transition(
        self,
        request_id: str,
        *,
        status: PortfolioReoptimizationStatus,
        replacement_decision_id: str | None = None,
        replacement_action: str | None = None,
        expected_state_version: int | None = None,
    ) -> PortfolioReoptimizationRequestRecord:
        existing = await self.get(request_id)
        if existing is None:
            raise KeyError(f"reoptimization request not found: {request_id}")
        if expected_state_version is not None and (
            existing.state_version != expected_state_version
        ):
            raise PortfolioTransitionConflict("reoptimization request version changed")
        allowed = {
            PortfolioReoptimizationStatus.PENDING: {
                PortfolioReoptimizationStatus.RUNNING,
                PortfolioReoptimizationStatus.FAILED,
            },
            PortfolioReoptimizationStatus.RUNNING: {
                PortfolioReoptimizationStatus.COMPLETED,
                PortfolioReoptimizationStatus.FAILED,
            },
            PortfolioReoptimizationStatus.COMPLETED: set(),
            PortfolioReoptimizationStatus.FAILED: set(),
        }
        if status == existing.status:
            return existing
        if status not in allowed[existing.status]:
            raise ValueError(f"invalid reoptimization transition {existing.status}->{status}")
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE portfolio_reoptimization_requests
                SET status = ?, replacement_decision_id = COALESCE(?, replacement_decision_id),
                    replacement_action = COALESCE(?, replacement_action),
                    state_version = state_version + 1, updated_at = ?
                WHERE request_id = ? AND state_version = ? AND status = ?
                """,
                (
                    status.value,
                    replacement_decision_id,
                    replacement_action,
                    datetime.now(timezone.utc).isoformat(),
                    request_id,
                    existing.state_version,
                    existing.status.value,
                ),
            )
            await db.commit()
        if cur.rowcount != 1:
            raise PortfolioTransitionConflict(
                "reoptimization transition lost compare-and-swap race"
            )
        stored = await self.get(request_id)
        assert stored is not None
        return stored

    @staticmethod
    def _row(row: aiosqlite.Row) -> PortfolioReoptimizationRequestRecord:
        return PortfolioReoptimizationRequestRecord(
            request_id=row["request_id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            broker_account_id=row["broker_account_id"],
            trading_date=row["trading_date"],
            original_portfolio_decision_id=row["original_portfolio_decision_id"],
            already_filled_tuple_ids=tuple(json.loads(row["already_filled_tuple_ids_json"])),
            open_positions=tuple(json.loads(row["open_positions_json"])),
            remaining_authorized_tuple_ids=tuple(
                json.loads(row["remaining_authorized_tuple_ids_json"])
            ),
            reason_codes=tuple(json.loads(row["reason_codes_json"])),
            latest_objective_state=json.loads(row["latest_objective_state_json"]),
            latest_objective_version=int(row["latest_objective_version"]),
            latest_snapshot_id=row["latest_snapshot_id"],
            created_exchange_time=row["created_exchange_time"],
            status=PortfolioReoptimizationStatus(row["status"]),
            replacement_decision_id=row["replacement_decision_id"],
            replacement_action=row["replacement_action"],
            state_version=int(row["state_version"]),
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
        self.portfolio_reoptimizations = PortfolioReoptimizationRepository(self._db_path)

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        apply_portfolio_execution_migration(self._db_path)
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
                        f"ALTER TABLE cognitive_execution_provenance ADD COLUMN {col} {typ}"
                    )
                except Exception:
                    pass
            await db.commit()
        self._initialized = True
        self.portfolio_executions._initialized = True
        self.portfolio_reoptimizations._initialized = True

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

    async def get_latest_by_contract_id(self, contract_id: str) -> ExecutionProvenanceRecord | None:
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

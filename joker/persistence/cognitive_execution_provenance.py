"""Durable cognitive execution provenance — maps Task 1 order/position events."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

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
    origin_run_id TEXT NOT NULL,
    last_resumed_run_id TEXT,
    resume_count INTEGER NOT NULL DEFAULT 0,
    last_resumed_at TEXT,
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
    component_order_conflicted INTEGER NOT NULL DEFAULT 0,
    resolution_status TEXT NOT NULL DEFAULT 'UNRESOLVED',
    resolved_at TEXT,
    resolved_by TEXT,
    resolution_reason TEXT,
    superseded_by_reoptimization_request_id TEXT,
    superseded_by_decision_id TEXT,
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
    origin_run_id TEXT NOT NULL,
    last_resumed_run_id TEXT,
    resume_count INTEGER NOT NULL DEFAULT 0,
    last_resumed_at TEXT,
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
    failure_reason TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_attempt_run_id TEXT,
    last_attempt_exchange_time TEXT,
    attempt_owner_run_id TEXT,
    attempt_started_at TEXT,
    attempt_lease_expires_at TEXT,
    attempt_heartbeat_at TEXT,
    attempt_generation INTEGER NOT NULL DEFAULT 0,
    attempt_token TEXT,
    resolution_status TEXT NOT NULL DEFAULT 'UNRESOLVED',
    resolved_at TEXT,
    resolved_by TEXT,
    resolution_reason TEXT,
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
    ("origin_run_id", "TEXT"),
    ("last_resumed_run_id", "TEXT"),
    ("resume_count", "INTEGER NOT NULL DEFAULT 0"),
    ("last_resumed_at", "TEXT"),
    ("broker_account_id", "TEXT"),
    ("trading_date", "TEXT"),
    ("post_fill_objective_version", "INTEGER"),
    ("post_fill_objective_fingerprint", "TEXT"),
    ("post_fill_snapshot_id", "TEXT"),
    ("post_fill_exchange_time", "TEXT"),
    ("reconciled_filled_quantity", "INTEGER"),
    ("continuation_ready", "INTEGER NOT NULL DEFAULT 0"),
    ("state_version", "INTEGER NOT NULL DEFAULT 0"),
    ("component_order_conflicted", "INTEGER NOT NULL DEFAULT 0"),
    ("resolution_status", "TEXT NOT NULL DEFAULT 'UNRESOLVED'"),
    ("resolved_at", "TEXT"),
    ("resolved_by", "TEXT"),
    ("resolution_reason", "TEXT"),
    ("superseded_by_reoptimization_request_id", "TEXT"),
    ("superseded_by_decision_id", "TEXT"),
)

_PORTFOLIO_REOPTIMIZATION_MIGRATION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("origin_run_id", "TEXT"),
    ("last_resumed_run_id", "TEXT"),
    ("resume_count", "INTEGER NOT NULL DEFAULT 0"),
    ("last_resumed_at", "TEXT"),
    ("failure_reason", "TEXT"),
    ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
    ("last_attempt_run_id", "TEXT"),
    ("last_attempt_exchange_time", "TEXT"),
    ("attempt_owner_run_id", "TEXT"),
    ("attempt_started_at", "TEXT"),
    ("attempt_lease_expires_at", "TEXT"),
    ("attempt_heartbeat_at", "TEXT"),
    ("attempt_generation", "INTEGER NOT NULL DEFAULT 0"),
    ("attempt_token", "TEXT"),
    ("resolution_status", "TEXT NOT NULL DEFAULT 'UNRESOLVED'"),
    ("resolved_at", "TEXT"),
    ("resolved_by", "TEXT"),
    ("resolution_reason", "TEXT"),
)

_UNRESOLVED_COMPONENT_STATUSES = (
    "AUTHORIZED",
    "READY",
    "SUBMITTED",
    "WORKING",
    "PARTIALLY_FILLED",
    "REOPTIMIZATION_REQUIRED",
)

DEFAULT_REOPTIMIZATION_LEASE_SECONDS = 300.0


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
        reoptimization_columns = {
            str(row[1])
            for row in db.execute(
                "PRAGMA table_info(portfolio_reoptimization_requests)"
            )
        }
        for name, declaration in _PORTFOLIO_REOPTIMIZATION_MIGRATION_COLUMNS:
            if name not in reoptimization_columns:
                db.execute(
                    f"ALTER TABLE portfolio_reoptimization_requests "
                    f"ADD COLUMN {name} {declaration}"
                )
        db.execute(
            """UPDATE portfolio_execution_components
            SET origin_run_id = run_id
            WHERE origin_run_id IS NULL OR origin_run_id = ''"""
        )
        db.execute(
            """UPDATE portfolio_reoptimization_requests
            SET origin_run_id = run_id
            WHERE origin_run_id IS NULL OR origin_run_id = ''"""
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
                OR origin_run_id IS NULL OR origin_run_id = ''
                OR broker_account_id IS NULL OR broker_account_id = ''
                OR broker_account_id IN ('webull_paper', 'webull_live')
                OR trading_date IS NULL OR trading_date = ''
            )
              AND (
                status != 'REOPTIMIZATION_REQUIRED'
                OR failure_reoptimization_reason IS NULL
              )
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        db.execute(
            """
            UPDATE portfolio_reoptimization_requests
            SET status = 'FAILED',
                failure_reason = COALESCE(
                    failure_reason,
                    'legacy_unscoped_reoptimization_request'
                ),
                state_version = state_version + 1,
                updated_at = ?
            WHERE (
                session_id IS NULL OR session_id = ''
                OR origin_run_id IS NULL OR origin_run_id = ''
                OR broker_account_id IS NULL OR broker_account_id = ''
                OR broker_account_id IN ('webull_paper', 'webull_live')
                OR trading_date IS NULL OR trading_date = ''
            )
              AND status IN ('PENDING', 'RUNNING')
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        # Preserve duplicate legacy rows for reconciliation, but remove their
        # executable authority before enforcing uniqueness for all clean rows.
        duplicate_keys = list(
            db.execute(
                """SELECT session_id, broker_account_id, trading_date,
                          target_portfolio_decision_id, component_index
                   FROM portfolio_execution_components
                   WHERE session_id IS NOT NULL AND session_id != ''
                     AND broker_account_id IS NOT NULL AND broker_account_id != ''
                     AND trading_date IS NOT NULL AND trading_date != ''
                   GROUP BY session_id, broker_account_id, trading_date,
                            target_portfolio_decision_id, component_index
                   HAVING COUNT(*) > 1"""
            )
        )
        migration_now = datetime.now(timezone.utc).isoformat()
        for key in duplicate_keys:
            db.execute(
                """UPDATE portfolio_execution_components
                   SET status = 'REOPTIMIZATION_REQUIRED',
                       failure_reoptimization_reason =
                           'duplicate_component_index_migration',
                       state_version = state_version + 1,
                       updated_at = ?
                   WHERE session_id = ? AND broker_account_id = ?
                     AND trading_date = ? AND target_portfolio_decision_id = ?
                     AND component_index = ?
                     AND component_order_conflicted = 0
                     AND status IN (
                         'AUTHORIZED', 'READY', 'SUBMITTED', 'WORKING',
                         'PARTIALLY_FILLED', 'REOPTIMIZATION_REQUIRED'
                     )""",
                (migration_now, *key),
            )
            db.execute(
                """UPDATE portfolio_execution_components
                   SET component_order_conflicted = 1
                   WHERE session_id = ? AND broker_account_id = ?
                     AND trading_date = ? AND target_portfolio_decision_id = ?
                     AND component_index = ?""",
                key,
            )
        db.executescript(
            """
            DROP INDEX IF EXISTS idx_portfolio_component_order;
            CREATE UNIQUE INDEX idx_portfolio_component_order
                ON portfolio_execution_components (
                    session_id, broker_account_id, trading_date,
                    target_portfolio_decision_id, component_index
                ) WHERE component_order_conflicted = 0;
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


class PortfolioComponentResolutionStatus(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    SUPERSEDED = "SUPERSEDED"
    OPERATOR_RESOLVED = "OPERATOR_RESOLVED"


class PortfolioTransitionConflict(RuntimeError):
    """A stale component writer lost the compare-and-swap race."""


class PortfolioAttemptLeaseActive(PortfolioTransitionConflict):
    """A different process owns an unexpired reoptimization attempt."""


@dataclass(frozen=True)
class PortfolioExecutionOwner:
    session_id: str
    broker_account_identity: str
    trading_date: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.session_id,
                self.broker_account_identity,
                self.trading_date,
            )
        ):
            raise ValueError("portfolio execution ownership is incomplete")
        if self.broker_account_identity.strip().lower() in {
            "webull",
            "webull_paper",
            "webull_live",
        }:
            raise ValueError("broker provider kind is not an account identity")
        date.fromisoformat(self.trading_date)

    def matches(self, record: PortfolioExecutionComponentRecord) -> bool:
        return (
            record.session_id == self.session_id
            and record.broker_account_identity == self.broker_account_identity
            and record.trading_date == self.trading_date
        )

    @property
    def broker_account_id(self) -> str:
        """Compatibility alias; the persisted value is an account identity."""
        return self.broker_account_identity


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


async def _resolve_component_obligations(
    db: aiosqlite.Connection,
    *,
    owner: PortfolioExecutionOwner,
    tuple_ids: tuple[str, ...],
    resolution_status: PortfolioComponentResolutionStatus,
    resolved_at: str,
    resolved_by: str,
    resolution_reason: str,
    superseded_by_reoptimization_request_id: str | None = None,
    superseded_by_decision_id: str | None = None,
) -> None:
    if not tuple_ids:
        return
    if resolution_status == PortfolioComponentResolutionStatus.SUPERSEDED and (
        not superseded_by_reoptimization_request_id or not superseded_by_decision_id
    ):
        raise ValueError("superseded components require replacement provenance")
    placeholders = ", ".join("?" for _ in tuple_ids)
    params: list[Any] = [
        resolution_status.value,
        resolved_at,
        resolved_by,
        resolution_reason,
        superseded_by_reoptimization_request_id,
        superseded_by_decision_id,
        resolved_at,
        owner.session_id,
        owner.broker_account_identity,
        owner.trading_date,
        *tuple_ids,
    ]
    await db.execute(
        f"""
        UPDATE portfolio_execution_components
        SET resolution_status = ?,
            resolved_at = ?,
            resolved_by = ?,
            resolution_reason = ?,
            superseded_by_reoptimization_request_id = COALESCE(
                ?, superseded_by_reoptimization_request_id
            ),
            superseded_by_decision_id = COALESCE(?, superseded_by_decision_id),
            state_version = state_version + 1,
            updated_at = ?
        WHERE session_id = ?
          AND broker_account_id = ?
          AND trading_date = ?
          AND authorized_position_tuple_id IN ({placeholders})
          AND status = 'REOPTIMIZATION_REQUIRED'
          AND COALESCE(resolution_status, 'UNRESOLVED') = 'UNRESOLVED'
        """,
        params,
    )


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
    origin_run_id: str | None
    broker_account_identity: str | None
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
    last_resumed_run_id: str | None = None
    resume_count: int = 0
    last_resumed_at: str | None = None
    post_fill_objective_version: int | None = None
    post_fill_objective_fingerprint: str | None = None
    post_fill_snapshot_id: str | None = None
    post_fill_exchange_time: str | None = None
    reconciled_filled_quantity: int | None = None
    continuation_ready: bool = False
    state_version: int = 0
    component_order_conflicted: bool = False
    resolution_status: PortfolioComponentResolutionStatus = (
        PortfolioComponentResolutionStatus.UNRESOLVED
    )
    resolved_at: str | None = None
    resolved_by: str | None = None
    resolution_reason: str | None = None
    superseded_by_reoptimization_request_id: str | None = None
    superseded_by_decision_id: str | None = None
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
        if self.resume_count < 0:
            raise ValueError("resume_count must be non-negative")
        if self.resolved_at and datetime.fromisoformat(self.resolved_at).tzinfo is None:
            raise ValueError("resolved_at must be timezone-aware")
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
        if self.resolution_status != PortfolioComponentResolutionStatus.UNRESOLVED:
            if not all((self.resolved_at, self.resolved_by, self.resolution_reason)):
                raise ValueError("resolved component requires full resolution provenance")
        if self.resolution_status == PortfolioComponentResolutionStatus.SUPERSEDED and (
            not self.superseded_by_reoptimization_request_id
            or not self.superseded_by_decision_id
        ):
            raise ValueError("superseded component requires replacement provenance")

    @property
    def has_scoped_owner(self) -> bool:
        return all((self.session_id, self.broker_account_identity, self.trading_date))

    @property
    def run_id(self) -> str | None:
        """Backward-compatible alias for immutable origin process provenance."""
        return self.origin_run_id

    @property
    def broker_account_id(self) -> str | None:
        """Compatibility alias for the legacy SQLite column name."""
        return self.broker_account_identity

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
            try:
                await db.execute(
                    """
                INSERT INTO portfolio_execution_components (
                    session_id, run_id, origin_run_id, last_resumed_run_id,
                    resume_count, last_resumed_at,
                    broker_account_id, trading_date,
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
                    continuation_ready, state_version, component_order_conflicted,
                    resolution_status, resolved_at, resolved_by, resolution_reason,
                    superseded_by_reoptimization_request_id, superseded_by_decision_id,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.origin_run_id,
                    record.origin_run_id,
                    record.last_resumed_run_id,
                    record.resume_count,
                    record.last_resumed_at,
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
                    int(record.component_order_conflicted),
                    record.resolution_status.value,
                    record.resolved_at,
                    record.resolved_by,
                    record.resolution_reason,
                    record.superseded_by_reoptimization_request_id,
                    record.superseded_by_decision_id,
                    now,
                    record.updated_at or now,
                    json.dumps(record.extra or {}, sort_keys=True),
                    ),
                )
            except aiosqlite.IntegrityError as exc:
                existing = await self.get(record.authorized_position_tuple_id)
                if existing is None:
                    raise ValueError(
                        "duplicate portfolio component index or client-order identity"
                    ) from exc
            await db.commit()
        stored = await self.get(record.authorized_position_tuple_id)
        if stored is None:
            raise ValueError("portfolio component authority was not durably inserted")
        immutable = (
            "session_id",
            "origin_run_id",
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
            "component_order_conflicted",
        )
        if any(getattr(stored, name) != getattr(record, name) for name in immutable):
            raise ValueError("authorized portfolio component conflicts with durable authority")
        return stored

    async def record_resume(
        self,
        authorized_position_tuple_id: str,
        *,
        owner: PortfolioExecutionOwner,
        current_run_id: str,
        resumed_at: str,
    ) -> PortfolioExecutionComponentRecord:
        """Record process provenance without changing the stable execution owner."""
        existing = await self.get(authorized_position_tuple_id)
        if existing is None:
            raise KeyError(f"portfolio component not found: {authorized_position_tuple_id}")
        if not owner.matches(existing) or not existing.has_scoped_owner:
            raise PermissionError("portfolio component owner does not match runtime")
        parsed = datetime.fromisoformat(resumed_at)
        if parsed.tzinfo is None:
            raise ValueError("resumed_at must be timezone-aware")
        if not current_run_id:
            raise ValueError("current_run_id is required")
        if existing.last_resumed_run_id == current_run_id:
            return existing
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE portfolio_execution_components
                SET last_resumed_run_id = ?, resume_count = resume_count + 1,
                    last_resumed_at = ?, state_version = state_version + 1,
                    updated_at = ?
                WHERE authorized_position_tuple_id = ?
                  AND state_version = ? AND status = ?
                """,
                (
                    current_run_id,
                    resumed_at,
                    resumed_at,
                    authorized_position_tuple_id,
                    existing.state_version,
                    existing.status.value,
                ),
            )
            await db.commit()
        if cur.rowcount != 1:
            raise PortfolioTransitionConflict(
                "portfolio component resume lost compare-and-swap race"
            )
        stored = await self.get(authorized_position_tuple_id)
        assert stored is not None
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
        resolution_status: PortfolioComponentResolutionStatus | None = None,
        resolved_at: str | None = None,
        resolved_by: str | None = None,
        resolution_reason: str | None = None,
        superseded_by_reoptimization_request_id: str | None = None,
        superseded_by_decision_id: str | None = None,
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
        resolved_status = (
            existing.resolution_status
            if resolution_status is None
            else PortfolioComponentResolutionStatus(resolution_status)
        )
        resolved_at_value = resolved_at if resolved_at is not None else existing.resolved_at
        resolved_by_value = resolved_by if resolved_by is not None else existing.resolved_by
        resolved_reason_value = (
            resolution_reason
            if resolution_reason is not None
            else existing.resolution_reason
        )
        superseded_request_id = (
            superseded_by_reoptimization_request_id
            if superseded_by_reoptimization_request_id is not None
            else existing.superseded_by_reoptimization_request_id
        )
        superseded_decision_id = (
            superseded_by_decision_id
            if superseded_by_decision_id is not None
            else existing.superseded_by_decision_id
        )
        if continuation and (
            status != PortfolioComponentStatus.FILLED
            or any(value is None or value == "" for value in continuation_fields)
            or int(continuation_fields[4]) != existing.authorized_quantity
        ):
            raise ValueError("post-fill continuation checkpoint is incomplete")
        if resolved_status != PortfolioComponentResolutionStatus.UNRESOLVED:
            if not all((resolved_at_value, resolved_by_value, resolved_reason_value)):
                raise ValueError("component resolution requires full provenance")
            if (
                resolved_status == PortfolioComponentResolutionStatus.SUPERSEDED
                and (not superseded_request_id or not superseded_decision_id)
            ):
                raise ValueError("superseded component requires replacement provenance")
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
            resolved_status,
            resolved_at_value,
            resolved_by_value,
            resolved_reason_value,
            superseded_request_id,
            superseded_decision_id,
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
            existing.resolution_status,
            existing.resolved_at,
            existing.resolved_by,
            existing.resolution_reason,
            existing.superseded_by_reoptimization_request_id,
            existing.superseded_by_decision_id,
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
                    resolution_status = COALESCE(?, resolution_status),
                    resolved_at = COALESCE(?, resolved_at),
                    resolved_by = COALESCE(?, resolved_by),
                    resolution_reason = COALESCE(?, resolution_reason),
                    superseded_by_reoptimization_request_id = COALESCE(
                        ?, superseded_by_reoptimization_request_id
                    ),
                    superseded_by_decision_id = COALESCE(?, superseded_by_decision_id),
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
                    resolved_status.value if resolution_status is not None else None,
                    resolved_at,
                    resolved_by,
                    resolution_reason,
                    superseded_by_reoptimization_request_id,
                    superseded_by_decision_id,
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

    async def resolve_stale_components(
        self,
        tuple_ids: tuple[str, ...],
        *,
        owner: PortfolioExecutionOwner,
        resolution_status: PortfolioComponentResolutionStatus,
        resolved_at: str,
        resolved_by: str,
        resolution_reason: str,
        superseded_by_reoptimization_request_id: str | None = None,
        superseded_by_decision_id: str | None = None,
    ) -> list[PortfolioExecutionComponentRecord]:
        await self._ensure()
        if not tuple_ids:
            return []
        parsed = datetime.fromisoformat(resolved_at)
        if parsed.tzinfo is None:
            raise ValueError("resolved_at must be timezone-aware")
        if resolution_status == PortfolioComponentResolutionStatus.SUPERSEDED and (
            not superseded_by_reoptimization_request_id or not superseded_by_decision_id
        ):
            raise ValueError("superseded components require replacement provenance")
        async with aiosqlite.connect(self._db_path) as db:
            await _resolve_component_obligations(
                db,
                owner=owner,
                tuple_ids=tuple_ids,
                resolution_status=resolution_status,
                resolved_at=resolved_at,
                resolved_by=resolved_by,
                resolution_reason=resolution_reason,
                superseded_by_reoptimization_request_id=superseded_by_reoptimization_request_id,
                superseded_by_decision_id=superseded_by_decision_id,
            )
            await db.commit()
        return [
            await self.get(tuple_id)
            for tuple_id in tuple_ids
            if await self.get(tuple_id) is not None
        ]

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
                  AND session_id = ?
                  AND broker_account_id = ? AND trading_date = ?""",
                (
                    client_order_id,
                    owner.session_id,
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
                      AND session_id = ?
                      AND broker_account_id = ? AND trading_date = ?
                    ORDER BY component_index ASC""",
                    (
                        target_portfolio_decision_id,
                        owner.session_id,
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
        broker_account_identity: str,
        trading_date: str,
    ) -> list[PortfolioExecutionComponentRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM portfolio_execution_components
                WHERE session_id = ?
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
                (session_id, broker_account_identity, trading_date),
            )
            rows = await cur.fetchall()
        return [self._row(row) for row in rows]

    async def has_unresolved(
        self,
        *,
        session_id: str,
        broker_account_identity: str,
        trading_date: str,
    ) -> bool:
        """Return whether this stable owner retains executable or stale authority."""
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            row = await (
                await db.execute(
                    f"""SELECT 1 FROM portfolio_execution_components
                        WHERE session_id = ? AND broker_account_id = ?
                          AND trading_date = ?
                          AND (
                            status IN ('AUTHORIZED', 'READY', 'SUBMITTED', 'WORKING', 'PARTIALLY_FILLED')
                            OR (
                                status = 'REOPTIMIZATION_REQUIRED'
                                AND COALESCE(resolution_status, 'UNRESOLVED') = 'UNRESOLVED'
                            )
                          )
                        LIMIT 1""",
                    (
                        session_id,
                        broker_account_identity,
                        trading_date,
                    ),
                )
            ).fetchone()
        return row is not None

    @staticmethod
    def _row(row: aiosqlite.Row) -> PortfolioExecutionComponentRecord:
        return PortfolioExecutionComponentRecord(
            session_id=row["session_id"],
            origin_run_id=row["origin_run_id"] or row["run_id"],
            broker_account_identity=row["broker_account_id"],
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
            last_resumed_run_id=row["last_resumed_run_id"],
            resume_count=int(row["resume_count"] or 0),
            last_resumed_at=row["last_resumed_at"],
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
            component_order_conflicted=bool(row["component_order_conflicted"]),
            resolution_status=PortfolioComponentResolutionStatus(
                row["resolution_status"] or "UNRESOLVED"
            ),
            resolved_at=row["resolved_at"],
            resolved_by=row["resolved_by"],
            resolution_reason=row["resolution_reason"],
            superseded_by_reoptimization_request_id=(
                row["superseded_by_reoptimization_request_id"]
            ),
            superseded_by_decision_id=row["superseded_by_decision_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            extra=json.loads(row["payload_json"] or "{}"),
        )


class PortfolioReoptimizationStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PortfolioReoptimizationResolutionStatus(StrEnum):
    UNRESOLVED = "UNRESOLVED"
    RETRY_REQUESTED = "RETRY_REQUESTED"
    RESOLVED = "RESOLVED"


@dataclass(frozen=True)
class PortfolioReoptimizationRequestRecord:
    request_id: str
    session_id: str
    origin_run_id: str
    broker_account_identity: str
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
    last_resumed_run_id: str | None = None
    resume_count: int = 0
    last_resumed_at: str | None = None
    failure_reason: str | None = None
    attempt_count: int = 0
    last_attempt_run_id: str | None = None
    last_attempt_exchange_time: str | None = None
    attempt_owner_run_id: str | None = None
    attempt_started_at: str | None = None
    attempt_lease_expires_at: str | None = None
    attempt_heartbeat_at: str | None = None
    attempt_generation: int = 0
    attempt_token: str | None = None
    resolution_status: PortfolioReoptimizationResolutionStatus = (
        PortfolioReoptimizationResolutionStatus.UNRESOLVED
    )
    resolved_at: str | None = None
    resolved_by: str | None = None
    resolution_reason: str | None = None
    state_version: int = 0
    created_at: str | None = None
    updated_at: str | None = None
    extra: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        PortfolioExecutionOwner(
            session_id=self.session_id,
            broker_account_identity=self.broker_account_identity,
            trading_date=self.trading_date,
        )
        created = datetime.fromisoformat(self.created_exchange_time)
        if created.tzinfo is None:
            raise ValueError("created_exchange_time must be timezone-aware")
        if not self.original_portfolio_decision_id or not self.latest_snapshot_id:
            raise ValueError("reoptimization provenance is incomplete")
        if (
            self.latest_objective_version < 0
            or self.state_version < 0
            or self.resume_count < 0
            or self.attempt_count < 0
            or self.attempt_generation < 0
        ):
            raise ValueError("reoptimization versions must be non-negative")
        for value in (
            self.attempt_started_at,
            self.attempt_lease_expires_at,
            self.attempt_heartbeat_at,
            self.resolved_at,
        ):
            if value and datetime.fromisoformat(value).tzinfo is None:
                raise ValueError("reoptimization timestamps must be timezone-aware")

    @property
    def run_id(self) -> str:
        """Backward-compatible alias for immutable origin process provenance."""
        return self.origin_run_id

    @property
    def owner(self) -> PortfolioExecutionOwner:
        return PortfolioExecutionOwner(
            session_id=self.session_id,
            broker_account_identity=self.broker_account_identity,
            trading_date=self.trading_date,
        )

    @property
    def broker_account_id(self) -> str:
        """Compatibility alias for the legacy SQLite column name."""
        return self.broker_account_identity


def stable_reoptimization_request_id(
    *,
    session_id: str,
    broker_account_identity: str,
    trading_date: str,
    original_portfolio_decision_id: str,
    remaining_authorized_tuple_ids: tuple[str, ...],
) -> str:
    return uuid5(
        NAMESPACE_URL,
        "joker:portfolio-reoptimization:"
        f"{session_id}:{broker_account_identity}:{trading_date}:"
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
            db.row_factory = aiosqlite.Row
            existing_row = await (
                await db.execute(
                    """
                    SELECT * FROM portfolio_reoptimization_requests
                    WHERE session_id = ? AND broker_account_id = ? AND trading_date = ?
                      AND original_portfolio_decision_id = ?
                      AND remaining_authorized_tuple_ids_json = ?
                    ORDER BY created_at LIMIT 1
                    """,
                    (
                        record.session_id,
                        record.broker_account_id,
                        record.trading_date,
                        record.original_portfolio_decision_id,
                        json.dumps(record.remaining_authorized_tuple_ids),
                    ),
                )
            ).fetchone()
            if existing_row is not None:
                return self._row(existing_row)
            await db.execute(
                """
                INSERT OR IGNORE INTO portfolio_reoptimization_requests (
                    request_id, session_id, run_id, origin_run_id,
                    last_resumed_run_id, resume_count, last_resumed_at,
                    broker_account_id, trading_date,
                    original_portfolio_decision_id,
                    already_filled_tuple_ids_json, open_positions_json,
                    remaining_authorized_tuple_ids_json, reason_codes_json,
                    latest_objective_state_json, latest_objective_version,
                    latest_snapshot_id, created_exchange_time, status,
                    replacement_decision_id, replacement_action, failure_reason,
                    attempt_count, last_attempt_run_id, last_attempt_exchange_time,
                    attempt_owner_run_id, attempt_started_at,
                    attempt_lease_expires_at, attempt_heartbeat_at,
                    attempt_generation, attempt_token,
                    resolution_status, resolved_at, resolved_by, resolution_reason,
                    state_version,
                    created_at, updated_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.request_id,
                    record.session_id,
                    record.origin_run_id,
                    record.origin_run_id,
                    record.last_resumed_run_id,
                    record.resume_count,
                    record.last_resumed_at,
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
                    record.failure_reason,
                    record.attempt_count,
                    record.last_attempt_run_id,
                    record.last_attempt_exchange_time,
                    record.attempt_owner_run_id,
                    record.attempt_started_at,
                    record.attempt_lease_expires_at,
                    record.attempt_heartbeat_at,
                    record.attempt_generation,
                    record.attempt_token,
                    record.resolution_status.value,
                    record.resolved_at,
                    record.resolved_by,
                    record.resolution_reason,
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
            or stored.origin_run_id != record.origin_run_id
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
        broker_account_identity: str,
        trading_date: str,
    ) -> list[PortfolioReoptimizationRequestRecord]:
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            rows = await (
                await db.execute(
                    """
                    SELECT * FROM portfolio_reoptimization_requests
                    WHERE session_id = ?
                      AND broker_account_id = ? AND trading_date = ?
                      AND status IN ('PENDING', 'RUNNING')
                    ORDER BY created_at, request_id
                    """,
                    (session_id, broker_account_identity, trading_date),
                )
            ).fetchall()
        return [self._row(row) for row in rows]

    async def has_unresolved(
        self,
        *,
        session_id: str,
        broker_account_identity: str,
        trading_date: str,
    ) -> bool:
        """FAILED remains owner-blocking until an explicit resolution is durable."""
        await self._ensure()
        async with aiosqlite.connect(self._db_path) as db:
            row = await (
                await db.execute(
                    """SELECT 1 FROM portfolio_reoptimization_requests
                       WHERE session_id = ? AND broker_account_id = ?
                         AND trading_date = ?
                         AND (
                           status IN ('PENDING', 'RUNNING')
                           OR (
                             status = 'FAILED'
                             AND COALESCE(resolution_status, 'UNRESOLVED') != 'RESOLVED'
                           )
                         )
                       LIMIT 1""",
                    (session_id, broker_account_identity, trading_date),
                )
            ).fetchone()
        return row is not None

    async def begin_attempt(
        self,
        request_id: str,
        *,
        owner: PortfolioExecutionOwner,
        current_run_id: str,
        attempt_exchange_time: str,
        lease_seconds: float = DEFAULT_REOPTIMIZATION_LEASE_SECONDS,
    ) -> PortfolioReoptimizationRequestRecord:
        """Claim an owned request and durably record this process attempt."""
        existing = await self.get(request_id)
        if existing is None:
            raise KeyError(f"reoptimization request not found: {request_id}")
        if existing.owner != owner:
            raise PermissionError("reoptimization request owner does not match runtime")
        if existing.status not in {
            PortfolioReoptimizationStatus.PENDING,
            PortfolioReoptimizationStatus.RUNNING,
        }:
            raise ValueError(f"reoptimization request is terminal: {existing.status}")
        attempted = datetime.fromisoformat(attempt_exchange_time)
        if attempted.tzinfo is None:
            raise ValueError("attempt_exchange_time must be timezone-aware")
        if not current_run_id:
            raise ValueError("current_run_id is required")
        if lease_seconds <= 0:
            raise ValueError("reoptimization attempt lease must be positive")
        if existing.status == PortfolioReoptimizationStatus.RUNNING:
            lease_expiry = (
                datetime.fromisoformat(existing.attempt_lease_expires_at)
                if existing.attempt_lease_expires_at
                else None
            )
            if (
                existing.attempt_owner_run_id == current_run_id
                and lease_expiry is not None
                and attempted < lease_expiry
            ):
                return existing
            if lease_expiry is not None and attempted < lease_expiry:
                raise PortfolioAttemptLeaseActive(
                    "reoptimization attempt lease is owned by another run"
                )
        lease_expires_at = (attempted + timedelta(seconds=lease_seconds)).isoformat()
        attempt_generation = existing.attempt_generation + 1
        attempt_token = uuid4().hex
        resumed = existing.last_resumed_run_id != current_run_id
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE portfolio_reoptimization_requests
                SET status = 'RUNNING', attempt_count = attempt_count + 1,
                    last_attempt_run_id = ?, last_attempt_exchange_time = ?,
                    attempt_owner_run_id = ?, attempt_started_at = ?,
                    attempt_lease_expires_at = ?, attempt_heartbeat_at = ?,
                    attempt_generation = ?, attempt_token = ?,
                    resolution_status = 'UNRESOLVED',
                    last_resumed_run_id = CASE WHEN ? THEN ? ELSE last_resumed_run_id END,
                    resume_count = resume_count + CASE WHEN ? THEN 1 ELSE 0 END,
                    last_resumed_at = CASE WHEN ? THEN ? ELSE last_resumed_at END,
                    state_version = state_version + 1, updated_at = ?
                WHERE request_id = ? AND state_version = ? AND status = ?
                """,
                (
                    current_run_id,
                    attempt_exchange_time,
                    current_run_id,
                    attempt_exchange_time,
                    lease_expires_at,
                    attempt_exchange_time,
                    attempt_generation,
                    attempt_token,
                    int(resumed),
                    current_run_id,
                    int(resumed),
                    int(resumed),
                    attempt_exchange_time,
                    attempt_exchange_time,
                    request_id,
                    existing.state_version,
                    existing.status.value,
                ),
            )
            await db.commit()
        if cur.rowcount != 1:
            raise PortfolioTransitionConflict(
                "reoptimization attempt lost compare-and-swap race"
            )
        stored = await self.get(request_id)
        assert stored is not None
        return stored

    async def resolve_failed(
        self,
        request_id: str,
        *,
        resolved_at: str,
        resolved_by: str,
        resolution_reason: str,
    ) -> PortfolioReoptimizationRequestRecord:
        """Explicitly release the stable owner after operator reconciliation."""
        existing = await self.get(request_id)
        if existing is None:
            raise KeyError(f"reoptimization request not found: {request_id}")
        if existing.status != PortfolioReoptimizationStatus.FAILED:
            raise ValueError("only a failed reoptimization request can be resolved")
        parsed = datetime.fromisoformat(resolved_at)
        if parsed.tzinfo is None or not resolved_by or not resolution_reason:
            raise ValueError("failed-request resolution provenance is incomplete")
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """UPDATE portfolio_reoptimization_requests
                   SET resolution_status = 'RESOLVED', resolved_at = ?, resolved_by = ?,
                       resolution_reason = ?, state_version = state_version + 1,
                       updated_at = ?
                   WHERE request_id = ? AND state_version = ? AND status = 'FAILED'
                     AND COALESCE(resolution_status, 'UNRESOLVED') != 'RESOLVED'""",
                (
                    resolved_at,
                    resolved_by,
                    resolution_reason,
                    resolved_at,
                    request_id,
                    existing.state_version,
                ),
            )
            await _resolve_component_obligations(
                db,
                owner=existing.owner,
                tuple_ids=existing.remaining_authorized_tuple_ids,
                resolution_status=PortfolioComponentResolutionStatus.OPERATOR_RESOLVED,
                resolved_at=resolved_at,
                resolved_by=resolved_by,
                resolution_reason=resolution_reason,
            )
            await db.commit()
        if cur.rowcount != 1:
            latest = await self.get(request_id)
            if latest is not None and (
                latest.resolution_status
                == PortfolioReoptimizationResolutionStatus.RESOLVED
            ):
                return latest
            raise PortfolioTransitionConflict("failed-request resolution lost CAS race")
        stored = await self.get(request_id)
        assert stored is not None
        return stored

    async def retry_failed(
        self,
        request_id: str,
        *,
        requested_at: str,
        requested_by: str,
        resolution_reason: str,
    ) -> PortfolioReoptimizationRequestRecord:
        """Explicitly return failed work to PENDING; it remains owner-blocking."""
        existing = await self.get(request_id)
        if existing is None:
            raise KeyError(f"reoptimization request not found: {request_id}")
        if existing.status != PortfolioReoptimizationStatus.FAILED:
            raise ValueError("only a failed reoptimization request can be retried")
        parsed = datetime.fromisoformat(requested_at)
        if parsed.tzinfo is None or not requested_by or not resolution_reason:
            raise ValueError("failed-request retry provenance is incomplete")
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """UPDATE portfolio_reoptimization_requests
                   SET status = 'PENDING', resolution_status = 'RETRY_REQUESTED',
                       resolved_at = ?, resolved_by = ?, resolution_reason = ?,
                       attempt_owner_run_id = NULL, attempt_started_at = NULL,
                       attempt_lease_expires_at = NULL, attempt_heartbeat_at = NULL,
                       state_version = state_version + 1, updated_at = ?
                   WHERE request_id = ? AND state_version = ? AND status = 'FAILED'""",
                (
                    requested_at,
                    requested_by,
                    resolution_reason,
                    requested_at,
                    request_id,
                    existing.state_version,
                ),
            )
            await db.commit()
        if cur.rowcount != 1:
            raise PortfolioTransitionConflict("failed-request retry lost CAS race")
        stored = await self.get(request_id)
        assert stored is not None
        return stored

    async def transition(
        self,
        request_id: str,
        *,
        status: PortfolioReoptimizationStatus,
        replacement_decision_id: str | None = None,
        replacement_action: str | None = None,
        failure_reason: str | None = None,
        expected_state_version: int | None = None,
    ) -> PortfolioReoptimizationRequestRecord:
        existing = await self.get(request_id)
        if existing is None:
            raise KeyError(f"reoptimization request not found: {request_id}")
        if status in {
            PortfolioReoptimizationStatus.COMPLETED,
            PortfolioReoptimizationStatus.FAILED,
        }:
            raise ValueError("terminal attempt updates must use complete_attempt/fail_attempt")
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
                    failure_reason = COALESCE(?, failure_reason),
                    resolution_status = CASE
                        WHEN ? = 'FAILED' THEN 'UNRESOLVED'
                        ELSE resolution_status
                    END,
                    attempt_lease_expires_at = CASE
                        WHEN ? IN ('COMPLETED', 'FAILED') THEN NULL
                        ELSE attempt_lease_expires_at
                    END,
                    state_version = state_version + 1, updated_at = ?
                WHERE request_id = ? AND state_version = ? AND status = ?
                """,
                (
                    status.value,
                    replacement_decision_id,
                    replacement_action,
                    failure_reason,
                    status.value,
                    status.value,
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

    async def complete_attempt(
        self,
        *,
        attempt: PortfolioReoptimizationRequestRecord,
        completed_at: str,
        replacement_decision_id: str,
        replacement_action: str,
    ) -> PortfolioReoptimizationRequestRecord:
        if attempt.status != PortfolioReoptimizationStatus.RUNNING:
            raise ValueError("only a running reoptimization attempt can complete")
        parsed = datetime.fromisoformat(completed_at)
        if parsed.tzinfo is None:
            raise ValueError("completed_at must be timezone-aware")
        if not replacement_decision_id or replacement_action not in {"WAIT", "ENTER"}:
            raise ValueError("replacement completion provenance is incomplete")
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE portfolio_reoptimization_requests
                SET status = 'COMPLETED',
                    replacement_decision_id = ?,
                    replacement_action = ?,
                    last_attempt_run_id = COALESCE(last_attempt_run_id, ?),
                    last_attempt_exchange_time = ?,
                    attempt_lease_expires_at = NULL,
                    attempt_heartbeat_at = ?,
                    state_version = state_version + 1,
                    updated_at = ?
                WHERE request_id = ?
                  AND status = 'RUNNING'
                  AND attempt_owner_run_id = ?
                  AND attempt_generation = ?
                  AND attempt_token = ?
                  AND state_version = ?
                """,
                (
                    replacement_decision_id,
                    replacement_action,
                    attempt.attempt_owner_run_id,
                    completed_at,
                    completed_at,
                    completed_at,
                    attempt.request_id,
                    attempt.attempt_owner_run_id,
                    attempt.attempt_generation,
                    attempt.attempt_token,
                    attempt.state_version,
                ),
            )
            if cur.rowcount != 1:
                await db.rollback()
                raise PortfolioTransitionConflict(
                    "reoptimization completion lost attempt fencing race"
                )
            await _resolve_component_obligations(
                db,
                owner=attempt.owner,
                tuple_ids=attempt.remaining_authorized_tuple_ids,
                resolution_status=PortfolioComponentResolutionStatus.SUPERSEDED,
                resolved_at=completed_at,
                resolved_by=str(attempt.attempt_owner_run_id or ""),
                resolution_reason="reoptimization_completed",
                superseded_by_reoptimization_request_id=attempt.request_id,
                superseded_by_decision_id=replacement_decision_id,
            )
            await db.commit()
        stored = await self.get(attempt.request_id)
        assert stored is not None
        return stored

    async def fail_attempt(
        self,
        *,
        attempt: PortfolioReoptimizationRequestRecord,
        failed_at: str,
        failure_reason: str,
    ) -> PortfolioReoptimizationRequestRecord:
        if attempt.status != PortfolioReoptimizationStatus.RUNNING:
            raise ValueError("only a running reoptimization attempt can fail")
        parsed = datetime.fromisoformat(failed_at)
        if parsed.tzinfo is None:
            raise ValueError("failed_at must be timezone-aware")
        if not failure_reason:
            raise ValueError("failure_reason is required")
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute(
                """
                UPDATE portfolio_reoptimization_requests
                SET status = 'FAILED',
                    failure_reason = ?,
                    resolution_status = 'UNRESOLVED',
                    last_attempt_run_id = COALESCE(last_attempt_run_id, ?),
                    last_attempt_exchange_time = ?,
                    attempt_lease_expires_at = NULL,
                    attempt_heartbeat_at = ?,
                    state_version = state_version + 1,
                    updated_at = ?
                WHERE request_id = ?
                  AND status = 'RUNNING'
                  AND attempt_owner_run_id = ?
                  AND attempt_generation = ?
                  AND attempt_token = ?
                  AND state_version = ?
                """,
                (
                    failure_reason,
                    attempt.attempt_owner_run_id,
                    failed_at,
                    failed_at,
                    failed_at,
                    attempt.request_id,
                    attempt.attempt_owner_run_id,
                    attempt.attempt_generation,
                    attempt.attempt_token,
                    attempt.state_version,
                ),
            )
            await db.commit()
        if cur.rowcount != 1:
            raise PortfolioTransitionConflict(
                "reoptimization failure lost attempt fencing race"
            )
        stored = await self.get(attempt.request_id)
        assert stored is not None
        return stored

    @staticmethod
    def _row(row: aiosqlite.Row) -> PortfolioReoptimizationRequestRecord:
        return PortfolioReoptimizationRequestRecord(
            request_id=row["request_id"],
            session_id=row["session_id"],
            origin_run_id=row["origin_run_id"] or row["run_id"],
            broker_account_identity=row["broker_account_id"],
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
            last_resumed_run_id=row["last_resumed_run_id"],
            resume_count=int(row["resume_count"] or 0),
            last_resumed_at=row["last_resumed_at"],
            failure_reason=row["failure_reason"],
            attempt_count=int(row["attempt_count"] or 0),
            last_attempt_run_id=row["last_attempt_run_id"],
            last_attempt_exchange_time=row["last_attempt_exchange_time"],
            attempt_owner_run_id=row["attempt_owner_run_id"],
            attempt_started_at=row["attempt_started_at"],
            attempt_lease_expires_at=row["attempt_lease_expires_at"],
            attempt_heartbeat_at=row["attempt_heartbeat_at"],
            attempt_generation=int(row["attempt_generation"] or 0),
            attempt_token=row["attempt_token"],
            resolution_status=PortfolioReoptimizationResolutionStatus(
                row["resolution_status"] or "UNRESOLVED"
            ),
            resolved_at=row["resolved_at"],
            resolved_by=row["resolved_by"],
            resolution_reason=row["resolution_reason"],
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

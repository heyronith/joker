"""Task 1 SQLite schema migrations.

Creates Task 1 tables if they do not exist. Does not own ``runs`` / run-record
lifecycle — that remains with ``joker.storage.database.Database``.

Note: legacy SQLModel already defines a ``market_snapshots`` table with a
different schema (run_id/payload). ``CREATE TABLE IF NOT EXISTS`` is a no-op when
that legacy table is present. Task 1 ``SnapshotRepository`` typically uses the
same logical table name on a Task-1-oriented database path (see config
``persistence.database_url`` / dedicated data DB). Prefer applying these
migrations to the Task 1 DB path rather than overwriting legacy run storage.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from joker.storage.database import Database

_TASK1_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id TEXT PRIMARY KEY NOT NULL,
    trading_date TEXT NOT NULL,
    exchange_time TEXT NOT NULL,
    session_id TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_task1_market_snapshots_date
    ON market_snapshots (trading_date, exchange_time);

CREATE TABLE IF NOT EXISTS option_surfaces (
    surface_id TEXT PRIMARY KEY NOT NULL,
    trading_date TEXT NOT NULL,
    exchange_time TEXT NOT NULL,
    session_id TEXT,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_task1_option_surfaces_date
    ON option_surfaces (trading_date, exchange_time);

CREATE TABLE IF NOT EXISTS ledger_events (
    ledger_event_id TEXT PRIMARY KEY NOT NULL,
    broker_account_id TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    broker_order_id TEXT,
    contract_id TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT,
    exchange_timestamp TEXT NOT NULL,
    source_event_id TEXT,
    idempotency_key TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    fees TEXT,
    metadata_json TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    position_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_session
    ON ledger_events (session_id, exchange_timestamp, created_at);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_order
    ON ledger_events (client_order_id, exchange_timestamp, created_at);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_contract
    ON ledger_events (contract_id, exchange_timestamp, created_at);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_position
    ON ledger_events (position_id, exchange_timestamp, created_at);

CREATE TABLE IF NOT EXISTS domain_events_seen (
    event_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    seen_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_domain_events_seen_session
    ON domain_events_seen (session_id, seen_at);

CREATE TABLE IF NOT EXISTS graph_checkpoints (
    checkpoint_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    state_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task1_graph_checkpoints_session
    ON graph_checkpoints (session_id, created_at DESC);

CREATE TABLE IF NOT EXISTS data_quality_reports (
    report_id TEXT PRIMARY KEY NOT NULL,
    snapshot_id TEXT,
    session_id TEXT,
    severity TEXT NOT NULL,
    usable_for_reasoning INTEGER NOT NULL,
    usable_for_execution INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_task1_dq_reports_session
    ON data_quality_reports (session_id, created_at);
"""


def apply_task1_migrations(db_path: str | Path) -> Path:
    """Apply Task 1 table DDL to ``db_path`` (create if not exists).

    Reuses ``Database`` only to ensure the parent directory exists and to avoid
    inventing a second run-record owner. Run tables are created via
    ``Database.initialize()`` when callers need them; this function adds Task 1
    tables only via raw SQL ``CREATE TABLE IF NOT EXISTS``.

    Returns the resolved database path.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Touch Database ownership boundary without creating a second run store API.
    # Callers that need run records still use Database.create_run / etc.
    _ = Database(path)

    conn = sqlite3.connect(path)
    try:
        conn.executescript(_TASK1_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path

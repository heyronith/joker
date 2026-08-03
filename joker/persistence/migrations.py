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

_TASK1_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    trading_date TEXT NOT NULL,
    exchange_time TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_trading_date
    ON market_snapshots(trading_date);

CREATE TABLE IF NOT EXISTS option_surfaces (
    surface_id TEXT PRIMARY KEY,
    trading_date TEXT NOT NULL,
    exchange_time TEXT NOT NULL,
    underlying_symbol TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_option_surfaces_trading_date
    ON option_surfaces(trading_date);

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
CREATE INDEX IF NOT EXISTS idx_task1_ledger_session
    ON ledger_events (session_id, exchange_timestamp, ledger_event_id);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_order
    ON ledger_events (client_order_id, exchange_timestamp, ledger_event_id);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_contract
    ON ledger_events (contract_id, exchange_timestamp, ledger_event_id);
CREATE INDEX IF NOT EXISTS idx_task1_ledger_position
    ON ledger_events (position_id, exchange_timestamp, ledger_event_id);

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

CREATE TABLE IF NOT EXISTS cognitive_artifacts (
    artifact_id TEXT PRIMARY KEY NOT NULL,
    artifact_type TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    session_id TEXT NOT NULL,
    cycle_id TEXT,
    snapshot_id TEXT NOT NULL,
    agent_role TEXT,
    prompt_version TEXT,
    model_call_id TEXT,
    parent_artifact_ids_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cognitive_artifacts_session
    ON cognitive_artifacts (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_artifacts_cycle
    ON cognitive_artifacts (session_id, cycle_id, created_at);
CREATE INDEX IF NOT EXISTS idx_cognitive_artifacts_snapshot
    ON cognitive_artifacts (snapshot_id, created_at);

CREATE TABLE IF NOT EXISTS model_calls (
    request_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    cycle_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    agent_role TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    escalation_source TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    error_code TEXT,
    validated_output_artifact_id TEXT,
    validated_output_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_calls_session
    ON model_calls (session_id, started_at);
CREATE INDEX IF NOT EXISTS idx_model_calls_snapshot
    ON model_calls (snapshot_id, started_at);
"""


def apply_task1_migrations(db_path: str | Path) -> Path:
    """Apply Task 1 table DDL to ``db_path`` (create if not exists).

    Does not instantiate ``joker.storage.database.Database`` so legacy SQLModel
    tables cannot shadow Task 1 schemas via ``CREATE TABLE IF NOT EXISTS``.

    Also applies Task 3 evolution DDL idempotently so paper sessions can enable
    evolution without a separate migration step.

    Returns the resolved database path.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_TASK1_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    from joker.objectives.repository import apply_objective_migrations
    from joker.evolution.migrations import apply_task3_migrations

    apply_objective_migrations(path)
    apply_task3_migrations(path)
    from joker.persistence.broker_submission_journal import (
        apply_broker_submission_journal_migration,
    )

    apply_broker_submission_journal_migration(path)
    return path

"""Task 3 SQLite schema migrations (append-only evolution artefacts)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_TASK3_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trading_episodes (
    episode_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    action_class TEXT NOT NULL,
    completed INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_session
    ON trading_episodes (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_date
    ON trading_episodes (trading_date, created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_config
    ON trading_episodes (configuration_version_id, created_at);

CREATE TABLE IF NOT EXISTS decision_trace_summaries (
    summary_id TEXT PRIMARY KEY NOT NULL,
    episode_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_episode
    ON decision_trace_summaries (episode_id, created_at);

CREATE TABLE IF NOT EXISTS episode_evaluations (
    evaluation_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    episode_id TEXT NOT NULL,
    configuration_version_id TEXT,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_episode
    ON episode_evaluations (episode_id, created_at);

CREATE TABLE IF NOT EXISTS evaluation_datasets (
    dataset_id TEXT PRIMARY KEY NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_episode_membership (
    dataset_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    partition_name TEXT NOT NULL,
    PRIMARY KEY (dataset_id, episode_id)
);
CREATE INDEX IF NOT EXISTS idx_dataset_partition
    ON dataset_episode_membership (dataset_id, partition_name);

CREATE TABLE IF NOT EXISTS cognitive_configuration_versions (
    configuration_version_id TEXT PRIMARY KEY NOT NULL,
    parent_version_id TEXT,
    status TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    scope_key TEXT NOT NULL DEFAULT 'default',
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_config_status
    ON cognitive_configuration_versions (scope_key, status, created_at);

CREATE TABLE IF NOT EXISTS prompt_versions (
    prompt_version_id TEXT PRIMARY KEY NOT NULL,
    role TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_policy_versions (
    version_id TEXT PRIMARY KEY NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_policy_versions (
    version_id TEXT PRIMARY KEY NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS debate_policy_versions (
    version_id TEXT PRIMARY KEY NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routing_policy_versions (
    version_id TEXT PRIMARY KEY NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS escalation_policy_versions (
    version_id TEXT PRIMARY KEY NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS improvement_proposals (
    proposal_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    parent_champion_version_id TEXT NOT NULL,
    status TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_definitions (
    experiment_id TEXT PRIMARY KEY NOT NULL,
    status TEXT NOT NULL,
    champion_version_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    recovery_cursor TEXT,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_status
    ON experiment_definitions (status, created_at);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_slice_results (
    result_row_id TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL,
    slice_name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_results (
    result_id TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_decisions (
    promotion_decision_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    final_status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS champion_history (
    transition_id TEXT PRIMARY KEY NOT NULL,
    scope_key TEXT NOT NULL,
    previous_version_id TEXT,
    new_version_id TEXT NOT NULL,
    activated_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_champion_scope
    ON champion_history (scope_key, activated_at DESC);

CREATE TABLE IF NOT EXISTS champion_current (
    scope_key TEXT PRIMARY KEY NOT NULL,
    configuration_version_id TEXT NOT NULL,
    transition_id TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_assignments (
    assignment_id TEXT PRIMARY KEY NOT NULL,
    challenger_version_id TEXT NOT NULL,
    champion_version_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS drift_observations (
    observation_id TEXT PRIMARY KEY NOT NULL,
    configuration_version_id TEXT NOT NULL,
    dimension TEXT NOT NULL,
    severity TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rollback_records (
    rollback_id TEXT PRIMARY KEY NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    rolled_back_version_id TEXT NOT NULL,
    restored_version_id TEXT NOT NULL,
    recovery_status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    detection_timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_cycles (
    cycle_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, cycle_id)
);
CREATE INDEX IF NOT EXISTS idx_evolution_cycles_status
    ON evolution_cycles (session_id, status);

CREATE TABLE IF NOT EXISTS memory_lesson_entries (
    lesson_id TEXT PRIMARY KEY NOT NULL,
    lesson_type TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_hypothetical_commands (
    command_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    snapshot_id TEXT,
    cycle_id TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evolution_evidence_claims (
    claim_id TEXT PRIMARY KEY NOT NULL,
    evaluation_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    evolution_cycle_id TEXT NOT NULL,
    dataset_id TEXT,
    claim_status TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    released_at TEXT,
    claim_reason TEXT NOT NULL,
    reuse_reason TEXT,
    prior_claim_id TEXT,
    actor TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_cycle
    ON evolution_evidence_claims (evolution_cycle_id, claim_status);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_status
    ON evolution_evidence_claims (claim_status, claimed_at);
CREATE INDEX IF NOT EXISTS idx_evidence_claims_evaluation
    ON evolution_evidence_claims (evaluation_id, claim_status);

CREATE TABLE IF NOT EXISTS adversarial_scenario_results (
    result_key TEXT PRIMARY KEY NOT NULL,
    experiment_id TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    scenario_version TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    sample_number INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    findings_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adv_results_experiment
    ON adversarial_scenario_results (experiment_id, scenario_id);

CREATE TABLE IF NOT EXISTS shadow_cycles (
    shadow_cycle_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shadow_cycles_assignment
    ON shadow_cycles (assignment_id, created_at);

CREATE TABLE IF NOT EXISTS shadow_orders (
    client_order_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    position_lifecycle_id TEXT,
    contract_id TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_fills (
    fill_id TEXT PRIMARY KEY NOT NULL,
    client_order_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    quantity TEXT NOT NULL,
    price TEXT NOT NULL,
    fee TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_shadow_fills_order
    ON shadow_fills (client_order_id, quantity, price);

CREATE TABLE IF NOT EXISTS shadow_positions (
    position_key TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    position_lifecycle_id TEXT NOT NULL,
    contract_id TEXT NOT NULL,
    configuration_version_id TEXT NOT NULL,
    quantity TEXT NOT NULL,
    average_price TEXT NOT NULL,
    realised_pnl TEXT NOT NULL,
    status TEXT NOT NULL,
    last_snapshot_id TEXT,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_position_events (
    event_id TEXT PRIMARY KEY NOT NULL,
    position_lifecycle_id TEXT NOT NULL,
    assignment_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_runtime_checkpoints (
    assignment_id TEXT PRIMARY KEY NOT NULL,
    last_snapshot_id TEXT,
    cursor_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shadow_evidence_summaries (
    shadow_evidence_id TEXT PRIMARY KEY NOT NULL,
    assignment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    champion_version_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS champion_activations (
    activation_id TEXT PRIMARY KEY NOT NULL,
    promotion_decision_id TEXT NOT NULL UNIQUE,
    experiment_id TEXT NOT NULL,
    challenger_version_id TEXT NOT NULL,
    previous_champion_version_id TEXT NOT NULL,
    registry_applied INTEGER NOT NULL DEFAULT 0,
    history_verified INTEGER NOT NULL DEFAULT 0,
    configuration_status_applied INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    failure_codes_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_champion_activations_decision
    ON champion_activations (promotion_decision_id);

CREATE TABLE IF NOT EXISTS session_event_index (
    event_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    exchange_timestamp TEXT NOT NULL,
    sequence INTEGER,
    correlation_id TEXT,
    cycle_id TEXT,
    snapshot_id TEXT,
    data_quality_id TEXT,
    option_surface_id TEXT,
    client_order_id TEXT,
    contract_id TEXT,
    position_lifecycle_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_event_horizon
    ON session_event_index (session_id, exchange_timestamp, sequence, event_id);
CREATE INDEX IF NOT EXISTS idx_session_event_cycle
    ON session_event_index (session_id, cycle_id, event_type);

CREATE TABLE IF NOT EXISTS adversarial_recovery_checkpoints (
  checkpoint_key TEXT PRIMARY KEY NOT NULL,
  experiment_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  scenario_version TEXT NOT NULL,
  configuration_version_id TEXT NOT NULL,
  sample_number INTEGER NOT NULL,
  crash_point TEXT,
  graph_thread_ids_json TEXT NOT NULL,
  cash TEXT NOT NULL,
  submitted_keys_json TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
"""


def apply_task3_migrations(db_path: str | Path) -> Path:
    """Apply Task 3 DDL to ``db_path`` (create-if-not-exists, idempotent)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Allow brief contention with aiosqlite workers during parallel Task-3 startup.
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.executescript(_TASK3_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path

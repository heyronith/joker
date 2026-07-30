"""SQLite persistence for session objectives and capital reservations."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from joker.objectives.schemas import (
    CapitalReservation,
    GoalFeasibilityAssessment,
    ObjectiveStrategyScore,
    SessionObjectiveDefinition,
    SessionObjectiveState,
)

OBJECTIVE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_objective_definitions (
    objective_id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    definition_version INTEGER NOT NULL,
    armed INTEGER NOT NULL DEFAULT 0,
    first_broker_submission_at TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_def_session
    ON session_objective_definitions (session_id, created_at);

CREATE TABLE IF NOT EXISTS session_objective_state_versions (
    state_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    objective_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (objective_id, version)
);
CREATE INDEX IF NOT EXISTS idx_obj_state_session
    ON session_objective_state_versions (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_obj_state_status
    ON session_objective_state_versions (status, created_at);

CREATE TABLE IF NOT EXISTS capital_reservations (
    reservation_id TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    client_order_id TEXT NOT NULL UNIQUE,
    broker_order_id TEXT,
    estimated_premium_usd TEXT NOT NULL,
    reserved_usd TEXT NOT NULL,
    status TEXT NOT NULL,
    objective_state_version INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cap_res_objective
    ON capital_reservations (objective_id, status);
CREATE INDEX IF NOT EXISTS idx_cap_res_session
    ON capital_reservations (session_id, status);

CREATE TABLE IF NOT EXISTS objective_feasibility_assessments (
    assessment_id TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    classification TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_feas_objective
    ON objective_feasibility_assessments (objective_id, created_at);

CREATE TABLE IF NOT EXISTS objective_strategy_scores (
    score_id TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    strategy_id TEXT,
    snapshot_id TEXT NOT NULL,
    valid INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_score_objective
    ON objective_strategy_scores (objective_id, created_at);

CREATE TABLE IF NOT EXISTS objective_decision_audit (
    audit_id TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_audit_session
    ON objective_decision_audit (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_obj_audit_objective
    ON objective_decision_audit (objective_id, created_at);
"""


def apply_objective_migrations(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(OBJECTIVE_SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()
    return path


def _dumps(model: Any) -> str:
    if hasattr(model, "model_dump"):
        return json.dumps(model.model_dump(mode="json"), sort_keys=True)
    return json.dumps(model, sort_keys=True)


class ObjectiveRepository:
    """Task 1 objective persistence on the Task 1 SQLite database."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        apply_objective_migrations(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def save_definition(self, definition: SessionObjectiveDefinition) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_objective_definitions (
                    objective_id, session_id, definition_version, armed,
                    first_broker_submission_at, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(definition.objective_id),
                    definition.session_id,
                    definition.definition_version,
                    1 if definition.armed else 0,
                    definition.first_broker_submission_at.isoformat()
                    if definition.first_broker_submission_at
                    else None,
                    _dumps(definition),
                    definition.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_definition(self, objective_id: UUID | str) -> SessionObjectiveDefinition | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM session_objective_definitions WHERE objective_id=?",
                (str(objective_id),),
            ).fetchone()
        if row is None:
            return None
        return SessionObjectiveDefinition.model_validate_json(row["payload_json"])

    def latest_definition_for_session(
        self, session_id: str
    ) -> SessionObjectiveDefinition | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM session_objective_definitions
                WHERE session_id=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionObjectiveDefinition.model_validate_json(row["payload_json"])

    def append_state(self, state: SessionObjectiveState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_objective_state_versions (
                    objective_id, session_id, version, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(state.objective_id),
                    state.session_id,
                    state.version,
                    state.status,
                    _dumps(state),
                    state.last_recomputed_at.isoformat(),
                ),
            )
            conn.commit()

    def latest_state(self, objective_id: UUID | str) -> SessionObjectiveState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM session_objective_state_versions
                WHERE objective_id=?
                ORDER BY version DESC LIMIT 1
                """,
                (str(objective_id),),
            ).fetchone()
        if row is None:
            return None
        return SessionObjectiveState.model_validate_json(row["payload_json"])

    def latest_state_for_session(self, session_id: str) -> SessionObjectiveState | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM session_objective_state_versions
                WHERE session_id=?
                ORDER BY version DESC, created_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return SessionObjectiveState.model_validate_json(row["payload_json"])

    def get_reservation_by_client_order(
        self, client_order_id: str
    ) -> CapitalReservation | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM capital_reservations WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        return CapitalReservation.model_validate_json(row["payload_json"])

    def upsert_reservation(self, reservation: CapitalReservation) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO capital_reservations (
                    reservation_id, objective_id, session_id, client_order_id,
                    broker_order_id, estimated_premium_usd, reserved_usd, status,
                    objective_state_version, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO UPDATE SET
                    broker_order_id=excluded.broker_order_id,
                    reserved_usd=excluded.reserved_usd,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    str(reservation.reservation_id),
                    str(reservation.objective_id),
                    reservation.session_id,
                    reservation.client_order_id,
                    reservation.broker_order_id,
                    str(reservation.estimated_premium_usd),
                    str(reservation.reserved_usd),
                    reservation.status,
                    reservation.objective_state_version,
                    _dumps(reservation),
                    reservation.created_at.isoformat(),
                    reservation.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def list_open_reservations(self, objective_id: UUID | str) -> list[CapitalReservation]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM capital_reservations
                WHERE objective_id=? AND status IN ('open', 'partial')
                """,
                (str(objective_id),),
            ).fetchall()
        return [CapitalReservation.model_validate_json(r["payload_json"]) for r in rows]

    def save_feasibility(self, assessment: GoalFeasibilityAssessment) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_feasibility_assessments (
                    assessment_id, objective_id, snapshot_id, classification,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(assessment.assessment_id),
                    str(assessment.objective_id),
                    str(assessment.snapshot_id),
                    assessment.classification,
                    _dumps(assessment),
                    assessment.created_at.isoformat(),
                ),
            )
            conn.commit()

    def save_strategy_score(self, score: ObjectiveStrategyScore) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_strategy_scores (
                    score_id, objective_id, strategy_id, snapshot_id, valid,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(score.score_id),
                    str(score.objective_id),
                    str(score.strategy_id) if score.strategy_id else None,
                    str(score.snapshot_id),
                    1 if score.valid else 0,
                    _dumps(score),
                    datetime.now().astimezone().isoformat(),
                ),
            )
            conn.commit()

    def append_audit(
        self,
        *,
        audit_id: str,
        objective_id: UUID | str,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO objective_decision_audit (
                    audit_id, objective_id, session_id, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    str(objective_id),
                    session_id,
                    event_type,
                    json.dumps(payload, sort_keys=True),
                    created_at.isoformat(),
                ),
            )
            conn.commit()

    def compare_and_swap_state_version(
        self,
        *,
        objective_id: UUID | str,
        expected_version: int,
        new_state: SessionObjectiveState,
    ) -> bool:
        """Optimistic version guard used by reserve/release."""
        with self._connect() as conn:
            latest = conn.execute(
                """
                SELECT version FROM session_objective_state_versions
                WHERE objective_id=? ORDER BY version DESC LIMIT 1
                """,
                (str(objective_id),),
            ).fetchone()
            current = int(latest["version"]) if latest else 0
            if current != expected_version:
                return False
            conn.execute(
                """
                INSERT INTO session_objective_state_versions (
                    objective_id, session_id, version, status, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(new_state.objective_id),
                    new_state.session_id,
                    new_state.version,
                    new_state.status,
                    _dumps(new_state),
                    new_state.last_recomputed_at.isoformat(),
                ),
            )
            conn.commit()
            return True

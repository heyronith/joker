"""SQLite persistence for session objectives and capital exposures."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from joker.objectives.historical_schemas import (
    HistoricalLeakageReport,
    HistoricalOutcomeQuery,
    HistoricalOutcomeSummary,
)
from joker.objectives.schemas import (
    CapitalExposure,
    GoalFeasibilityAssessment,
    ObjectiveStrategyScore,
    SessionObjectiveDefinition,
    SessionObjectiveState,
    StrategyObjectiveEstimate,
)

CrashPoint = Literal[
    "before_transaction",
    "after_exposure_write",
    "after_state_append",
    "after_audit_append",
    "after_broker_accept_before_association",
]


class CrashInjected(RuntimeError):
    """Raised by crash-injection hooks during atomic objective mutations."""


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
CREATE INDEX IF NOT EXISTS idx_obj_score_strategy
    ON objective_strategy_scores (strategy_id, snapshot_id);

CREATE TABLE IF NOT EXISTS objective_strategy_estimates (
    estimate_id TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    valid INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_est_strategy
    ON objective_strategy_estimates (strategy_id, snapshot_id);
CREATE INDEX IF NOT EXISTS idx_obj_est_objective
    ON objective_strategy_estimates (objective_id, created_at);

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

CREATE TABLE IF NOT EXISTS objective_projection_dedupe (
    dedupe_key TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_dedupe_objective
    ON objective_projection_dedupe (objective_id, created_at);

CREATE TABLE IF NOT EXISTS objective_historical_queries (
    query_id TEXT PRIMARY KEY NOT NULL,
    objective_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_hist_query_strategy
    ON objective_historical_queries (strategy_id, snapshot_id);

CREATE TABLE IF NOT EXISTS objective_historical_summaries (
    summary_id TEXT PRIMARY KEY NOT NULL,
    query_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL,
    valid_for_ev INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obj_hist_summary_query
    ON objective_historical_summaries (query_id);
CREATE INDEX IF NOT EXISTS idx_obj_hist_summary_strategy
    ON objective_historical_summaries (strategy_id, snapshot_id);

CREATE TABLE IF NOT EXISTS objective_historical_leakage_reports (
    query_id TEXT PRIMARY KEY NOT NULL,
    safe INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

_ENCUMBERING_STATUSES = (
    "working_order_reservation",
    "filled_position_exposure",
    "partial",
    # legacy statuses from earlier schema versions
    "open",
    "converted",
)


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


def _normalize_exposure_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy reservation payloads to CapitalExposure shape."""
    data = dict(raw)
    status = str(data.get("status") or "working_order_reservation")
    if status == "open":
        status = "working_order_reservation"
    elif status == "converted":
        status = "filled_position_exposure"
    data["status"] = status

    if "exposure_id" not in data and "reservation_id" in data:
        data["exposure_id"] = data["reservation_id"]
    if "estimated_premium_per_contract_usd" not in data:
        est = data.get("estimated_premium_usd") or "0"
        qty = int(data.get("requested_quantity") or 1)
        # Legacy stored total notional; recover per-contract premium.
        from decimal import Decimal

        total = Decimal(str(est))
        per = (total / (Decimal("100") * Decimal(max(1, qty)))).quantize(Decimal("0.01"))
        data["estimated_premium_per_contract_usd"] = str(per)
    if "working_order_reservation_usd" not in data or "filled_exposure_usd" not in data:
        reserved = data.get("reserved_usd") or "0"
        if status in {"working_order_reservation", "open", "partial"}:
            if status == "partial":
                data.setdefault("working_order_reservation_usd", reserved)
                data.setdefault("filled_exposure_usd", "0.00")
            elif status in {"filled_position_exposure", "converted"}:
                data.setdefault("working_order_reservation_usd", "0.00")
                data.setdefault("filled_exposure_usd", reserved)
            else:
                data.setdefault("working_order_reservation_usd", reserved)
                data.setdefault("filled_exposure_usd", "0.00")
        elif status in {"filled_position_exposure", "converted"}:
            data.setdefault("working_order_reservation_usd", "0.00")
            data.setdefault("filled_exposure_usd", reserved)
        else:
            data.setdefault("working_order_reservation_usd", "0.00")
            data.setdefault("filled_exposure_usd", "0.00")
    data.setdefault("requested_quantity", 1)
    data.setdefault("working_quantity", data.get("requested_quantity", 1))
    data.setdefault("filled_quantity", 0)
    return data


class ObjectiveRepository:
    """Task 1 objective persistence on the Task 1 SQLite database."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        crash_hook: Callable[[CrashPoint], None] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self._crash_hook = crash_hook
        apply_objective_migrations(self.db_path)

    def set_crash_hook(self, hook: Callable[[CrashPoint], None] | None) -> None:
        self._crash_hook = hook

    def _maybe_crash(self, point: CrashPoint) -> None:
        if self._crash_hook is not None:
            self._crash_hook(point)

    def _connect(self) -> sqlite3.Connection:
        # Share the Task-1/3 SQLite file with aiosqlite workers; WAL + busy
        # timeout avoid fail-closed on transient multi-writer contention.
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode=WAL")
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

    def list_sessions_for_account_identity(
        self,
        *,
        account_identity: str,
        mode: str = "paper",
    ) -> list[str]:
        """List durable objective session ids for one stable account identity."""
        prefix = (
            f"cog:{(mode or 'paper').strip().lower()}:"
            f"{(account_identity or '').strip().lower()}:"
        )
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, MAX(created_at) AS latest_created_at
                FROM session_objective_definitions
                WHERE LOWER(session_id) LIKE ?
                GROUP BY session_id
                ORDER BY latest_created_at DESC, session_id ASC
                """,
                (prefix + "%",),
            ).fetchall()
        return [str(row["session_id"]) for row in rows if row["session_id"]]

    def append_state(self, state: SessionObjectiveState) -> None:
        with self._connect() as conn:
            self._insert_state(conn, state)
            conn.commit()

    def _insert_state(self, conn: sqlite3.Connection, state: SessionObjectiveState) -> None:
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
    ) -> CapitalExposure | None:
        return self.get_exposure_by_client_order(client_order_id)

    def get_exposure_by_client_order(self, client_order_id: str) -> CapitalExposure | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM capital_reservations WHERE client_order_id=?",
                (client_order_id,),
            ).fetchone()
        if row is None:
            return None
        payload = _normalize_exposure_payload(json.loads(row["payload_json"]))
        return CapitalExposure.model_validate(payload)

    def upsert_reservation(self, reservation: CapitalExposure) -> None:
        self.upsert_exposure(reservation)

    def upsert_exposure(self, exposure: CapitalExposure) -> None:
        with self._connect() as conn:
            self._upsert_exposure(conn, exposure)
            conn.commit()

    def _upsert_exposure(self, conn: sqlite3.Connection, exposure: CapitalExposure) -> None:
        reserved = str(exposure.total_encumbered_usd)
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
                updated_at=excluded.updated_at,
                objective_state_version=excluded.objective_state_version
            """,
            (
                str(exposure.exposure_id),
                str(exposure.objective_id),
                exposure.session_id,
                exposure.client_order_id,
                exposure.broker_order_id,
                str(exposure.estimated_premium_usd),
                reserved,
                exposure.status,
                exposure.objective_state_version,
                _dumps(exposure),
                exposure.created_at.isoformat(),
                exposure.updated_at.isoformat(),
            ),
        )

    def list_open_reservations(self, objective_id: UUID | str) -> list[CapitalExposure]:
        return self.list_encumbering_exposures(objective_id)

    def list_encumbering_exposures(self, objective_id: UUID | str) -> list[CapitalExposure]:
        placeholders = ",".join("?" for _ in _ENCUMBERING_STATUSES)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT payload_json FROM capital_reservations
                WHERE objective_id=? AND status IN ({placeholders})
                """,
                (str(objective_id), *_ENCUMBERING_STATUSES),
            ).fetchall()
        out: list[CapitalExposure] = []
        for r in rows:
            payload = _normalize_exposure_payload(json.loads(r["payload_json"]))
            out.append(CapitalExposure.model_validate(payload))
        return out

    def list_all_exposures(self, objective_id: UUID | str) -> list[CapitalExposure]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload_json FROM capital_reservations WHERE objective_id=?",
                (str(objective_id),),
            ).fetchall()
        out: list[CapitalExposure] = []
        for r in rows:
            payload = _normalize_exposure_payload(json.loads(r["payload_json"]))
            out.append(CapitalExposure.model_validate(payload))
        return out

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

    def list_strategy_scores_for_snapshot(
        self, *, objective_id: UUID | str, snapshot_id: UUID | str
    ) -> list[ObjectiveStrategyScore]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json FROM objective_strategy_scores
                WHERE objective_id=? AND snapshot_id=?
                ORDER BY created_at ASC
                """,
                (str(objective_id), str(snapshot_id)),
            ).fetchall()
        return [ObjectiveStrategyScore.model_validate_json(r["payload_json"]) for r in rows]

    def save_strategy_estimate(self, estimate: StrategyObjectiveEstimate) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_strategy_estimates (
                    estimate_id, objective_id, strategy_id, snapshot_id, valid,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(estimate.estimate_id),
                    str(estimate.objective_id),
                    str(estimate.strategy_id),
                    str(estimate.snapshot_id),
                    1 if estimate.valid else 0,
                    _dumps(estimate),
                    estimate.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_strategy_estimate(
        self, estimate_id: UUID | str
    ) -> StrategyObjectiveEstimate | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM objective_strategy_estimates WHERE estimate_id=?",
                (str(estimate_id),),
            ).fetchone()
        if row is None:
            return None
        return StrategyObjectiveEstimate.model_validate_json(row["payload_json"])

    def get_latest_estimate_for_strategy(
        self, *, strategy_id: UUID | str, objective_id: UUID | str
    ) -> StrategyObjectiveEstimate | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM objective_strategy_estimates
                WHERE strategy_id=? AND objective_id=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (str(strategy_id), str(objective_id)),
            ).fetchone()
        if row is None:
            return None
        return StrategyObjectiveEstimate.model_validate_json(row["payload_json"])

    def save_historical_query(self, query: HistoricalOutcomeQuery) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_historical_queries (
                    query_id, objective_id, strategy_id, snapshot_id,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(query.query_id),
                    str(query.objective_id),
                    str(query.strategy_id),
                    str(query.snapshot_id),
                    _dumps(query),
                    query.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_historical_query(
        self, query_id: UUID | str
    ) -> HistoricalOutcomeQuery | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM objective_historical_queries WHERE query_id=?",
                (str(query_id),),
            ).fetchone()
        if row is None:
            return None
        return HistoricalOutcomeQuery.model_validate_json(row["payload_json"])

    def save_historical_summary(self, summary: HistoricalOutcomeSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_historical_summaries (
                    summary_id, query_id, strategy_id, snapshot_id, valid_for_ev,
                    payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(summary.summary_id),
                    str(summary.query_id),
                    str(summary.strategy_id),
                    str(summary.snapshot_id),
                    1 if summary.valid_for_ev else 0,
                    _dumps(summary),
                    summary.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_historical_summary(
        self, summary_id: UUID | str
    ) -> HistoricalOutcomeSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM objective_historical_summaries WHERE summary_id=?",
                (str(summary_id),),
            ).fetchone()
        if row is None:
            return None
        return HistoricalOutcomeSummary.model_validate_json(row["payload_json"])

    def get_latest_historical_summary_for_strategy(
        self, *, strategy_id: UUID | str, snapshot_id: UUID | str
    ) -> HistoricalOutcomeSummary | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload_json FROM objective_historical_summaries
                WHERE strategy_id=? AND snapshot_id=?
                ORDER BY created_at DESC LIMIT 1
                """,
                (str(strategy_id), str(snapshot_id)),
            ).fetchone()
        if row is None:
            return None
        return HistoricalOutcomeSummary.model_validate_json(row["payload_json"])

    def save_leakage_report(self, report: HistoricalLeakageReport) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO objective_historical_leakage_reports (
                    query_id, safe, payload_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    str(report.query_id),
                    1 if report.safe else 0,
                    _dumps(report),
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
            self._append_audit(
                conn,
                audit_id=audit_id,
                objective_id=objective_id,
                session_id=session_id,
                event_type=event_type,
                payload=payload,
                created_at=created_at,
            )
            conn.commit()

    def _append_audit(
        self,
        conn: sqlite3.Connection,
        *,
        audit_id: str,
        objective_id: UUID | str,
        session_id: str,
        event_type: str,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
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

    def has_projection_dedupe(self, dedupe_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM objective_projection_dedupe WHERE dedupe_key=?",
                (dedupe_key,),
            ).fetchone()
        return row is not None

    def compare_and_swap_state_version(
        self,
        *,
        objective_id: UUID | str,
        expected_version: int,
        new_state: SessionObjectiveState,
    ) -> bool:
        """Optimistic version guard (legacy non-exposure path)."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                """
                SELECT version FROM session_objective_state_versions
                WHERE objective_id=? ORDER BY version DESC LIMIT 1
                """,
                (str(objective_id),),
            ).fetchone()
            current = int(latest["version"]) if latest else 0
            if current != expected_version:
                conn.rollback()
                return False
            self._insert_state(conn, new_state)
            conn.commit()
            return True

    def atomic_mutate_exposure(
        self,
        *,
        objective_id: UUID | str,
        expected_version: int,
        exposure: CapitalExposure,
        new_state: SessionObjectiveState,
        audit: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> bool:
        """Atomically: version check → exposure upsert → state append → audit/dedupe.

        Uses ``BEGIN IMMEDIATE`` so concurrent writers cannot interleave.
        """
        self._maybe_crash("before_transaction")
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                if dedupe_key is not None:
                    existing = conn.execute(
                        "SELECT 1 FROM objective_projection_dedupe WHERE dedupe_key=?",
                        (dedupe_key,),
                    ).fetchone()
                    if existing is not None:
                        conn.rollback()
                        return True  # already applied — idempotent success
                latest = conn.execute(
                    """
                    SELECT version FROM session_objective_state_versions
                    WHERE objective_id=? ORDER BY version DESC LIMIT 1
                    """,
                    (str(objective_id),),
                ).fetchone()
                current = int(latest["version"]) if latest else 0
                if current != expected_version:
                    conn.rollback()
                    return False
                self._upsert_exposure(conn, exposure)
                self._maybe_crash("after_exposure_write")
                self._insert_state(conn, new_state)
                self._maybe_crash("after_state_append")
                if audit is not None:
                    self._append_audit(
                        conn,
                        audit_id=str(audit["audit_id"]),
                        objective_id=objective_id,
                        session_id=str(audit["session_id"]),
                        event_type=str(audit["event_type"]),
                        payload=dict(audit.get("payload") or {}),
                        created_at=audit["created_at"],
                    )
                    self._maybe_crash("after_audit_append")
                if dedupe_key is not None:
                    conn.execute(
                        """
                        INSERT INTO objective_projection_dedupe (
                            dedupe_key, objective_id, event_type, created_at
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            dedupe_key,
                            str(objective_id),
                            str((audit or {}).get("event_type") or "exposure_mutation"),
                            new_state.last_recomputed_at.isoformat(),
                        ),
                    )
                conn.commit()
                return True
            except CrashInjected:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

    def atomic_associate_broker_order(
        self,
        *,
        client_order_id: str,
        broker_order_id: str,
        crash_after_accept: bool = False,
    ) -> CapitalExposure | None:
        """Associate broker order id; optional crash before commit for tests."""
        with self._connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload_json FROM capital_reservations WHERE client_order_id=?",
                    (client_order_id,),
                ).fetchone()
                if row is None:
                    conn.rollback()
                    return None
                payload = _normalize_exposure_payload(json.loads(row["payload_json"]))
                exposure = CapitalExposure.model_validate(payload)
                updated = exposure.model_copy(
                    update={
                        "broker_order_id": broker_order_id,
                        "updated_at": datetime.now().astimezone(),
                    }
                )
                self._upsert_exposure(conn, updated)
                if crash_after_accept:
                    self._maybe_crash("after_broker_accept_before_association")
                conn.commit()
                return updated
            except CrashInjected:
                conn.rollback()
                raise
            except Exception:
                conn.rollback()
                raise

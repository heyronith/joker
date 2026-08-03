"""Persistent account/session P&L baseline — survives process restart."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS session_pnl_baseline (
    account_id_hash TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    session_id TEXT NOT NULL,
    starting_nlv TEXT,
    starting_cash TEXT,
    captured_at TEXT NOT NULL,
    external_cash_adjustment TEXT,
    PRIMARY KEY (account_id_hash, trading_date, session_id)
);
"""


@dataclass(frozen=True)
class SessionPnlBaseline:
    account_id_hash: str
    trading_date: str
    session_id: str
    starting_nlv: Decimal | None
    starting_cash: Decimal | None
    captured_at: datetime
    external_cash_adjustment: Decimal | None = None


class SessionPnlBaselineStore:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        apply_session_pnl_baseline_migration(self._db_path)

    def get(
        self, *, account_id_hash: str, trading_date: str, session_id: str
    ) -> SessionPnlBaseline | None:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                """
                SELECT * FROM session_pnl_baseline
                WHERE account_id_hash = ? AND trading_date = ? AND session_id = ?
                """,
                (account_id_hash, trading_date, session_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return SessionPnlBaseline(
            account_id_hash=row["account_id_hash"],
            trading_date=row["trading_date"],
            session_id=row["session_id"],
            starting_nlv=_dec(row["starting_nlv"]),
            starting_cash=_dec(row["starting_cash"]),
            captured_at=datetime.fromisoformat(row["captured_at"]),
            external_cash_adjustment=_dec(row["external_cash_adjustment"]),
        )

    def put(self, baseline: SessionPnlBaseline) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO session_pnl_baseline (
                    account_id_hash, trading_date, session_id,
                    starting_nlv, starting_cash, captured_at,
                    external_cash_adjustment
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    baseline.account_id_hash,
                    baseline.trading_date,
                    baseline.session_id,
                    _fmt(baseline.starting_nlv),
                    _fmt(baseline.starting_cash),
                    baseline.captured_at.isoformat(),
                    _fmt(baseline.external_cash_adjustment),
                ),
            )
            conn.commit()
        finally:
            conn.close()


def apply_session_pnl_baseline_migration(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_CREATE_SQL)
        conn.commit()
    finally:
        conn.close()


def _dec(raw: str | None) -> Decimal | None:
    if raw is None or raw == "":
        return None
    return Decimal(str(raw))


def _fmt(value: Decimal | None) -> str | None:
    return str(value) if value is not None else None

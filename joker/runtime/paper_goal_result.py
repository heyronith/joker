"""Paper goal-test classification and redacted evidence package."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

PaperGoalClassification = Literal[
    "PAPER_OBJECTIVE_ACHIEVED",
    "PAPER_OBJECTIVE_MISSED",
    "PAPER_OBJECTIVE_INCONCLUSIVE",
    "PAPER_SESSION_FAILED",
]

_SECRET_KEY_RE = re.compile(
    r"(secret|token|password|credential|api[_-]?key|access[_-]?token|"
    r"account[_-]?id|authorization|nonce)",
    re.IGNORECASE,
)


@dataclass
class PaperGoalResult:
    """Final one-hour paper goal-test outcome."""

    classification: PaperGoalClassification
    objective_id: str | None
    session_id: str
    authorized_capital_usd: float
    target_profit_pct: float
    target_profit_usd: float
    objective_duration_minutes: float
    starting_realized_pnl_usd: float
    ending_realized_pnl_usd: float
    max_unrealized_gain_usd: float | None = None
    max_drawdown_usd: float | None = None
    capital_reserved_peak_usd: float | None = None
    graph_cycles: int = 0
    entry_proposals: int = 0
    entry_approvals: int = 0
    trades_entered: int = 0
    trades_exited: int = 0
    wins: int = 0
    losses: int = 0
    no_trade_decisions: int = 0
    goal_achieved: bool = False
    time_goal_achieved: str | None = None
    open_positions_remaining: int = 0
    working_orders_remaining: int = 0
    reconciliation_clean: bool | None = None
    reason: str = ""
    trades: list[dict[str, Any]] = field(default_factory=list)
    blocking_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def redact_mapping(value: Any) -> Any:
    """Recursively drop secret-looking keys and mask long identifier strings."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                continue
            out[str(key)] = redact_mapping(item)
        return out
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, str) and len(value) > 24 and value.isalnum():
        return f"{value[:4]}…{value[-4:]}"
    return value


def classify_paper_goal(
    *,
    ending_realized_pnl_usd: float,
    target_profit_usd: float,
    open_positions_remaining: int,
    working_orders_remaining: int,
    reconciliation_clean: bool | None,
    deadline_reached: bool,
    system_operational: bool,
    unresolved_submission_unknown: bool = False,
    market_data_interrupted: bool = False,
    execution_unusable_surface: bool = False,
    broker_outage: bool = False,
    model_provider_outage: bool = False,
    invariant_failures: list[str] | None = None,
    session_failed_errors: list[str] | None = None,
) -> tuple[PaperGoalClassification, str]:
    """Return classification + reason for a completed paper goal session."""
    failures = list(invariant_failures or []) + list(session_failed_errors or [])
    if failures:
        return "PAPER_SESSION_FAILED", "; ".join(failures[:8])

    inconclusive_bits: list[str] = []
    if unresolved_submission_unknown:
        inconclusive_bits.append("unresolved_submission_unknown")
    if open_positions_remaining > 0:
        inconclusive_bits.append("unresolved_position")
    if working_orders_remaining > 0:
        inconclusive_bits.append("unresolved_working_order")
    if market_data_interrupted:
        inconclusive_bits.append("market_data_interruption")
    if execution_unusable_surface:
        inconclusive_bits.append("execution_unusable_option_surface")
    if broker_outage:
        inconclusive_bits.append("broker_outage")
    if model_provider_outage:
        inconclusive_bits.append("model_provider_outage")
    if reconciliation_clean is False:
        inconclusive_bits.append("reconciliation_not_clean")
    if inconclusive_bits:
        return "PAPER_OBJECTIVE_INCONCLUSIVE", ",".join(inconclusive_bits)

    if not system_operational:
        return "PAPER_SESSION_FAILED", "system_not_operational"

    positions_terminal = open_positions_remaining == 0 and working_orders_remaining == 0
    if (
        ending_realized_pnl_usd >= target_profit_usd
        and positions_terminal
        and reconciliation_clean is not False
    ):
        return (
            "PAPER_OBJECTIVE_ACHIEVED",
            "realized_pnl_met_or_exceeded_target_with_terminal_flat_state",
        )

    if deadline_reached and positions_terminal and system_operational:
        return (
            "PAPER_OBJECTIVE_MISSED",
            "deadline_reached_with_realized_pnl_below_target",
        )

    return (
        "PAPER_OBJECTIVE_INCONCLUSIVE",
        "unable_to_evaluate_goal_outcome",
    )


def evidence_dir(
    *,
    code_sha: str,
    exchange_date: date,
    session_id: str,
    root: Path | None = None,
) -> Path:
    base = root or Path("artifacts/paper-goal-test")
    # Sanitize session_id for filesystem paths.
    safe_session = re.sub(r"[^A-Za-z0-9:._-]+", "_", session_id)[:120]
    path = base / code_sha / exchange_date.isoformat() / safe_session
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(redact_mapping(payload), indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(redact_mapping(payload), default=str) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_checks(db_paths: list[Path]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "databases": {},
    }
    for db in db_paths:
        if not db.exists():
            continue
        entry: dict[str, Any] = {"path": str(db)}
        conn = sqlite3.connect(str(db))
        try:
            entry["integrity_check"] = conn.execute("PRAGMA integrity_check").fetchone()[
                0
            ]
            fk = conn.execute("PRAGMA foreign_key_check").fetchall()
            entry["foreign_key_check_ok"] = len(fk) == 0
            entry["foreign_key_violations"] = len(fk)
            tables = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            if "broker_submission_journal" in tables:
                entry["submission_unknown_count"] = conn.execute(
                    "SELECT COUNT(*) FROM broker_submission_journal "
                    "WHERE status='submission_unknown'"
                ).fetchone()[0]
                entry["duplicate_journal_identity_count"] = conn.execute(
                    """
                    SELECT COUNT(*) FROM (
                      SELECT 1 FROM broker_submission_journal
                      GROUP BY account_id_hash, client_order_id
                      HAVING COUNT(*) > 1
                    )
                    """
                ).fetchone()[0]
        finally:
            conn.close()
        report["databases"][str(db)] = entry
    return report


def build_manifest(
    *,
    code_sha: str,
    branch: str,
    timing: dict[str, Any],
    objective_id: str | None,
    session_id: str,
    paper_account_hash: str | None,
    model_providers: list[str],
    artifact_dir: Path,
) -> dict[str, Any]:
    checksums: dict[str, str] = {}
    for path in sorted(artifact_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            checksums[path.name] = file_sha256(path)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": code_sha,
        "branch": branch,
        "exchange_start_time": timing.get("exchange_now"),
        "exchange_deadline": timing.get("objective_deadline"),
        "runtime_duration_minutes": timing.get("runtime_duration_minutes"),
        "objective_duration_minutes": timing.get("objective_duration_minutes"),
        "objective_id": objective_id,
        "session_id": session_id,
        "paper_account_hash": paper_account_hash,
        "configured_model_providers": model_providers,
        "artifact_checksums": checksums,
    }


def contains_secrets(payload: Any) -> list[str]:
    """Return paths of secret-looking keys found in a JSON-serialisable payload."""
    found: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if _SECRET_KEY_RE.search(str(key)):
                    found.append(path)
                walk(value, path)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                walk(value, f"{prefix}[{idx}]")

    walk(payload, "")
    return found

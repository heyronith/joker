"""Phase 1 storage and event logging tests."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from joker.config.settings import AppSettings
from joker.logging.event_log import EventLogWriter
from joker.runtime.run_manager import RunManager
from joker.storage.database import Database, StorageError, ensure_database
from joker.storage.models import (
    AgentDecisionRecord,
    OrderRecord,
    RunStatus,
    SystemEventRecord,
    TradingDayStateRecord,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Database:
    db_path = tmp_path / "test.db"
    return ensure_database(db_path)


@pytest.fixture
def event_log(tmp_path: Path) -> EventLogWriter:
    return EventLogWriter(
        tmp_path / "logs",
        redact_keys=["OPENAI_API_KEY", "WEBULL_TRADE_PIN"],
    )


def test_db_initializes_locally(tmp_db: Database) -> None:
    assert tmp_db.db_path.exists()


def test_db_write_read_roundtrip(tmp_db: Database) -> None:
    run = tmp_db.create_run(mode="PAPER", trading_day=date(2026, 7, 1))
    saved = tmp_db.save(
        AgentDecisionRecord(
            run_id=run.run_id,
            agent_name="MarketRegimeAgent",
            decision_type="opinion",
            payload={"regime": "trend"},
        )
    )
    decisions = tmp_db.list_by_run(AgentDecisionRecord, run.run_id)
    assert len(decisions) == 1
    assert decisions[0].agent_name == "MarketRegimeAgent"
    assert saved.id is not None


def test_events_append_to_jsonl(event_log: EventLogWriter) -> None:
    entry = event_log.append(
        run_id="run-123",
        mode="PAPER",
        source="test",
        event_type="system.ping",
        payload={"ok": True},
    )
    events = event_log.read_all("run-123")
    assert len(events) == 1
    assert events[0]["run_id"] == "run-123"
    assert events[0]["event_type"] == "system.ping"
    assert entry.timestamp == events[0]["timestamp"]


def test_events_include_required_fields(event_log: EventLogWriter) -> None:
    event_log.append(
        run_id="run-abc",
        mode="SHADOW",
        source="runtime",
        event_type="run.started",
        payload={},
    )
    event = event_log.read_all("run-abc")[0]
    for key in ("timestamp", "run_id", "mode", "source", "event_type"):
        assert key in event


def test_corrupted_db_path_handled_safely(tmp_path: Path) -> None:
    bad_path = tmp_path / "not-a-dir" / "db.sqlite"
    bad_path.parent.write_text("blocker")
    with pytest.raises(StorageError):
        ensure_database(bad_path)


def test_no_event_log_contains_secret_values(event_log: EventLogWriter) -> None:
    event_log.append(
        run_id="run-secret",
        mode="PAPER",
        source="test",
        event_type="config.loaded",
        payload={"OPENAI_API_KEY": "sk-real-secret-value-here"},
    )
    raw = (event_log.log_dir / "run-secret.jsonl").read_text()
    assert "sk-real-secret-value-here" not in raw
    assert "[REDACTED]" in raw

    event_log.append(
        run_id="run-secret3",
        mode="PAPER",
        source="test",
        event_type="error",
        payload={"message": "failed with sk-abcdefghijklmnop"},
    )
    raw3 = (event_log.log_dir / "run-secret3.jsonl").read_text()
    assert "sk-abcdefghijklmnop" not in raw3
    assert "[REDACTED]" in raw3


def test_run_manager_creates_run_and_state(
    tmp_db: Database,
    event_log: EventLogWriter,
) -> None:
    app = AppSettings.model_validate({"mode": "PAPER"})
    manager = RunManager(tmp_db, event_log, app)
    run_id = manager.start_run(trading_day=date(2026, 7, 1))
    assert tmp_db.get_run(run_id) is not None
    state = tmp_db.get_trading_day_state(date(2026, 7, 1))
    assert state is not None
    assert state.run_id == run_id
    events = event_log.read_all(run_id)
    assert any(e["event_type"] == "run.started" for e in events)

    manager.end_run(run_id, status=RunStatus.COMPLETED)
    run = tmp_db.get_run(run_id)
    assert run is not None
    assert run.status == RunStatus.COMPLETED.value

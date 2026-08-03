"""Submission journal durability and uniqueness."""

from __future__ import annotations

import pytest

from joker.persistence.broker_submission_journal import (
    BrokerSubmissionJournal,
    BrokerSubmissionRecord,
    DuplicateSubmissionError,
    SyncBrokerSubmissionJournal,
    apply_broker_submission_journal_migration,
    payload_hash,
)
from joker.persistence.migrations import apply_task1_migrations


def test_unique_constraint_rejects_duplicate(tmp_path) -> None:
    journal = SyncBrokerSubmissionJournal(tmp_path / "j.db")
    rec = BrokerSubmissionRecord(
        client_order_id="c" * 32,
        broker_mode="webull_live",
        account_id_hash="abc123",
        status="prepared",
        payload_hash=payload_hash({"x": 1}),
    )
    journal.prepare(rec)
    with pytest.raises(DuplicateSubmissionError):
        journal.prepare(rec)


def test_status_transitions_preserve_created_at(tmp_path) -> None:
    journal = SyncBrokerSubmissionJournal(tmp_path / "j.db")
    journal.prepare(
        BrokerSubmissionRecord(
            client_order_id="d" * 32,
            broker_mode="webull_live",
            account_id_hash="hash1",
            status="prepared",
        )
    )
    first = journal.get("hash1", "d" * 32)
    assert first is not None
    journal.transition(
        account_id_hash="hash1",
        client_order_id="d" * 32,
        status="submission_started",
    )
    journal.transition(
        account_id_hash="hash1",
        client_order_id="d" * 32,
        status="submission_unknown",
        last_error_code="timeout",
    )
    stored = journal.get("hash1", "d" * 32)
    assert stored is not None
    assert stored.status == "submission_unknown"
    assert stored.created_at == first.created_at
    assert stored.last_error_code == "timeout"


@pytest.mark.asyncio
async def test_async_journal_and_migration(tmp_path) -> None:
    db = tmp_path / "task1.db"
    apply_task1_migrations(db)
    apply_broker_submission_journal_migration(db)
    journal = BrokerSubmissionJournal(db)
    await journal.initialize()
    await journal.prepare(
        BrokerSubmissionRecord(
            client_order_id="e" * 32,
            broker_mode="webull_live",
            account_id_hash="h2",
            status="prepared",
        )
    )
    row = await journal.get("h2", "e" * 32)
    assert row is not None
    await journal.transition(
        account_id_hash="h2",
        client_order_id="e" * 32,
        status="accepted",
        broker_order_id="WB-1",
    )
    row2 = await journal.get("h2", "e" * 32)
    assert row2 is not None
    assert row2.broker_order_id == "WB-1"

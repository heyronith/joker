"""Anchored authoritative event-horizon verification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from joker.evolution.episode_metadata import verify_event_horizon
from joker.evolution.event_horizon import Task1EventHorizon, Task1HorizonEvent


def _ts(minutes: int = 0) -> datetime:
    return datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc) + timedelta(minutes=minutes)


def _horizon(
    *,
    entry_id=None,
    terminal_id=None,
    sequences=(1, 2),
    start_offset=0,
    end_offset=5,
    duplicate=False,
) -> Task1EventHorizon:
    e1 = entry_id or uuid4()
    e2 = terminal_id or uuid4()
    events = [
        Task1HorizonEvent(
            event_id=e1,
            event_type="MARKET_SNAPSHOT_CREATED",
            exchange_timestamp=_ts(start_offset),
            sequence=sequences[0],
        ),
        Task1HorizonEvent(
            event_id=e2,
            event_type="POSITION_CLOSED",
            exchange_timestamp=_ts(end_offset),
            sequence=sequences[1],
        ),
    ]
    if duplicate:
        events.append(
            Task1HorizonEvent(
                event_id=e1,
                event_type="MARKET_SNAPSHOT_CREATED",
                exchange_timestamp=_ts(end_offset),
                sequence=sequences[1] + 1,
            )
        )
    ids = tuple(ev.event_id for ev in events)
    return Task1EventHorizon(
        session_id="s",
        events=tuple(events),
        market_event_ids=ids,
    )


def test_verify_horizon_missing_entry_id_fails() -> None:
    """None entry anchor fails closed — never skipped."""
    terminal = uuid4()
    horizon = _horizon(entry_id=uuid4(), terminal_id=terminal)
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=None,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_entry_missing" in findings
    assert "historical_ev_eligible=false" in findings
    assert "promotion_eligible=false" in findings
    assert "truth_degraded=true" in findings


def test_verify_horizon_missing_terminal_id_fails() -> None:
    """None terminal anchor fails closed — never skipped."""
    entry = uuid4()
    horizon = _horizon(entry_id=entry, terminal_id=uuid4())
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=None,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_terminal_missing" in findings
    assert "historical_ev_eligible=false" in findings
    assert "promotion_eligible=false" in findings
    assert "truth_degraded=true" in findings


def test_horizon_missing_entry_event_fails() -> None:
    entry = uuid4()
    terminal = uuid4()
    horizon = _horizon(entry_id=uuid4(), terminal_id=terminal)
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_entry_missing" in findings
    assert "historical_ev_eligible=false" in findings


def test_horizon_missing_terminal_event_fails() -> None:
    entry = uuid4()
    terminal = uuid4()
    horizon = _horizon(entry_id=entry, terminal_id=uuid4())
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_terminal_missing" in findings


def test_horizon_sequence_gap_fails_under_contiguous_policy() -> None:
    entry, terminal = uuid4(), uuid4()
    horizon = _horizon(
        entry_id=entry, terminal_id=terminal, sequences=(1, 3)
    )
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_sequence_gap" in findings


def test_horizon_duplicate_event_fails() -> None:
    entry, terminal = uuid4(), uuid4()
    horizon = _horizon(entry_id=entry, terminal_id=terminal, duplicate=True)
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_duplicate_event" in findings


def test_horizon_start_after_entry_fails() -> None:
    entry, terminal = uuid4(), uuid4()
    horizon = _horizon(
        entry_id=entry, terminal_id=terminal, start_offset=2, end_offset=5
    )
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_time_coverage_incomplete" in findings


def test_horizon_end_before_terminal_fails() -> None:
    entry, terminal = uuid4(), uuid4()
    horizon = _horizon(
        entry_id=entry, terminal_id=terminal, start_offset=0, end_offset=3
    )
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is False
    assert "authoritative_horizon_time_coverage_incomplete" in findings


def test_complete_anchored_horizon_passes() -> None:
    entry, terminal = uuid4(), uuid4()
    horizon = _horizon(entry_id=entry, terminal_id=terminal)
    ok, findings = verify_event_horizon(
        horizon,
        entry_ts=_ts(0),
        terminal_ts=_ts(5),
        entry_event_id=entry,
        terminal_event_id=terminal,
        sequence_policy="globally_contiguous",
    )
    assert ok is True
    assert findings == ()

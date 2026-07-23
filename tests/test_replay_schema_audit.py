"""Replay schema audit tests."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from joker.data.replay_loader import (
    ReplayLoadError,
    deserialize_session,
    event_type_name,
    load_replay_file,
    serialize_session,
)
from joker.data.replay_provider import ReplayMarketDataProvider
from joker.schemas.domain import Candle
from joker.schemas.replay import (
    OptionQuoteEvent,
    ReplaySession,
    SpyCandleEvent,
    SpyQuoteEvent,
)


def test_load_preserves_subclass_identity(synthetic_replay_path: Path) -> None:
    session = load_replay_file(synthetic_replay_path)
    types = {type(e).__name__ for e in session.events}
    assert "SpyQuoteEvent" in types
    assert "SpyCandleEvent" in types
    assert "OptionQuoteEvent" in types


def test_session_roundtrip_preserves_event_types(synthetic_replay_path: Path) -> None:
    session = load_replay_file(synthetic_replay_path)
    restored = deserialize_session(serialize_session(session))
    assert len(restored.events) == len(session.events)
    for orig, back in zip(session.events, restored.events):
        assert type(orig) is type(back)
        assert orig.event_type == back.event_type


def test_provider_works_after_roundtrip(synthetic_replay_path: Path) -> None:
    session = load_replay_file(synthetic_replay_path)
    restored = deserialize_session(serialize_session(session))
    provider = ReplayMarketDataProvider(restored)
    count = sum(1 for _ in provider.stream_events())
    assert count == len(restored.events)
    assert provider.get_latest_snapshot() is not None


def test_invalid_event_type_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"event_type":"unknown_event","timestamp":"2026-07-01T14:00:00+00:00","symbol":"SPY","source":"test"}'
    )
    with pytest.raises(ReplayLoadError, match="validation failed"):
        load_replay_file(path)


def test_event_type_name_helper() -> None:
    ev = SpyQuoteEvent(
        timestamp=datetime(2026, 7, 1, 14, 0, tzinfo=timezone.utc),
        source="test",
        price=550.0,
    )
    assert event_type_name(ev) == "SpyQuoteEvent"

"""Load and validate replay JSONL files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from joker.schemas.replay import (
    MarketEvent,
    OptionChainSnapshot,
    OptionQuoteEvent,
    ReplayMetadata,
    ReplaySession,
    SpyCandleEvent,
    SpyQuoteEvent,
)

EVENT_ADAPTER: TypeAdapter[MarketEvent] = TypeAdapter(MarketEvent)
SESSION_ADAPTER: TypeAdapter[ReplaySession] = TypeAdapter(ReplaySession)


class ReplayLoadError(Exception):
    pass


def parse_event_line(line: str, line_no: int) -> MarketEvent:
    line = line.strip()
    if not line or line.startswith("#"):
        raise ReplayLoadError(f"Line {line_no}: empty or comment")
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ReplayLoadError(f"Line {line_no}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ReplayLoadError(f"Line {line_no}: expected JSON object")
    try:
        return EVENT_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ReplayLoadError(f"Line {line_no}: schema validation failed: {exc}") from exc


def load_replay_file(path: Path) -> ReplaySession:
    path = Path(path)
    if not path.exists():
        raise ReplayLoadError(f"Replay file not found: {path}")

    events: list[MarketEvent] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                events.append(parse_event_line(line, line_no))
    except ReplayLoadError:
        raise
    except OSError as exc:
        raise ReplayLoadError(f"Failed to read replay file: {exc}") from exc

    if not events:
        raise ReplayLoadError("Replay file contains no events")

    events.sort(key=lambda e: (e.timestamp, e.event_id))
    trading_day = events[0].timestamp.date()
    is_synthetic = any(
        e.source in ("synthetic", "synthetic_replay", "test_fixture") for e in events
    ) or "synthetic" in path.name.lower()

    metadata = ReplayMetadata(
        name=path.stem,
        trading_day=trading_day,
        is_synthetic=is_synthetic,
        description=f"Loaded from {path.name}",
        event_count=len(events),
        start_time=events[0].timestamp,
        end_time=events[-1].timestamp,
        source_file=str(path),
    )
    return ReplaySession(metadata=metadata, events=events)


def serialize_session(session: ReplaySession) -> str:
    return SESSION_ADAPTER.dump_json(session).decode()


def deserialize_session(data: str | bytes) -> ReplaySession:
    try:
        return SESSION_ADAPTER.validate_json(data)
    except ValidationError as exc:
        raise ReplayLoadError(f"Replay session deserialization failed: {exc}") from exc


def event_type_name(event: MarketEvent) -> str:
    return type(event).__name__


def inspect_replay(path: Path) -> dict:
    session = load_replay_file(path)
    counts: dict[str, int] = {}
    for event in session.events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1
    return {
        "name": session.metadata.name,
        "trading_day": session.metadata.trading_day.isoformat(),
        "is_synthetic": session.metadata.is_synthetic,
        "event_count": session.metadata.event_count,
        "start_time": session.metadata.start_time.isoformat() if session.metadata.start_time else None,
        "end_time": session.metadata.end_time.isoformat() if session.metadata.end_time else None,
        "event_types": counts,
    }

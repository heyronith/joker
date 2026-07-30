"""Sanitised operator events for objective lifecycle (UI prep)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from threading import Lock
from typing import Any
from uuid import UUID


class ObjectiveOperatorEventType(StrEnum):
    CREATED = "objective.created"
    CONFIRMED = "objective.confirmed"
    RECOMPUTED = "objective.recomputed"
    FEASIBILITY_ASSESSED = "objective.feasibility_assessed"
    STRATEGY_SCORED = "objective.strategy_scored"
    SIZING_DECIDED = "objective.sizing_decided"
    CAPITAL_RESERVED = "objective.capital_reserved"
    CAPITAL_RELEASED = "objective.capital_released"
    PROGRESS_CHANGED = "objective.progress_changed"
    TARGET_REACHED = "objective.target_reached"
    INFEASIBLE = "objective.infeasible"
    DEADLINE_REACHED = "objective.deadline_reached"
    PAUSED = "objective.paused"
    RESUMED = "objective.resumed"


@dataclass(frozen=True)
class ObjectiveOperatorEvent:
    event_type: ObjectiveOperatorEventType
    objective_id: str
    session_id: str
    timestamp: datetime
    reason_codes: tuple[str, ...] = ()
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    linked_ids: dict[str, str] = field(default_factory=dict)

    def sanitised_payload(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "objective_id": self.objective_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "reason_codes": list(self.reason_codes),
            "before": dict(self.before),
            "after": dict(self.after),
            "linked_ids": dict(self.linked_ids),
        }


def make_objective_event(
    event_type: ObjectiveOperatorEventType,
    *,
    objective_id: UUID | str,
    session_id: str,
    reason_codes: tuple[str, ...] = (),
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    linked_ids: dict[str, str] | None = None,
    timestamp: datetime | None = None,
) -> ObjectiveOperatorEvent:
    ts = timestamp or datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ObjectiveOperatorEvent(
        event_type=event_type,
        objective_id=str(objective_id),
        session_id=session_id,
        timestamp=ts,
        reason_codes=reason_codes,
        before=dict(before or {}),
        after=dict(after or {}),
        linked_ids=dict(linked_ids or {}),
    )


class BoundedOperatorEventProjection:
    """Non-blocking ring buffer for later Textual UI subscription.

    Slow consumers must never block market ingest or the cognitive graph:
    publish drops oldest events when full and never waits on subscribers.
    """

    def __init__(self, *, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._events: deque[ObjectiveOperatorEvent] = deque(maxlen=capacity)
        self._lock = Lock()
        self._dropped = 0

    def publish(self, event: ObjectiveOperatorEvent) -> None:
        with self._lock:
            if len(self._events) >= self._capacity:
                self._dropped += 1
            self._events.append(event)

    def snapshot(self, *, limit: int | None = None) -> list[ObjectiveOperatorEvent]:
        with self._lock:
            items = list(self._events)
        if limit is not None:
            return items[-max(0, limit) :]
        return items

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

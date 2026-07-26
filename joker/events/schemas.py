"""Typed domain event schemas for Joker's in-process event bus."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(StrEnum):
    """Canonical event types for market and execution lifecycle."""

    SESSION_STARTED = "session_started"
    QUOTE_RECEIVED = "quote_received"
    TRADE_RECEIVED = "trade_received"
    BAR_CLOSED = "bar_closed"
    MARKET_SNAPSHOT_CREATED = "market_snapshot_created"
    OPTION_SURFACE_CREATED = "option_surface_created"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CHANGED = "position_changed"
    POSITION_CLOSED = "position_closed"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    SESSION_ENDING = "session_ending"
    SESSION_ENDED = "session_ended"
    # Task 2 cognitive lifecycle
    COGNITIVE_CYCLE_STARTED = "cognitive_cycle_started"
    AGENT_EVIDENCE_CREATED = "agent_evidence_created"
    WORLD_MODEL_CREATED = "world_model_created"
    HYPOTHESIS_CREATED = "hypothesis_created"
    STRATEGY_CREATED = "strategy_created"
    DEBATE_REVIEW_CREATED = "debate_review_created"
    META_DECISION_CREATED = "meta_decision_created"
    MORE_EVIDENCE_REQUESTED = "more_evidence_requested"
    EXECUTION_PROPOSAL_CREATED = "execution_proposal_created"
    COGNITIVE_DECISION_STALE = "cognitive_decision_stale"
    COGNITIVE_RUNTIME_DEGRADED = "cognitive_runtime_degraded"
    POSITION_THESIS_UPDATED = "position_thesis_updated"
    POSITION_ACTION_PROPOSED = "position_action_proposed"
    COGNITIVE_CYCLE_COMPLETED = "cognitive_cycle_completed"


class DomainEvent(BaseModel):
    """Immutable domain event envelope with flexible validated payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID = Field(default_factory=uuid4)
    event_type: EventType
    schema_version: str = "1"
    correlation_id: UUID = Field(default_factory=uuid4)
    causation_id: UUID | None = None
    session_id: str
    exchange_timestamp: datetime
    created_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("exchange_timestamp", "created_timestamp")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def _coerce_payload(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("payload must be a dict")
        return value

    @field_validator("session_id", "source")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("must be a non-empty string")
        return value


def make_event(
    event_type: EventType,
    *,
    session_id: str,
    source: str,
    exchange_timestamp: datetime,
    payload: dict[str, Any] | None = None,
    correlation_id: UUID | None = None,
    causation_id: UUID | None = None,
    event_id: UUID | None = None,
    created_timestamp: datetime | None = None,
    schema_version: str = "1",
) -> DomainEvent:
    """Factory for DomainEvent with sensible defaults for IDs and created time."""
    kwargs: dict[str, Any] = {
        "event_type": event_type,
        "schema_version": schema_version,
        "correlation_id": correlation_id or uuid4(),
        "causation_id": causation_id,
        "session_id": session_id,
        "exchange_timestamp": exchange_timestamp,
        "source": source,
        "payload": payload or {},
    }
    if event_id is not None:
        kwargs["event_id"] = event_id
    if created_timestamp is not None:
        kwargs["created_timestamp"] = created_timestamp
    return DomainEvent(**kwargs)

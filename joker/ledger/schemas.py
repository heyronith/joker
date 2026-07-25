"""Append-only execution ledger event schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

Side = Literal["buy", "sell"]


class LedgerEventType(StrEnum):
    """Canonical ledger event types for order and position truth."""

    ORDER_SUBMISSION_REQUESTED = "order_submission_requested"
    BROKER_ORDER_ACCEPTED = "broker_order_accepted"
    PARTIAL_FILL = "partial_fill"
    FINAL_FILL = "final_fill"
    CANCELLATION = "cancellation"
    REJECTION = "rejection"
    POSITION_OPENED = "position_opened"
    POSITION_RESIZED = "position_resized"
    POSITION_CLOSED = "position_closed"
    FEE_RECORDED = "fee_recorded"
    RECONCILIATION_CORRECTION = "reconciliation_correction"


class LedgerEvent(BaseModel):
    """Immutable append-only ledger fact. Never mutate after persistence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ledger_event_id: UUID = Field(default_factory=uuid4)
    broker_account_id: str
    client_order_id: str
    broker_order_id: str | None = None
    contract_id: str
    side: Side
    quantity: Decimal
    price: Decimal | None = None
    exchange_timestamp: datetime
    source_event_id: UUID | None = None
    idempotency_key: str
    event_type: LedgerEventType
    fees: Decimal | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_id: str
    position_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("exchange_timestamp", "created_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @field_validator("quantity")
    @classmethod
    def _require_non_negative_qty(cls, value: Decimal) -> Decimal:
        if value < 0:
            raise ValueError("quantity must be >= 0")
        return value

    @field_validator("price", "fees")
    @classmethod
    def _require_non_negative_money(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and value < 0:
            raise ValueError("price/fees must be >= 0 when present")
        return value

    @field_validator(
        "broker_account_id",
        "client_order_id",
        "contract_id",
        "idempotency_key",
        "session_id",
    )
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        if not value or not str(value).strip():
            raise ValueError("must be a non-empty string")
        return value


def make_ledger_event(
    event_type: LedgerEventType,
    *,
    broker_account_id: str,
    client_order_id: str,
    contract_id: str,
    side: Side,
    quantity: Decimal,
    exchange_timestamp: datetime,
    idempotency_key: str,
    session_id: str,
    broker_order_id: str | None = None,
    price: Decimal | None = None,
    source_event_id: UUID | None = None,
    fees: Decimal | None = None,
    metadata: dict[str, Any] | None = None,
    position_id: str | None = None,
    ledger_event_id: UUID | None = None,
    created_at: datetime | None = None,
) -> LedgerEvent:
    """Factory for LedgerEvent with generated IDs when omitted."""
    kwargs: dict[str, Any] = {
        "event_type": event_type,
        "broker_account_id": broker_account_id,
        "client_order_id": client_order_id,
        "broker_order_id": broker_order_id,
        "contract_id": contract_id,
        "side": side,
        "quantity": quantity,
        "price": price,
        "exchange_timestamp": exchange_timestamp,
        "source_event_id": source_event_id,
        "idempotency_key": idempotency_key,
        "fees": fees,
        "metadata": metadata or {},
        "session_id": session_id,
        "position_id": position_id,
    }
    if ledger_event_id is not None:
        kwargs["ledger_event_id"] = ledger_event_id
    if created_at is not None:
        kwargs["created_at"] = created_at
    return LedgerEvent(**kwargs)

"""Immutable market observations — raw exchange/provider facts only."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _reject_naive(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("Naive datetimes are not allowed in market observations")
    return ts


class QuoteObservation(BaseModel):
    """Top-of-book quote observation for an underlying or listed instrument."""

    model_config = ConfigDict(frozen=True)

    observation_id: UUID = Field(default_factory=uuid4)
    symbol: str
    source_timestamp: datetime
    received_timestamp: datetime
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    last_size: int | None = None
    cumulative_volume: int | None = None
    source: str

    @field_validator("source_timestamp", "received_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


class TradeObservation(BaseModel):
    """Print / last-sale trade observation."""

    model_config = ConfigDict(frozen=True)

    observation_id: UUID = Field(default_factory=uuid4)
    symbol: str
    source_timestamp: datetime
    received_timestamp: datetime
    price: Decimal
    size: int
    cumulative_volume: int | None = None
    source: str

    @field_validator("source_timestamp", "received_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)

    @field_validator("size")
    @classmethod
    def _non_negative_size(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Trade size cannot be negative")
        return value


class UnderlyingObservation(BaseModel):
    """Underlying instrument observation (price + optional top-of-book)."""

    model_config = ConfigDict(frozen=True)

    observation_id: UUID = Field(default_factory=uuid4)
    symbol: str
    source_timestamp: datetime
    received_timestamp: datetime
    last: Decimal | None = None
    bid: Decimal | None = None
    ask: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    cumulative_volume: int | None = None
    source: str

    @field_validator("source_timestamp", "received_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)


class OptionQuoteObservation(BaseModel):
    """Single-contract option quote observation."""

    model_config = ConfigDict(frozen=True)

    observation_id: UUID = Field(default_factory=uuid4)
    underlying_symbol: str
    contract_symbol: str
    expiry: date
    strike: Decimal
    option_type: Literal["call", "put"]
    source_timestamp: datetime
    received_timestamp: datetime
    bid: Decimal | None = None
    ask: Decimal | None = None
    last: Decimal | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    volume: int | None = None
    open_interest: int | None = None
    implied_volatility: Decimal | None = None
    delta: Decimal | None = None
    gamma: Decimal | None = None
    theta: Decimal | None = None
    vega: Decimal | None = None
    source: str

    @field_validator("source_timestamp", "received_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return _reject_naive(value)

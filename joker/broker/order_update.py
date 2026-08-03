"""Typed broker order update for partial/final fill truth."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

BrokerOrderStatus = Literal[
    "pending", "open", "partially_filled", "filled", "cancelled", "rejected"
]


@dataclass(frozen=True)
class BrokerOrderUpdate:
    client_order_id: str
    broker_order_id: str
    status: BrokerOrderStatus
    quantity: int
    cumulative_filled_quantity: int
    remaining_quantity: int
    average_fill_price: Decimal | None
    last_fill_quantity: int | None = None
    last_fill_price: Decimal | None = None
    limit_price: Decimal | None = None
    side: Literal["buy", "sell"] | None = None
    position_intent: str | None = None

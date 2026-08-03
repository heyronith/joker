"""Typed broker account / preview truth — never fabricate missing financial values."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from joker.schemas.domain import BrokerOrder, Position


def hash_account_id(account_id: str) -> str:
    """Stable non-reversible account identifier for logs and journal keys."""
    digest = hashlib.sha256(account_id.encode("utf-8")).hexdigest()
    return digest[:16]


def mask_account_id(account_id: str) -> str:
    """Masked suffix for operator display (never full account id in logs)."""
    value = (account_id or "").strip()
    if len(value) <= 4:
        return "****"
    return f"…{value[-4:]}"


def decimal_or_none(raw: Any) -> Decimal | None:
    if raw is None or raw == "":
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, TypeError, ValueError):
        return None


@dataclass(frozen=True)
class BrokerAccountTruth:
    account_id_hash: str
    cash_usd: Decimal | None
    buying_power_usd: Decimal | None
    net_liquidation_value_usd: Decimal | None
    session_pnl_usd: Decimal | None
    session_pnl_available: bool
    positions: tuple[Position, ...]
    working_orders: tuple[BrokerOrder, ...]
    captured_at: datetime


@dataclass(frozen=True)
class OrderPreviewTruth:
    accepted: bool
    estimated_cost_usd: Decimal | None
    estimated_fees_usd: Decimal | None
    buying_power_effect_usd: Decimal | None
    rejection_code: str | None
    rejection_message: str | None
    raw_response_hash: str


def parse_balance_truth(payload: dict[str, Any]) -> dict[str, Decimal | None]:
    """Extract cash / BP / NLV without fabricating zeros for missing fields."""

    def _first(*keys: str) -> Decimal | None:
        for key in keys:
            if key in payload:
                value = decimal_or_none(payload.get(key))
                if value is not None:
                    return value
        nested = payload.get("account")
        if isinstance(nested, dict):
            for key in keys:
                if key in nested:
                    value = decimal_or_none(nested.get(key))
                    if value is not None:
                        return value
        assets = payload.get("account_currency_assets")
        if isinstance(assets, list):
            for row in assets:
                if not isinstance(row, dict):
                    continue
                for key in keys:
                    if key in row:
                        value = decimal_or_none(row.get(key))
                        if value is not None:
                            return value
        return None

    return {
        "cash_usd": _first(
            "total_cash_balance", "total_cash", "cash_balance", "cash", "available_cash"
        ),
        "buying_power_usd": _first("buying_power", "bp", "day_buying_power"),
        "net_liquidation_value_usd": _first(
            "total_net_liquidation_value",
            "net_liquidation",
            "totalCashValue",
            "net_liquidation_value",
        ),
    }

"""Deterministic options position-intent resolution for broker payloads."""

from __future__ import annotations

from typing import Literal

from joker.broker.interface import BrokerError
from joker.schemas.domain import Position

PositionIntent = Literal[
    "BUY_TO_OPEN",
    "BUY_TO_CLOSE",
    "SELL_TO_OPEN",
    "SELL_TO_CLOSE",
]


def resolve_position_intent(
    *,
    action: Literal["entry", "exit", "replace"] | str,
    side: Literal["buy", "sell"],
    contract_id: str,
    open_positions: tuple[Position, ...] | list[Position] = (),
    allow_short_open: bool = False,
) -> PositionIntent:
    """Resolve explicit Webull position intent from action + side + broker truth.

    Long-option system:
      entry + buy  → BUY_TO_OPEN
      exit  + sell → SELL_TO_CLOSE (only when a matching long exists)
    """
    action_norm = str(action).strip().lower()
    side_norm = str(side).strip().lower()
    matching_long = _matching_long_qty(contract_id, open_positions)

    if action_norm in {"entry", "open"} and side_norm == "buy":
        return "BUY_TO_OPEN"
    if action_norm in {"exit", "close", "reduce"} and side_norm == "sell":
        if matching_long <= 0:
            raise BrokerError(
                "position_intent rejected: SELL_TO_CLOSE requires a matching long position"
            )
        return "SELL_TO_CLOSE"
    if action_norm in {"entry", "open"} and side_norm == "sell":
        if not allow_short_open:
            raise BrokerError(
                "position_intent rejected: SELL_TO_OPEN is not enabled for this system"
            )
        return "SELL_TO_OPEN"
    if action_norm in {"exit", "close"} and side_norm == "buy":
        return "BUY_TO_CLOSE"

    # Side-only inference is forbidden for closes.
    if side_norm == "sell" and matching_long > 0 and action_norm in {"replace", ""}:
        raise BrokerError(
            "position_intent rejected: cannot infer SELL_TO_CLOSE from side alone"
        )
    raise BrokerError(
        f"position_intent rejected: contradictory action={action!r} side={side!r}"
    )


def validate_position_intent(
    intent: PositionIntent | None,
    *,
    side: Literal["buy", "sell"],
    open_positions: tuple[Position, ...] | list[Position] = (),
    contract_id: str | None = None,
) -> PositionIntent:
    if intent is None:
        raise BrokerError("position_intent is required for live option orders")
    side_norm = str(side).strip().lower()
    if intent.startswith("BUY") and side_norm != "buy":
        raise BrokerError("position_intent BUY_* requires side=buy")
    if intent.startswith("SELL") and side_norm != "sell":
        raise BrokerError("position_intent SELL_* requires side=sell")
    if intent == "SELL_TO_CLOSE":
        if not contract_id or _matching_long_qty(contract_id, open_positions) <= 0:
            raise BrokerError(
                "position_intent SELL_TO_CLOSE requires a matching long position"
            )
    return intent


def _matching_long_qty(
    contract_id: str, positions: tuple[Position, ...] | list[Position]
) -> int:
    total = 0
    for pos in positions:
        if not getattr(pos, "is_open", True):
            continue
        cid = _position_contract_id(pos)
        if cid != contract_id:
            continue
        qty = int(getattr(pos, "quantity", 0) or 0)
        if qty > 0:
            total += qty
    return total


def _position_contract_id(pos: Position) -> str:
    contract = getattr(pos, "contract", None)
    if contract is None:
        return ""
    symbol = getattr(contract, "symbol", "SPY")
    expiration = getattr(contract, "expiration", None)
    strike = getattr(contract, "strike", None)
    option_type = getattr(contract, "option_type", "call")
    exp = expiration.isoformat() if hasattr(expiration, "isoformat") else str(expiration)
    return f"{symbol}:{exp}:{strike}:{option_type}"

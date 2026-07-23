"""Build enriched DecisionAgent context from live session state."""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from joker.agents.session_memory import SessionMicroMemory
from joker.schemas.domain import TechnicalFeatures

_ET = ZoneInfo("America/New_York")
_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)


def session_clock(now: datetime | None = None) -> dict[str, Any]:
    """RTH clock helpers for 0DTE (America/New_York)."""
    ref = now or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    et = ref.astimezone(_ET)
    open_dt = datetime.combine(et.date(), _RTH_OPEN, tzinfo=_ET)
    close_dt = datetime.combine(et.date(), _RTH_CLOSE, tzinfo=_ET)
    minutes_from_open = (et - open_dt).total_seconds() / 60.0
    minutes_to_close = (close_dt - et).total_seconds() / 60.0
    in_rth = open_dt <= et <= close_dt
    return {
        "et_time": et.strftime("%H:%M:%S"),
        "in_rth": in_rth,
        "minutes_from_open": round(minutes_from_open, 1),
        "minutes_to_close": round(minutes_to_close, 1),
        "day_part": _day_part(minutes_from_open, minutes_to_close, in_rth),
    }


def _day_part(minutes_from_open: float, minutes_to_close: float, in_rth: bool) -> str:
    if not in_rth:
        return "outside_rth"
    if minutes_from_open < 30:
        return "open_drive"
    if minutes_to_close < 45:
        return "power_hour"
    if minutes_from_open < 120:
        return "morning"
    if minutes_to_close < 150:
        return "afternoon"
    return "midday"


def enrich_option_context(
    option_context: dict[str, Any],
    memory: SessionMicroMemory | None,
) -> dict[str, Any]:
    """Add mid deltas vs prior decision tick."""
    out = dict(option_context or {})
    if memory is None:
        return out
    prior = memory.last_option_mids
    for key in ("atm_call", "atm_put"):
        block = out.get(key)
        if not isinstance(block, dict):
            continue
        mid = block.get("mid")
        prev = prior.get(key)
        if mid is not None and prev is not None and prev > 0:
            block = dict(block)
            block["mid_change_pct"] = round(((float(mid) - float(prev)) / float(prev)) * 100.0, 3)
            out[key] = block
    return out


def features_for_prompt(features: TechnicalFeatures) -> dict[str, Any]:
    """Serialize features including richer fields for the agent."""
    return features.model_dump(mode="json")


def build_agent_market_context(
    *,
    features: TechnicalFeatures,
    spy_price: float | None,
    option_context: dict[str, Any],
    memory: SessionMicroMemory | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    clock = session_clock(now or features.as_of)
    opts = enrich_option_context(option_context, memory)
    return {
        "spy_price": spy_price,
        "session_clock": clock,
        "features": features_for_prompt(features),
        "option_context": opts,
        "session_memory": memory.prompt_dict() if memory is not None else {},
    }

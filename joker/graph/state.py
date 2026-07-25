"""Task 1 LangGraph state contract (no cognitive agents)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict


class JokerGraphState(TypedDict, total=False):
    """Minimal graph state for market/execution grounding."""

    run_id: str
    session_id: str
    exchange_time: datetime
    market_snapshot_id: str
    feature_snapshot_id: str
    option_surface_id: str
    data_quality_id: str
    active_order_id: str
    active_position_id: str
    pending_event_ids: list[str]
    errors: list[dict[str, Any]]

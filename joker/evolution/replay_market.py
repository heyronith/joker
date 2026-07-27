"""Frozen Task 1 truth loaded for isolated cognitive replay."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from joker.evolution.replay_truth import ReplayMarketFrame


class ReplayPositionSeed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str
    quantity: Decimal
    avg_price: Decimal
    side: str = "long"
    position_lifecycle_id: str | None = None


class ReplayWorkingOrderSeed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    client_order_id: str
    contract_id: str
    side: Literal["buy", "sell"]
    quantity: Decimal
    filled_qty: Decimal = Decimal("0")
    limit_price: Decimal | None = None
    status: str = "accepted"
    parent_client_order_id: str | None = None
    position_lifecycle_id: str | None = None


class ReplayEpisodeTruth(BaseModel):
    """Authoritative frozen market sequence for one replay unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: UUID
    session_id: str = ""
    trading_date: date | None = None
    initial_snapshot_id: UUID
    terminal_snapshot_id: UUID | None = None
    snapshot_sequence: tuple[UUID, ...] = ()
    option_surface_sequence: tuple[UUID, ...] = ()
    data_quality_sequence: tuple[UUID, ...] = ()
    market_event_ids: tuple[UUID, ...] = ()
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    entry_decision_timestamp: datetime | None = None
    terminal_event_timestamp: datetime | None = None
    position_lifecycle_id: str | None = None
    starting_cash: Decimal
    starting_positions: tuple[ReplayPositionSeed, ...] = ()
    starting_working_orders: tuple[ReplayWorkingOrderSeed, ...] = ()
    market_calendar_version: str = "us_equity_rth_v1"
    fill_model_version: str = "replay_fill_v1"
    random_seed: int = 42
    # Deprecated — production path uses frames; kept for legacy unit tests only.
    contract_quotes: dict[str, dict[str, Any]] = Field(default_factory=dict)
    frames: tuple[Any, ...] = ()

    def frame_quotes(self, index: int = 0) -> dict[str, dict[str, str]]:
        if self.frames:
            frame = self.frames[index]
            return {
                c.contract_id: {
                    "bid": str(c.bid),
                    "ask": str(c.ask),
                    "mid": str((c.bid + c.ask) / Decimal("2")),
                }
                for c in frame.contracts
            }
        return {k: dict(v) for k, v in self.contract_quotes.items()}

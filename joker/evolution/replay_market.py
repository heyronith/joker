"""Frozen Task 1 truth loaded for isolated cognitive replay."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any
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


class ReplayEpisodeTruth(BaseModel):
    """Authoritative frozen market sequence for one replay unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: UUID
    initial_snapshot_id: UUID
    terminal_snapshot_id: UUID | None = None
    snapshot_sequence: tuple[UUID, ...] = ()
    option_surface_sequence: tuple[UUID, ...] = ()
    data_quality_sequence: tuple[UUID, ...] = ()
    market_event_ids: tuple[UUID, ...] = ()
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    starting_cash: Decimal = Decimal("100000")
    starting_positions: tuple[ReplayPositionSeed, ...] = ()
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

"""Frozen Task 1 truth loaded for isolated cognitive replay."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


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
    contract_quotes: dict[str, dict[str, Any]] = Field(default_factory=dict)


def build_truth_from_episode(
    episode: Any,
    *,
    starting_cash: Decimal = Decimal("100000"),
    random_seed: int = 42,
    contract_quotes: dict[str, dict[str, Any]] | None = None,
) -> ReplayEpisodeTruth:
    """Derive frozen truth from an authoritative TradingEpisode (no fabrication)."""
    snaps: list[UUID] = [episode.initial_snapshot_id]
    if episode.terminal_snapshot_id and episode.terminal_snapshot_id != episode.initial_snapshot_id:
        snaps.append(episode.terminal_snapshot_id)
    return ReplayEpisodeTruth(
        episode_id=episode.episode_id,
        initial_snapshot_id=episode.initial_snapshot_id,
        terminal_snapshot_id=episode.terminal_snapshot_id,
        snapshot_sequence=tuple(snaps),
        option_surface_sequence=tuple(getattr(episode, "option_surface_ids", ()) or ()),
        data_quality_sequence=tuple(getattr(episode, "data_quality_ids", ()) or ()),
        market_event_ids=tuple(getattr(episode, "source_event_ids", ()) or ()),
        starting_cash=starting_cash,
        fill_model_version="replay_fill_v1",
        random_seed=random_seed,
        contract_quotes=contract_quotes or {},
    )

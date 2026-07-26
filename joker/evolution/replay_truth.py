"""Repository-backed frozen Task 1 replay truth."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from joker.evolution.replay_market import ReplayEpisodeTruth, ReplayPositionSeed
from joker.evolution.schemas import TradingEpisode


class ReplayContractQuote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_id: str
    symbol: str
    expiry: str | None = None
    strike: Decimal | None = None
    option_type: str | None = None
    is_0dte: bool | None = None
    bid: Decimal
    ask: Decimal
    last: Decimal | None = None
    quote_timestamp: datetime | None = None


class ReplayMarketFrame(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: UUID
    timestamp: datetime
    data_quality_id: UUID
    option_surface_id: UUID | None = None
    underlying_bid: Decimal
    underlying_ask: Decimal
    underlying_last: Decimal
    contracts: tuple[ReplayContractQuote, ...] = ()


class ReplayTruthLoadError(RuntimeError):
    pass


class ReplayTruthLoader:
    """Hydrate authoritative replay frames from Task 1 repositories only."""

    def __init__(
        self,
        *,
        snapshot_repo: Any,
        option_surface_repo: Any | None = None,
        data_quality_repo: Any | None = None,
        fill_model_version: str = "replay_fill_v1",
        random_seed: int = 42,
    ) -> None:
        self._snapshots = snapshot_repo
        self._surfaces = option_surface_repo
        self._dq = data_quality_repo
        self._fill_model_version = fill_model_version
        self._random_seed = random_seed

    async def load_for_episode(self, episode: TradingEpisode) -> ReplayEpisodeTruth:
        if episode.snapshot_identity_status == "missing" or episode.initial_snapshot_id is None:
            raise ReplayTruthLoadError("episode_missing_verified_snapshot")
        initial = await self._snapshots.get_by_id(episode.initial_snapshot_id)
        if initial is None:
            raise ReplayTruthLoadError(
                f"initial_snapshot_not_found:{episode.initial_snapshot_id}"
            )

        terminal = None
        if episode.terminal_snapshot_id:
            terminal = await self._snapshots.get_by_id(episode.terminal_snapshot_id)

        # Load all session snapshots in the episode horizon when list API exists.
        frames: list[ReplayMarketFrame] = []
        snapshot_ids: list[UUID] = [episode.initial_snapshot_id]
        if (
            episode.terminal_snapshot_id
            and episode.terminal_snapshot_id != episode.initial_snapshot_id
        ):
            snapshot_ids.append(episode.terminal_snapshot_id)

        # Prefer chronological session listing between start/end when available.
        listed = await self._list_horizon_snapshots(episode, initial, terminal)
        if listed:
            snapshot_ids = listed

        for sid in snapshot_ids:
            snap = await self._snapshots.get_by_id(sid)
            if snap is None:
                raise ReplayTruthLoadError(f"snapshot_not_found:{sid}")
            frames.append(await self._frame_from_snapshot(snap))

        if not frames:
            raise ReplayTruthLoadError("empty_replay_frame_sequence")

        start_ts = frames[0].timestamp
        end_ts = frames[-1].timestamp
        return ReplayEpisodeTruth(
            episode_id=episode.episode_id,
            initial_snapshot_id=episode.initial_snapshot_id,
            terminal_snapshot_id=episode.terminal_snapshot_id,
            snapshot_sequence=tuple(f.snapshot_id for f in frames),
            option_surface_sequence=tuple(
                f.option_surface_id for f in frames if f.option_surface_id is not None
            ),
            data_quality_sequence=tuple(f.data_quality_id for f in frames),
            market_event_ids=tuple(episode.source_event_ids or ()),
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            starting_cash=Decimal("100000"),
            starting_positions=(),
            fill_model_version=self._fill_model_version,
            random_seed=self._random_seed,
            contract_quotes={},  # quotes live on frames only
            frames=tuple(frames),  # type: ignore[call-arg]
        )

    async def _list_horizon_snapshots(
        self,
        episode: TradingEpisode,
        initial: Any,
        terminal: Any | None,
    ) -> list[UUID]:
        list_fn = getattr(self._snapshots, "list_by_session", None)
        if list_fn is None:
            return []
        try:
            records = await list_fn(episode.session_id)
        except Exception:
            return []
        start = getattr(initial, "exchange_time", None) or getattr(
            initial, "exchange_timestamp", None
        )
        end = None
        if terminal is not None:
            end = getattr(terminal, "exchange_time", None) or getattr(
                terminal, "exchange_timestamp", None
            )
        out: list[UUID] = []
        for rec in records:
            ts = getattr(rec, "exchange_time", None) or getattr(
                rec, "exchange_timestamp", None
            )
            sid = getattr(rec, "snapshot_id", None)
            if sid is None:
                continue
            if start is not None and ts is not None and ts < start:
                continue
            if end is not None and ts is not None and ts > end:
                continue
            out.append(UUID(str(sid)))
        return out or [episode.initial_snapshot_id]

    async def _frame_from_snapshot(self, snap: Any) -> ReplayMarketFrame:
        underlying = getattr(snap, "underlying", None)
        if underlying is None:
            raise ReplayTruthLoadError(f"snapshot_missing_underlying:{snap.snapshot_id}")
        surface_id = getattr(snap, "option_surface_id", None)
        contracts: list[ReplayContractQuote] = []
        if surface_id is not None and self._surfaces is not None:
            surface = await self._surfaces.get_by_id(surface_id)
            if surface is not None:
                for c in getattr(surface, "contracts", ()) or ():
                    bid = Decimal(str(getattr(c, "bid", "0") or "0"))
                    ask = Decimal(str(getattr(c, "ask", "0") or "0"))
                    if bid <= 0 and ask <= 0:
                        continue
                    contracts.append(
                        ReplayContractQuote(
                            contract_id=str(getattr(c, "contract_id")),
                            symbol=str(getattr(c, "symbol", "SPY")),
                            expiry=str(getattr(c, "expiry", None) or getattr(c, "expiration", "") or None),
                            strike=(
                                Decimal(str(getattr(c, "strike")))
                                if getattr(c, "strike", None) is not None
                                else None
                            ),
                            option_type=str(getattr(c, "option_type", "") or "") or None,
                            is_0dte=getattr(c, "is_0dte", None),
                            bid=bid if bid > 0 else ask,
                            ask=ask if ask > 0 else bid,
                            last=(
                                Decimal(str(getattr(c, "last")))
                                if getattr(c, "last", None) is not None
                                else None
                            ),
                            quote_timestamp=getattr(c, "quote_timestamp", None),
                        )
                    )
        ts = getattr(snap, "exchange_time", None) or getattr(snap, "exchange_timestamp")
        dq_id = getattr(snap, "data_quality_id")
        if self._dq is not None:
            dq = await self._dq.get_by_id(dq_id)
            if dq is None:
                raise ReplayTruthLoadError(f"missing_data_quality:{dq_id}")
        return ReplayMarketFrame(
            snapshot_id=UUID(str(snap.snapshot_id)),
            timestamp=ts,
            data_quality_id=UUID(str(dq_id)),
            option_surface_id=UUID(str(surface_id)) if surface_id else None,
            underlying_bid=Decimal(str(underlying.bid)),
            underlying_ask=Decimal(str(underlying.ask)),
            underlying_last=Decimal(str(getattr(underlying, "last", underlying.bid))),
            contracts=tuple(contracts),
        )

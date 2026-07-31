"""Repository-backed frozen Task 1 replay truth."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Awaitable
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from joker.evolution.replay_market import (
    ReplayEpisodeTruth,
    ReplayPositionSeed,
    ReplayWorkingOrderSeed,
)
from joker.evolution.schemas import TradingEpisode
from joker.ledger.projector import LedgerProjector, OrderStatus
from joker.ledger.schemas import LedgerEventType


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


ProjectionLoader = Callable[[str], Awaitable[Any]]
CashLoader = Callable[[str, datetime | None], Awaitable[Decimal]]


class ReplayTruthLoader:
    """Hydrate authoritative replay frames and starting ledger state from Task 1."""

    def __init__(
        self,
        *,
        snapshot_repo: Any,
        option_surface_repo: Any | None = None,
        data_quality_repo: Any | None = None,
        ledger_store: Any | None = None,
        projection_loader: ProjectionLoader | None = None,
        session_starting_cash: Decimal | None = None,
        cash_loader: CashLoader | None = None,
        allow_synthetic_starting_cash: bool = False,
        fill_model_version: str = "replay_fill_v1",
        random_seed: int = 42,
        market_calendar_version: str = "us_equity_rth_v1",
        event_horizon_loader: Any | None = None,
    ) -> None:
        self._snapshots = snapshot_repo
        self._surfaces = option_surface_repo
        self._dq = data_quality_repo
        self._ledger = ledger_store
        self._projection_loader = projection_loader
        self._event_horizon_loader = event_horizon_loader
        self._session_starting_cash = session_starting_cash
        self._cash_loader = cash_loader
        self._allow_synthetic_starting_cash = allow_synthetic_starting_cash
        self._fill_model_version = fill_model_version
        self._random_seed = random_seed
        self._market_calendar_version = market_calendar_version
        self._projector = LedgerProjector()

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

        frames: list[ReplayMarketFrame] = []
        snapshot_ids: list[UUID] = [episode.initial_snapshot_id]
        if (
            episode.terminal_snapshot_id
            and episode.terminal_snapshot_id != episode.initial_snapshot_id
        ):
            snapshot_ids.append(episode.terminal_snapshot_id)

        listed = await self._list_horizon_snapshots(episode, initial, terminal)
        horizon_complete = True
        horizon_findings: list[str] = []
        if listed:
            snapshot_ids = listed
        else:
            # Diagnostic entry+terminal frames only — never treat as complete horizon.
            horizon_complete = False
            horizon_findings.extend(
                (
                    "authoritative_horizon_incomplete",
                    "historical_ev_eligible=false",
                    "promotion_eligible=false",
                    "truth_degraded=true",
                )
            )

        for sid in snapshot_ids:
            snap = await self._snapshots.get_by_id(sid)
            if snap is None:
                raise ReplayTruthLoadError(f"snapshot_not_found:{sid}")
            frames.append(await self._frame_from_snapshot(snap))

        if not frames:
            raise ReplayTruthLoadError("empty_replay_frame_sequence")

        start_ts = frames[0].timestamp
        end_ts = frames[-1].timestamp
        entry_ts = episode.entry_decision_timestamp or start_ts
        if episode.entry_decision_event_id is not None and entry_ts is None:
            raise ReplayTruthLoadError("missing_entry_decision_timestamp")
        if episode.terminal_event_timestamp is not None:
            terminal_ts = episode.terminal_event_timestamp
        elif episode.terminal_event_id is None:
            terminal_ts = end_ts if episode.terminal_snapshot_id else None
        else:
            # Event identity without durable timestamp: bound replay using last frame.
            terminal_ts = end_ts if episode.terminal_snapshot_id else None
        if terminal_ts is None and episode.terminal_event_id is not None:
            raise ReplayTruthLoadError("missing_terminal_event_timestamp")
        if (
            entry_ts is not None
            and terminal_ts is not None
            and terminal_ts < entry_ts
        ):
            raise ReplayTruthLoadError("terminal_event_predates_entry")

        market_ids = episode.market_event_ids or episode.source_event_ids or ()

        starting_cash, positions, working = await self._load_starting_ledger(
            episode, as_of=entry_ts
        )
        trading_day = entry_ts.date() if hasattr(entry_ts, "date") else date.today()

        return ReplayEpisodeTruth(
            episode_id=episode.episode_id,
            session_id=episode.session_id,
            trading_date=trading_day,
            initial_snapshot_id=episode.initial_snapshot_id,
            terminal_snapshot_id=episode.terminal_snapshot_id,
            snapshot_sequence=tuple(f.snapshot_id for f in frames),
            option_surface_sequence=tuple(
                f.option_surface_id for f in frames if f.option_surface_id is not None
            ),
            data_quality_sequence=tuple(f.data_quality_id for f in frames),
            market_event_ids=tuple(market_ids),
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            entry_decision_timestamp=entry_ts,
            terminal_event_timestamp=terminal_ts,
            position_lifecycle_id=episode.position_lifecycle_id,
            starting_cash=starting_cash,
            starting_positions=positions,
            starting_working_orders=working,
            market_calendar_version=self._market_calendar_version,
            fill_model_version=self._fill_model_version,
            random_seed=self._random_seed,
            contract_quotes={},
            frames=tuple(frames),
            authoritative_horizon_complete=horizon_complete,
            horizon_integrity_findings=tuple(horizon_findings),
        )

    async def _load_starting_ledger(
        self,
        episode: TradingEpisode,
        *,
        as_of: datetime | None,
    ) -> tuple[Decimal, tuple[ReplayPositionSeed, ...], tuple[ReplayWorkingOrderSeed, ...]]:
        projection = None
        if self._projection_loader is not None:
            projection = await self._projection_loader(episode.session_id)
        elif self._ledger is not None:
            events = await self._ledger.get_by_session(episode.session_id)
            if as_of is not None:
                events = [
                    e
                    for e in events
                    if getattr(e, "exchange_timestamp", None) is None
                    or e.exchange_timestamp <= as_of
                ]
            projection = self._projector.project(events)

        positions: list[ReplayPositionSeed] = []
        working: list[ReplayWorkingOrderSeed] = []
        if projection is not None:
            for cid, pos in (getattr(projection, "positions", None) or {}).items():
                qty = Decimal(str(getattr(pos, "quantity", 0)))
                if qty == 0:
                    continue
                positions.append(
                    ReplayPositionSeed(
                        contract_id=str(cid),
                        quantity=qty,
                        avg_price=Decimal(str(getattr(pos, "average_price", 0) or 0)),
                        side="long" if qty > 0 else "short",
                        position_lifecycle_id=getattr(pos, "position_lifecycle_id", None),
                    )
                )
            for oid, order in (getattr(projection, "orders", None) or {}).items():
                status = getattr(order, "status", None)
                status_val = getattr(status, "value", str(status) if status else "")
                if status_val not in {
                    OrderStatus.ACCEPTED.value,
                    OrderStatus.SUBMITTED.value,
                    OrderStatus.PARTIALLY_FILLED.value,
                    "accepted",
                    "submitted",
                    "partially_filled",
                    "working",
                }:
                    continue
                working.append(
                    ReplayWorkingOrderSeed(
                        client_order_id=str(oid),
                        contract_id=str(getattr(order, "contract_id", "")),
                        side=str(getattr(order, "side", "buy")),
                        quantity=Decimal(str(getattr(order, "submitted_qty", 0) or 0)),
                        filled_qty=Decimal(str(getattr(order, "filled_qty", 0) or 0)),
                        limit_price=(
                            Decimal(str(getattr(order, "limit_price")))
                            if getattr(order, "limit_price", None) is not None
                            else None
                        ),
                        status=status_val,
                        parent_client_order_id=getattr(
                            order, "parent_client_order_id", None
                        ),
                        position_lifecycle_id=getattr(
                            order, "position_lifecycle_id", None
                        ),
                    )
                )

        cash = await self._resolve_starting_cash(episode, as_of=as_of, projection=projection)
        return cash, tuple(positions), tuple(working)

    async def _resolve_starting_cash(
        self,
        episode: TradingEpisode,
        *,
        as_of: datetime | None,
        projection: Any,
    ) -> Decimal:
        if self._cash_loader is not None:
            return await self._cash_loader(episode.session_id, as_of)
        if projection is not None and getattr(projection, "cash", None) is not None:
            return Decimal(str(projection.cash))
        if self._ledger is not None and self._session_starting_cash is not None:
            return await self._cash_from_ledger_fills(
                episode.session_id,
                as_of=as_of,
                starting=self._session_starting_cash,
            )
        if self._session_starting_cash is not None:
            return Decimal(self._session_starting_cash)
        if self._allow_synthetic_starting_cash:
            return Decimal("100000")
        raise ReplayTruthLoadError("missing_authoritative_starting_cash")

    async def _cash_from_ledger_fills(
        self,
        session_id: str,
        *,
        as_of: datetime | None,
        starting: Decimal,
    ) -> Decimal:
        events = await self._ledger.get_by_session(session_id)
        cash = Decimal(starting)
        multiplier = Decimal("100")
        for event in events:
            ts = getattr(event, "exchange_timestamp", None)
            if as_of is not None and ts is not None and ts > as_of:
                continue
            et = getattr(event, "event_type", None)
            et_val = getattr(et, "value", str(et) if et else "")
            if et_val not in {
                LedgerEventType.PARTIAL_FILL.value,
                LedgerEventType.FINAL_FILL.value,
                "partial_fill",
                "final_fill",
            }:
                continue
            qty = Decimal(str(getattr(event, "quantity", 0) or 0))
            price = Decimal(str(getattr(event, "price", 0) or 0))
            fees = Decimal(str(getattr(event, "fees", 0) or 0))
            side = str(getattr(event, "side", "buy")).lower()
            notional = price * qty * multiplier
            if side == "buy":
                cash -= notional + fees
            else:
                cash += notional - fees
        return cash

    async def _list_horizon_snapshots(
        self,
        episode: TradingEpisode,
        initial: Any,
        terminal: Any | None,
    ) -> list[UUID]:
        initial_ts = getattr(initial, "exchange_time", None) or getattr(
            initial, "exchange_timestamp", None
        )
        terminal_snap_ts = None
        if terminal is not None:
            terminal_snap_ts = getattr(terminal, "exchange_time", None) or getattr(
                terminal, "exchange_timestamp", None
            )
        entry_ts = episode.entry_decision_timestamp or initial_ts
        if episode.terminal_event_timestamp is not None:
            terminal_ts = episode.terminal_event_timestamp
        elif episode.terminal_event_id is None:
            terminal_ts = terminal_snap_ts
        else:
            terminal_ts = terminal_snap_ts
        if episode.entry_decision_event_id is not None and entry_ts is None:
            raise ReplayTruthLoadError("missing_entry_decision_timestamp")
        if episode.terminal_event_id is not None and terminal_ts is None:
            raise ReplayTruthLoadError("missing_terminal_event_timestamp")
        if (
            entry_ts is not None
            and terminal_ts is not None
            and terminal_ts < entry_ts
        ):
            raise ReplayTruthLoadError("terminal_event_predates_entry")

        if (
            self._event_horizon_loader is not None
            and entry_ts is not None
            and terminal_ts is not None
        ):
            try:
                horizon = await self._event_horizon_loader.load(
                    session_id=episode.session_id,
                    start_timestamp=entry_ts,
                    end_timestamp=terminal_ts,
                    entry_decision_event_id=episode.entry_decision_event_id,
                    terminal_event_id=episode.terminal_event_id,
                )
                if horizon.snapshot_ids:
                    self._validate_horizon_snapshots(
                        horizon.snapshot_ids, entry_ts, terminal_ts, terminal
                    )
                    return list(horizon.snapshot_ids)
            except ReplayTruthLoadError:
                raise
            except Exception:
                pass

        start = entry_ts or getattr(initial, "exchange_time", None) or getattr(
            initial, "exchange_timestamp", None
        )
        end = terminal_ts or (
            getattr(terminal, "exchange_time", None)
            or getattr(terminal, "exchange_timestamp", None)
            if terminal is not None
            else None
        )
        if start is None or end is None:
            if episode.entry_decision_event_id is not None or episode.terminal_event_id:
                raise ReplayTruthLoadError("missing_horizon_time_bounds")
            return []

        list_fn = getattr(self._snapshots, "list_by_session", None)
        records: list[Any] = []
        if list_fn is not None:
            try:
                records = await list_fn(episode.session_id)
            except Exception:
                records = []
        if not records:
            list_by_date = getattr(self._snapshots, "list_by_trading_date", None)
            if list_by_date is not None:
                try:
                    records = await list_by_date(episode.trading_date)
                except Exception:
                    records = []

        out: list[UUID] = []
        prev_ts: datetime | None = None
        for rec in records:
            ts = getattr(rec, "exchange_time", None) or getattr(
                rec, "exchange_timestamp", None
            )
            sid = getattr(rec, "snapshot_id", None)
            if sid is None:
                continue
            if ts is not None and ts < start:
                continue
            if ts is not None and ts > end:
                continue
            if prev_ts is not None and ts is not None and ts < prev_ts:
                raise ReplayTruthLoadError("horizon_snapshot_non_monotonic")
            if ts is not None:
                prev_ts = ts
            out.append(UUID(str(sid)))

        if not out:
            # Empty authoritative window — caller must treat entry+terminal
            # diagnostic frames as incomplete (not promotion/EV eligible).
            return []

        self._validate_horizon_snapshots(tuple(out), start, end, terminal)
        return out

    @staticmethod
    def _validate_horizon_snapshots(
        snapshot_ids: tuple[UUID, ...] | list[UUID],
        start: datetime,
        end: datetime,
        terminal: Any | None,
    ) -> None:
        if terminal is not None:
            term_snap_ts = getattr(terminal, "exchange_time", None) or getattr(
                terminal, "exchange_timestamp", None
            )
            if term_snap_ts is not None and term_snap_ts > end:
                raise ReplayTruthLoadError("terminal_snapshot_after_terminal_event")
        if not snapshot_ids:
            raise ReplayTruthLoadError("empty_horizon_snapshot_sequence")

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
                            expiry=str(
                                getattr(c, "expiry", None)
                                or getattr(c, "expiration", "")
                                or None
                            ),
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

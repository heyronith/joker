"""Market data runtime — observations, bars, snapshots, surfaces, quality.

Must not call an LLM, select direction, submit orders, or manage exits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID, uuid4

from joker.events.bus import InProcessAsyncEventBus
from joker.events.schemas import EventType, make_event
from joker.market.bars import BarBuilder, BarTimeframe, MarketBar
from joker.market.observations import (
    OptionQuoteObservation,
    QuoteObservation,
    TradeObservation,
    UnderlyingObservation,
)
from joker.market.option_surface import (
    OptionSurfaceBuilder,
    OptionSurfaceRepository,
    OptionSurfaceSnapshot,
)
from joker.market.quality import DataQualityReport, evaluate_data_quality
from joker.market.snapshots import (
    DataQualitySnapshot,
    MarketSnapshot,
    SnapshotRepository,
    UnderlyingSnapshot,
)
from joker.time.clock import ExchangeClock

logger = logging.getLogger(__name__)


@dataclass
class MarketRuntimeConfig:
    """Tunables for market ingestion (no strategy knobs)."""

    symbol: str = "SPY"
    underlying_stale_seconds: float = 5.0
    option_stale_seconds: float = 10.0
    maximum_relative_spread: float = 0.25
    min_option_contracts: int = 2
    bars_1m_window: int = 60
    bars_5m_window: int = 24


@dataclass
class MarketTickResult:
    """Outcome of a single ``tick`` — closed bars and optional snapshot."""

    closed_bars: list[MarketBar] = field(default_factory=list)
    snapshot: MarketSnapshot | None = None
    surface: OptionSurfaceSnapshot | None = None
    quality: DataQualityReport | None = None
    events_published: int = 0


class MarketRuntime:
    """Deterministic market-data path: ingest → bars → snapshot → events.

    Explicitly forbidden: LLM calls, direction selection, order submission,
    exit management.
    """

    def __init__(
        self,
        *,
        clock: ExchangeClock,
        bar_builder: BarBuilder,
        event_bus: InProcessAsyncEventBus,
        snapshot_repo: SnapshotRepository,
        surface_builder: OptionSurfaceBuilder | None = None,
        surface_repo: OptionSurfaceRepository | None = None,
        quality_evaluator: Any = evaluate_data_quality,
        session_id: str,
        config: MarketRuntimeConfig | None = None,
    ) -> None:
        self._clock = clock
        self._bars = bar_builder
        self._bus = event_bus
        self._snapshots = snapshot_repo
        self._surface_builder = surface_builder or OptionSurfaceBuilder()
        self._surfaces = surface_repo
        self._evaluate_quality = quality_evaluator
        self._session_id = session_id
        self._config = config or MarketRuntimeConfig()

        self._latest_underlying: UnderlyingSnapshot | None = None
        self._pending_option_rows: list[dict[str, Any]] = []
        self._latest_surface: OptionSurfaceSnapshot | None = None
        self._source_event_ids: list[UUID] = []
        self._correlation_id = uuid4()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def latest_underlying(self) -> UnderlyingSnapshot | None:
        return self._latest_underlying

    @property
    def latest_surface(self) -> OptionSurfaceSnapshot | None:
        return self._latest_surface

    async def ingest_underlying_quote(
        self,
        *,
        symbol: str | None = None,
        bid: Decimal | None = None,
        ask: Decimal | None = None,
        last: Decimal | None = None,
        bid_size: int | None = None,
        ask_size: int | None = None,
        last_size: int | None = None,
        cumulative_volume: int | None = None,
        source_timestamp: datetime | None = None,
        received_timestamp: datetime | None = None,
        source: str = "market_runtime",
        observation: UnderlyingObservation | QuoteObservation | None = None,
    ) -> UnderlyingObservation | QuoteObservation:
        """Ingest an underlying quote/observation and publish QUOTE_RECEIVED."""
        now = self._clock.now()
        if observation is None:
            observation = UnderlyingObservation(
                symbol=symbol or self._config.symbol,
                source_timestamp=source_timestamp or now,
                received_timestamp=received_timestamp or now,
                bid=bid,
                ask=ask,
                last=last,
                bid_size=bid_size,
                ask_size=ask_size,
                last_size=last_size,
                cumulative_volume=cumulative_volume,
                source=source,
            )

        if isinstance(observation, UnderlyingObservation):
            self._bars.ingest_underlying(observation)
            self._latest_underlying = UnderlyingSnapshot(
                symbol=observation.symbol,
                bid=observation.bid,
                ask=observation.ask,
                last=observation.last,
                quote_timestamp=observation.source_timestamp,
            )
        else:
            self._bars.ingest_quote(observation)
            self._latest_underlying = UnderlyingSnapshot(
                symbol=observation.symbol,
                bid=observation.bid,
                ask=observation.ask,
                last=observation.last,
                quote_timestamp=observation.source_timestamp,
            )

        event = make_event(
            EventType.QUOTE_RECEIVED,
            session_id=self._session_id,
            source="market_runtime",
            exchange_timestamp=observation.source_timestamp,
            correlation_id=self._correlation_id,
            payload={
                "observation_id": str(observation.observation_id),
                "symbol": observation.symbol,
            },
        )
        self._source_event_ids.append(event.event_id)
        await self._bus.publish(event)
        return observation

    async def ingest_trade(
        self,
        *,
        symbol: str | None = None,
        price: Decimal | None = None,
        size: int | None = None,
        cumulative_volume: int | None = None,
        source_timestamp: datetime | None = None,
        received_timestamp: datetime | None = None,
        source: str = "market_runtime",
        observation: TradeObservation | None = None,
    ) -> TradeObservation:
        """Ingest a trade print and publish TRADE_RECEIVED."""
        now = self._clock.now()
        if observation is None:
            if price is None or size is None:
                raise ValueError("price and size are required when observation is omitted")
            observation = TradeObservation(
                symbol=symbol or self._config.symbol,
                source_timestamp=source_timestamp or now,
                received_timestamp=received_timestamp or now,
                price=price,
                size=size,
                cumulative_volume=cumulative_volume,
                source=source,
            )
        self._bars.ingest_trade(observation)
        event = make_event(
            EventType.TRADE_RECEIVED,
            session_id=self._session_id,
            source="market_runtime",
            exchange_timestamp=observation.source_timestamp,
            correlation_id=self._correlation_id,
            payload={
                "observation_id": str(observation.observation_id),
                "symbol": observation.symbol,
                "price": str(observation.price),
                "size": observation.size,
            },
        )
        self._source_event_ids.append(event.event_id)
        await self._bus.publish(event)
        return observation

    async def ingest_option_quotes(
        self,
        quotes: Sequence[OptionQuoteObservation | dict[str, Any]],
        *,
        publish_event: bool = True,
    ) -> list[dict[str, Any]]:
        """Buffer option quotes for the next surface build (no direction logic)."""
        rows: list[dict[str, Any]] = []
        for q in quotes:
            if isinstance(q, OptionQuoteObservation):
                rows.append(
                    {
                        "contract_id": q.contract_symbol,
                        "symbol": q.contract_symbol,
                        "expiry": q.expiry,
                        "strike": q.strike,
                        "option_type": q.option_type,
                        "bid": q.bid,
                        "ask": q.ask,
                        "last": q.last,
                        "bid_size": q.bid_size,
                        "ask_size": q.ask_size,
                        "volume": q.volume,
                        "open_interest": q.open_interest,
                        "implied_volatility": q.implied_volatility,
                        "delta": q.delta,
                        "gamma": q.gamma,
                        "theta": q.theta,
                        "vega": q.vega,
                        "quote_timestamp": q.source_timestamp,
                    }
                )
            else:
                rows.append(dict(q))
        self._pending_option_rows = rows
        if publish_event and rows:
            now = self._clock.now()
            event = make_event(
                EventType.QUOTE_RECEIVED,
                session_id=self._session_id,
                source="market_runtime",
                exchange_timestamp=now,
                correlation_id=self._correlation_id,
                payload={"option_quote_count": len(rows), "kind": "option_surface_buffer"},
            )
            self._source_event_ids.append(event.event_id)
            await self._bus.publish(event)
        return rows

    async def tick(self, now: datetime | None = None) -> MarketTickResult:
        """Close ready bars; optionally build surface, quality report, and snapshot."""
        reference = now if now is not None else self._clock.now()
        result = MarketTickResult()
        closed = self._bars.close_ready_bars(reference)
        result.closed_bars = closed

        for bar in closed:
            event = make_event(
                EventType.BAR_CLOSED,
                session_id=self._session_id,
                source="market_runtime",
                exchange_timestamp=bar.end,
                correlation_id=self._correlation_id,
                payload={
                    "symbol": bar.symbol,
                    "timeframe": bar.timeframe.value,
                    "start": bar.start.isoformat(),
                    "end": bar.end.isoformat(),
                    "volume": bar.volume,
                    "incomplete": bar.incomplete,
                },
            )
            self._source_event_ids.append(event.event_id)
            await self._bus.publish(event)
            result.events_published += 1

        surface: OptionSurfaceSnapshot | None = None
        if self._pending_option_rows:
            trading_day = self._clock.trading_date()
            underlying_px = None
            if self._latest_underlying is not None:
                underlying_px = (
                    self._latest_underlying.last
                    or self._latest_underlying.bid
                    or self._latest_underlying.ask
                )
            surface = self._surface_builder.build(
                underlying_symbol=self._config.symbol,
                exchange_time=reference,
                trading_date=trading_day,
                rows=self._pending_option_rows,
                underlying_price=underlying_px,
                now=reference,
            )
            self._latest_surface = surface
            self._pending_option_rows = []
            if self._surfaces is not None:
                await self._surfaces.save(surface)
            surface_event = make_event(
                EventType.OPTION_SURFACE_CREATED,
                session_id=self._session_id,
                source="market_runtime",
                exchange_timestamp=reference,
                correlation_id=self._correlation_id,
                payload={
                    "surface_id": str(surface.surface_id),
                    "contract_count": len(surface.contracts),
                },
            )
            self._source_event_ids.append(surface_event.event_id)
            await self._bus.publish(surface_event)
            result.events_published += 1
            result.surface = surface

        # Build snapshot when we have an underlying and at least one closed bar
        # this tick, or when a surface was produced.
        if self._latest_underlying is None:
            return result
        if not closed and surface is None:
            return result

        bars_1m = tuple(
            b
            for b in self._bars.closed_bars()
            if b.timeframe == BarTimeframe.M1
        )[-self._config.bars_1m_window :]
        bars_5m = tuple(
            b
            for b in self._bars.closed_bars()
            if b.timeframe == BarTimeframe.M5
        )[-self._config.bars_5m_window :]

        quality: DataQualityReport = self._evaluate_quality(
            underlying=self._latest_underlying,
            bars_1m=bars_1m,
            option_surface=self._latest_surface,
            now=reference,
            underlying_stale_seconds=self._config.underlying_stale_seconds,
            option_stale_seconds=self._config.option_stale_seconds,
            maximum_relative_spread=self._config.maximum_relative_spread,
            min_contracts=self._config.min_option_contracts,
        )
        result.quality = quality
        dq_snap = DataQualitySnapshot(
            data_quality_id=quality.report_id,
            severity=quality.severity.value,
            finding_codes=tuple(f.code for f in quality.findings),
            usable_for_reasoning=quality.usable_for_reasoning,
            usable_for_execution=quality.usable_for_execution,
        )

        trading_day: date = self._clock.trading_date()
        snapshot = MarketSnapshot(
            exchange_time=reference,
            trading_date=trading_day,
            underlying=self._latest_underlying,
            bars_1m=bars_1m,
            bars_5m=bars_5m,
            option_surface_id=self._latest_surface.surface_id if self._latest_surface else None,
            data_quality_id=dq_snap.data_quality_id,
            source_event_ids=tuple(self._source_event_ids[-64:]),
        )
        await self._snapshots.save(snapshot)
        snap_event = make_event(
            EventType.MARKET_SNAPSHOT_CREATED,
            session_id=self._session_id,
            source="market_runtime",
            exchange_timestamp=reference,
            correlation_id=self._correlation_id,
            payload={
                "snapshot_id": str(snapshot.snapshot_id),
                "data_quality_id": str(snapshot.data_quality_id),
                "option_surface_id": (
                    str(snapshot.option_surface_id) if snapshot.option_surface_id else None
                ),
                "usable_for_execution": quality.usable_for_execution,
            },
        )
        self._source_event_ids.append(snap_event.event_id)
        await self._bus.publish(snap_event)
        result.events_published += 1
        result.snapshot = snapshot

        logger.info(
            "market_snapshot_created",
            extra={
                "session_id": self._session_id,
                "snapshot_id": str(snapshot.snapshot_id),
                "correlation_id": str(self._correlation_id),
            },
        )
        return result

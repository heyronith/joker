"""Replay market data provider."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone

from joker.data.replay_loader import load_replay_file
from joker.data.provider import MarketDataProvider
from joker.schemas.domain import Candle, MarketSnapshot, OptionContract, OptionQuote
from joker.schemas.replay import (
    MarketEvent,
    OptionChainSnapshot,
    OptionQuoteEvent,
    ReplayClock,
    ReplaySession,
    ReplaySpeedMode,
    SpyCandleEvent,
    SpyQuoteEvent,
)


class ReplayMarketDataProvider(MarketDataProvider):
    """Feeds market events from a replay session in timestamp order."""

    def __init__(
        self,
        session: ReplaySession,
        clock: ReplayClock | None = None,
    ) -> None:
        self.session = session
        self._events = list(session.events)
        self._index = 0
        self._candles: list[Candle] = []
        self._latest_snapshot: MarketSnapshot | None = None
        self._option_quotes: dict[str, OptionQuoteEvent] = {}
        self._option_chain: OptionChainSnapshot | None = None
        start = session.metadata.start_time or session.events[0].timestamp
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        self.clock = clock or ReplayClock(
            current_time=start,
            trading_day=session.metadata.trading_day,
            mode=ReplaySpeedMode.DETERMINISTIC,
        )

    @classmethod
    def from_file(cls, path: str) -> ReplayMarketDataProvider:
        session = load_replay_file(path)
        return cls(session)

    @property
    def current_time(self) -> datetime:
        return self.clock.current_time

    def _apply_event(self, event: MarketEvent) -> None:
        ts = event.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        self.clock.advance_to(ts)

        if isinstance(event, SpyQuoteEvent):
            self._latest_snapshot = MarketSnapshot(
                symbol=event.symbol,
                timestamp=ts,
                price=event.price,
                bid=event.bid,
                ask=event.ask,
                candles=list(self._candles),
            )
        elif isinstance(event, SpyCandleEvent):
            self._candles.append(event.candle)
            if self._latest_snapshot:
                self._latest_snapshot = self._latest_snapshot.model_copy(
                    update={"candles": list(self._candles), "timestamp": ts}
                )
        elif isinstance(event, OptionQuoteEvent):
            self._option_quotes[event.contract_id] = event
        elif isinstance(event, OptionChainSnapshot):
            self._option_chain = event

    def stream_events(self) -> Iterator[MarketEvent]:
        while self._index < len(self._events):
            event = self._events[self._index]
            self._index += 1
            self._apply_event(event)
            yield event

    def peek_remaining(self) -> int:
        return len(self._events) - self._index

    def get_latest_snapshot(self) -> MarketSnapshot | None:
        return self._latest_snapshot

    def get_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        if symbol != "SPY":
            return []
        return list(self._candles)

    def get_option_chain(self, symbol: str, expiration: date) -> OptionChainSnapshot | None:
        if self._option_chain and self._option_chain.expiration == expiration:
            return self._option_chain
        return None

    def get_option_quote(self, contract_id: str) -> OptionQuote | None:
        raw = self._option_quotes.get(contract_id)
        if raw is None:
            return None
        contract = OptionContract(
            symbol=raw.symbol,
            expiration=raw.expiration,
            strike=raw.strike,
            option_type=raw.option_type,
            is_0dte=True,
        )
        return OptionQuote(
            contract=contract,
            bid=raw.bid,
            ask=raw.ask,
            last=raw.mid,
            timestamp=raw.quote_timestamp,
        )

    def list_option_quote_events(self) -> list[OptionQuoteEvent]:
        return list(self._option_quotes.values())

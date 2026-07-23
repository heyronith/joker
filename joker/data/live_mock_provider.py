"""Static mock market data provider for tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone

from joker.data.provider import MarketDataProvider
from joker.schemas.domain import Candle, MarketSnapshot, OptionContract, OptionQuote
from joker.schemas.replay import MarketEvent, OptionChainSnapshot, SpyQuoteEvent


class MockMarketDataProvider(MarketDataProvider):
    """In-memory provider with configurable events."""

    def __init__(
        self,
        events: list[MarketEvent] | None = None,
        snapshot: MarketSnapshot | None = None,
    ) -> None:
        self._events = list(events or [])
        self._snapshot = snapshot
        self._candles: list[Candle] = []
        self._chain: OptionChainSnapshot | None = None
        self._quotes: dict[str, OptionQuote] = {}
        self._time = datetime.now(timezone.utc)

    def add_quote(self, contract_id: str, quote: OptionQuote) -> None:
        self._quotes[contract_id] = quote

    def stream_events(self) -> Iterator[MarketEvent]:
        for event in self._events:
            self._time = event.timestamp
            if isinstance(event, SpyQuoteEvent) and self._snapshot:
                self._snapshot = self._snapshot.model_copy(
                    update={"price": event.price, "timestamp": event.timestamp}
                )
            yield event

    def get_latest_snapshot(self) -> MarketSnapshot | None:
        return self._snapshot

    def get_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        return list(self._candles)

    def get_option_chain(self, symbol: str, expiration: date) -> OptionChainSnapshot | None:
        return self._chain

    def get_option_quote(self, contract_id: str) -> OptionQuote | None:
        return self._quotes.get(contract_id)

    @property
    def current_time(self) -> datetime:
        return self._time

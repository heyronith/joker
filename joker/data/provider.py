"""Market data provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import date, datetime

from joker.schemas.domain import Candle, MarketSnapshot, OptionContract, OptionQuote
from joker.schemas.replay import MarketEvent, OptionChainSnapshot


class MarketDataProvider(ABC):
    """Abstract market data source for live or replay feeds."""

    @abstractmethod
    def stream_events(self) -> Iterator[MarketEvent]:
        ...

    @abstractmethod
    def get_latest_snapshot(self) -> MarketSnapshot | None:
        ...

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        ...

    @abstractmethod
    def get_option_chain(self, symbol: str, expiration: date) -> OptionChainSnapshot | None:
        ...

    @abstractmethod
    def get_option_quote(self, contract_id: str) -> OptionQuote | None:
        ...

    @property
    @abstractmethod
    def current_time(self) -> datetime:
        ...

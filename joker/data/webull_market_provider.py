"""Webull market data provider — SPY stock quotes and candles only."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, timezone
from typing import Any

from joker.config.settings import EnvSettings
from joker.data.provider import MarketDataProvider
from joker.data.webull_api import (
    ALLOWED_SYMBOL,
    HttpWebullMarketApi,
    MockWebullMarketApi,
    WebullApiError,
    WebullCandle,
    WebullMarketApi,
    WebullQuote,
    _parse_quote_payload,
)
from joker.schemas.domain import Candle, MarketSnapshot, OptionQuote
from joker.compliance.data_classification import DataClassification, SOURCE_WEBULL_STOCK
from joker.schemas.replay import MarketEvent, OptionChainSnapshot, SpyCandleEvent, SpyQuoteEvent


class WebullMarketDataProvider(MarketDataProvider):
    """Real SPY market data via Webull OpenAPI — data-only, no options or orders."""

    OPTIONS_UNAVAILABLE = "Options data unavailable in market-data-only phase"

    def __init__(
        self,
        env: EnvSettings,
        *,
        api: WebullMarketApi | None = None,
        quote_max_age_seconds: int = 30,
        feed_max_silence_seconds: int = 60,
        allow_delayed_quotes: bool = True,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self._env = env
        self._api = api or HttpWebullMarketApi(env)
        self._quote_max_age_seconds = quote_max_age_seconds
        self._feed_max_silence_seconds = feed_max_silence_seconds
        self._allow_delayed_quotes = allow_delayed_quotes
        self._poll_interval_seconds = poll_interval_seconds
        self._candles: list[Candle] = []
        self._latest_snapshot: MarketSnapshot | None = None
        self._current_time = datetime.now(timezone.utc)
        self._last_received_at: datetime | None = None
        self._feed_health: str = "OK"
        self._permission_warning: str | None = None
        self._last_quote_delayed: bool = False
        self._stream_events: list[MarketEvent] = []
        self._stream_index = 0
        self._authenticated = False
        # "bars" = Webull 1m OHLCV loaded; "quotes" = quote-derived fallback
        self._candle_source: str = "none"

    def close(self) -> None:
        """Close the underlying Webull market HTTP client when present."""
        close = getattr(self._api, "close", None)
        if callable(close):
            close()

    @property
    def feed_health(self) -> str:
        return self._feed_health

    @property
    def candle_source(self) -> str:
        return self._candle_source

    @property
    def has_volume_bars(self) -> bool:
        return self._candle_source == "bars" and any(c.volume > 0 for c in self._candles)

    @property
    def permission_warning(self) -> str | None:
        return self._permission_warning

    @property
    def options_available(self) -> bool:
        return False

    @property
    def current_time(self) -> datetime:
        return self._current_time

    @property
    def last_received_at(self) -> datetime | None:
        return self._last_received_at

    @property
    def last_quote_delayed(self) -> bool:
        return self._last_quote_delayed

    def authenticate(self) -> bool:
        result = self._api.authenticate()
        self._authenticated = result.success
        if not result.success:
            self._feed_health = "ERROR"
            self._permission_warning = result.message
        return result.success

    def _quote_to_event(self, quote: WebullQuote) -> SpyQuoteEvent:
        if quote.symbol.upper() != ALLOWED_SYMBOL:
            raise WebullApiError(f"Non-SPY symbol rejected: {quote.symbol}")
        self._current_time = quote.timestamp
        if quote.delayed:
            self._last_quote_delayed = True
        return SpyQuoteEvent(
            timestamp=quote.timestamp,
            symbol=quote.symbol.upper(),
            source=SOURCE_WEBULL_STOCK,
            price=quote.price,
            bid=quote.bid,
            ask=quote.ask,
            data_classification=DataClassification.STOCK_MARKET_DATA.value,
        )

    def _candle_to_event(self, row: WebullCandle) -> SpyCandleEvent:
        candle = Candle(
            symbol=ALLOWED_SYMBOL,
            timestamp=row.timestamp,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
        )
        self._current_time = row.timestamp
        return SpyCandleEvent(
            timestamp=row.timestamp,
            symbol=ALLOWED_SYMBOL,
            source=SOURCE_WEBULL_STOCK,
            candle=candle,
            data_classification=DataClassification.STOCK_MARKET_DATA.value,
        )

    def _apply_quote_event(self, event: SpyQuoteEvent) -> None:
        from joker.data.freshness import feed_health_from_received

        now = datetime.now(timezone.utc)
        self._last_received_at = now
        ts = event.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Wall-clock time for exit/session logic; exchange ts kept on the event.
        self._current_time = now
        self._latest_snapshot = MarketSnapshot(
            symbol=event.symbol,
            timestamp=ts,
            price=event.price,
            bid=event.bid,
            ask=event.ask,
            candles=list(self._candles),
        )
        silence = self._feed_max_silence_seconds
        if self._last_quote_delayed and self._allow_delayed_quotes:
            self._feed_health = feed_health_from_received(
                last_received_at=self._last_received_at,
                now=now,
                feed_max_silence_seconds=silence,
            )
        else:
            age = (now - ts).total_seconds()
            if age > self._quote_max_age_seconds:
                self._feed_health = "STALE"
            else:
                self._feed_health = "OK"

    def _apply_candle_event(self, event: SpyCandleEvent, *, replace_same_minute: bool = False) -> None:
        candle = event.candle
        if replace_same_minute and self._candles:
            last = self._candles[-1]
            last_m = last.timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
            new_m = candle.timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
            if last_m == new_m:
                self._candles[-1] = candle
            else:
                self._candles.append(candle)
        else:
            self._candles.append(candle)
        ts = event.timestamp
        if self._latest_snapshot:
            self._latest_snapshot = self._latest_snapshot.model_copy(
                update={"candles": list(self._candles), "timestamp": ts}
            )

    def append_quote_as_candle(self, event: SpyQuoteEvent) -> None:
        """Upsert the forming 1-minute bar from a live quote (fallback or bar refresh)."""
        ts = event.timestamp if event.timestamp.tzinfo else event.timestamp.replace(tzinfo=timezone.utc)
        minute_ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
        price = float(event.price)
        if self._candles:
            last = self._candles[-1]
            last_m = last.timestamp.astimezone(timezone.utc).replace(second=0, microsecond=0)
            if last_m == minute_ts:
                updated = Candle(
                    symbol=ALLOWED_SYMBOL,
                    timestamp=minute_ts,
                    open=last.open,
                    high=max(last.high, price),
                    low=min(last.low, price),
                    close=price,
                    # Preserve real bar volume when present; quote ticks have no volume.
                    volume=last.volume,
                )
                self._candles[-1] = updated
                if self._latest_snapshot:
                    self._latest_snapshot = self._latest_snapshot.model_copy(
                        update={"candles": list(self._candles), "timestamp": ts, "price": price}
                    )
                return

        candle = Candle(
            symbol=ALLOWED_SYMBOL,
            timestamp=minute_ts,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0.0,
        )
        if self._candle_source == "none":
            self._candle_source = "quotes"
        self._apply_candle_event(
            SpyCandleEvent(
                timestamp=minute_ts,
                symbol=ALLOWED_SYMBOL,
                source=SOURCE_WEBULL_STOCK,
                candle=candle,
                data_classification=DataClassification.STOCK_MARKET_DATA.value,
            ),
            replace_same_minute=True,
        )

    def fetch_snapshot_event(self) -> SpyQuoteEvent:
        """Fetch live snapshot and return normalized SpyQuoteEvent."""
        try:
            quote = self._api.get_snapshot(ALLOWED_SYMBOL)
        except WebullApiError as exc:
            self._feed_health = "ERROR"
            if exc.subscription_related:
                self._permission_warning = (
                    "OpenAPI market-data subscription may be required"
                )
            raise
        event = self._quote_to_event(quote)
        self._apply_quote_event(event)
        return event

    def fetch_candle_events(self, timeframe: str) -> list[SpyCandleEvent]:
        rows = self._api.get_candles(ALLOWED_SYMBOL, timeframe)
        # Replace prior history with authoritative bars (chronological).
        self._candles = []
        events = [self._candle_to_event(row) for row in rows]
        for event in events:
            self._apply_candle_event(event)
        self._candle_source = "bars" if events else self._candle_source
        return events

    def prepare_stream(self, *, duration_seconds: float) -> None:
        """Pre-fetch stream quotes into internal buffer for stream_events()."""
        self._stream_events = []
        self._stream_index = 0
        try:
            for quote in self._api.stream_quotes(
                ALLOWED_SYMBOL,
                duration_seconds=duration_seconds,
                poll_interval_seconds=self._poll_interval_seconds,
            ):
                event = self._quote_to_event(quote)
                self._stream_events.append(event)
        except WebullApiError as exc:
            self._feed_health = "ERROR"
            if exc.subscription_related:
                self._permission_warning = (
                    "OpenAPI market-data subscription may be required"
                )
            raise

    def stream_events(self) -> Iterator[MarketEvent]:
        if not self._stream_events:
            event = self.fetch_snapshot_event()
            self._apply_quote_event(event)
            yield event
            return
        while self._stream_index < len(self._stream_events):
            event = self._stream_events[self._stream_index]
            self._stream_index += 1
            if isinstance(event, SpyQuoteEvent):
                self._apply_quote_event(event)
            elif isinstance(event, SpyCandleEvent):
                self._apply_candle_event(event)
            yield event

    def get_latest_snapshot(self) -> MarketSnapshot | None:
        return self._latest_snapshot

    def get_candles(self, symbol: str, timeframe: str) -> list[Candle]:
        if symbol.upper() != ALLOWED_SYMBOL:
            return []
        if not self._candles:
            self.fetch_candle_events(timeframe)
        return list(self._candles)

    def get_option_chain(self, symbol: str, expiration: date) -> OptionChainSnapshot | None:
        return None

    def get_option_quote(self, contract_id: str) -> OptionQuote | None:
        return None

    def normalize_raw_quote(self, symbol: str, data: dict[str, Any]) -> SpyQuoteEvent:
        """Normalize a raw Webull quote payload (for tests and diagnostics)."""
        quote = _parse_quote_payload(symbol, data)
        return self._quote_to_event(quote)


def create_webull_market_api(
    env: EnvSettings,
    *,
    api: WebullMarketApi | None = None,
) -> WebullMarketApi:
    return api or HttpWebullMarketApi(env)

"""Webull OpenAPI market-data client (SPY stock only — no trading endpoints)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Protocol

import httpx

from joker.config.settings import EnvSettings
from joker.data.webull_config import WebullApiEnv, ensure_live_trading_disabled
from joker.data.webull_endpoints import require_verified
from joker.data.webull_errors import WebullApiError, WebullAuthResult
from joker.data.webull_http import WebullHttpClient

ALLOWED_SYMBOL = "SPY"


@dataclass
class WebullQuote:
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    timestamp: datetime
    delayed: bool = False


@dataclass
class WebullCandle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class WebullMarketApi(Protocol):
    """Market-data API surface — no order or account methods."""

    def authenticate(self) -> WebullAuthResult: ...

    def get_snapshot(self, symbol: str) -> WebullQuote: ...

    def get_candles(self, symbol: str, timeframe: str) -> list[WebullCandle]: ...

    def stream_quotes(
        self,
        symbol: str,
        *,
        duration_seconds: float,
        poll_interval_seconds: float = 1.0,
    ) -> Iterator[WebullQuote]: ...


def _parse_timestamp(raw: Any) -> datetime:
    if isinstance(raw, (int, float)):
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(raw, str):
        text = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise WebullApiError("Unknown timestamp format in Webull response")


def _ensure_spy(symbol: str) -> None:
    if symbol.upper() != ALLOWED_SYMBOL:
        raise WebullApiError(f"Only SPY is supported; got {symbol!r}")


def _first_row(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        if not payload:
            raise WebullApiError("Empty Webull snapshot response")
        row = payload[0]
        if not isinstance(row, dict):
            raise WebullApiError("Malformed Webull snapshot row")
        return row
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data:
            row = data[0]
            if isinstance(row, dict):
                return row
        if data is None:
            return payload
        if isinstance(data, dict):
            return data
        return payload
    raise WebullApiError("Unexpected Webull response shape")


def _parse_quote_payload(symbol: str, data: dict[str, Any]) -> WebullQuote:
    price = data.get("price") or data.get("close") or data.get("last")
    if price is None:
        raise WebullApiError("Malformed Webull quote: missing price field")
    bid = data.get("bid")
    ask = data.get("ask")
    ts_raw = (
        data.get("quote_time")
        or data.get("timestamp")
        or data.get("tradeTime")
        or data.get("time")
    )
    if ts_raw is None:
        raise WebullApiError("Malformed Webull quote: missing timestamp")
    delayed = bool(data.get("delayed", data.get("isDelayed", False)))
    return WebullQuote(
        symbol=symbol.upper(),
        price=float(price),
        bid=float(bid) if bid is not None else None,
        ask=float(ask) if ask is not None else None,
        timestamp=_parse_timestamp(ts_raw),
        delayed=delayed,
    )


def normalize_stock_timespan(timeframe: str) -> str:
    """Map joker timeframes to Webull stock bars timespan codes."""
    key = (timeframe or "").strip().lower()
    mapping = {
        "1m": "M1",
        "m1": "M1",
        "5m": "M5",
        "m5": "M5",
        "15m": "M15",
        "m15": "M15",
        "30m": "M30",
        "m30": "M30",
        "60m": "M60",
        "1h": "M60",
        "1d": "D",
        "d": "D",
    }
    if key.upper() in {"M1", "M5", "M15", "M30", "M60", "D", "W", "MO"}:
        return key.upper()
    return mapping.get(key, timeframe)


def _parse_candle_row(row: dict[str, Any]) -> WebullCandle:
    ts_raw = row.get("timestamp") or row.get("time") or row.get("t")
    if ts_raw is None:
        raise WebullApiError("Malformed Webull candle: missing timestamp")
    open_v = row.get("open") if row.get("open") is not None else row.get("o")
    high_v = row.get("high") if row.get("high") is not None else row.get("h")
    low_v = row.get("low") if row.get("low") is not None else row.get("l")
    close_v = row.get("close") if row.get("close") is not None else row.get("c")
    if open_v is None or high_v is None or low_v is None or close_v is None:
        raise WebullApiError("Malformed Webull candle: missing OHLC")
    return WebullCandle(
        timestamp=_parse_timestamp(ts_raw),
        open=float(open_v),
        high=float(high_v),
        low=float(low_v),
        close=float(close_v),
        volume=float(row.get("volume") or row.get("v") or 0),
    )


class HttpWebullMarketApi:
    """HTTP client for Webull OpenAPI market data (verified endpoints only)."""

    MARKET_DATA_CALLS_ENABLED = True

    def __init__(
        self,
        env: EnvSettings,
        *,
        client: httpx.Client | None = None,
        http_client: WebullHttpClient | None = None,
    ) -> None:
        ensure_live_trading_disabled(env)
        self._env = env
        self._api_env = WebullApiEnv.from_string(env.webull_api_env)
        self._http = http_client or WebullHttpClient(env, client=client)

    def close(self) -> None:
        """Close the owned Webull HTTP client."""
        close = getattr(self._http, "close", None)
        if callable(close):
            close()

    def _ensure_auth(self) -> None:
        if self._http.access_token:
            return
        auth = self.authenticate()
        if not auth.success:
            raise WebullApiError(f"Auth failed: {auth.message}")

    def authenticate(self) -> WebullAuthResult:
        if self._env.webull_access_token:
            self._http.set_access_token(self._env.webull_access_token)
            return WebullAuthResult(success=True, token_present=True, message="Using configured token")
        return self._http.authenticate_legacy()

    def get_snapshot(self, symbol: str) -> WebullQuote:
        _ensure_spy(symbol)
        require_verified("stock_snapshot")
        self._ensure_auth()
        payload = self._http.request_json(
            "stock_snapshot",
            params={"symbols": symbol.upper(), "category": "US_STOCK"},
        )
        row = _first_row(payload)
        return _parse_quote_payload(symbol, row)

    def get_candles(self, symbol: str, timeframe: str) -> list[WebullCandle]:
        _ensure_spy(symbol)
        require_verified("stock_bars")
        self._ensure_auth()
        timespan = normalize_stock_timespan(timeframe)
        payload = self._http.request_json(
            "stock_bars",
            params={
                "symbol": symbol.upper(),
                "category": "US_STOCK",
                "timespan": timespan,
            },
        )
        rows = payload if isinstance(payload, list) else payload.get("data") or payload.get("bars") or []
        if not isinstance(rows, list):
            raise WebullApiError("Malformed Webull candles response")
        candles = [_parse_candle_row(row) for row in rows if isinstance(row, dict)]
        # Webull returns newest-first; FeatureEngine expects chronological order.
        candles.sort(key=lambda c: c.timestamp)
        return candles

    def stream_quotes(
        self,
        symbol: str,
        *,
        duration_seconds: float,
        poll_interval_seconds: float = 1.0,
    ) -> Iterator[WebullQuote]:
        _ensure_spy(symbol)
        deadline = time.monotonic() + max(duration_seconds, 0)
        while time.monotonic() < deadline:
            yield self.get_snapshot(symbol)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_seconds, remaining))


class MockWebullMarketApi:
    """Offline test double for Webull market data."""

    def __init__(
        self,
        *,
        quote: WebullQuote | None = None,
        candles: list[WebullCandle] | None = None,
        auth_success: bool = True,
        fail_snapshot: WebullApiError | None = None,
        fail_candles: WebullApiError | None = None,
        stream_quotes: list[WebullQuote] | None = None,
        disconnect_after: int | None = None,
    ) -> None:
        self._quote = quote
        self._candles = candles or []
        self._auth_success = auth_success
        self._fail_snapshot = fail_snapshot
        self._fail_candles = fail_candles
        self._stream_quotes = stream_quotes
        self._disconnect_after = disconnect_after
        self._stream_count = 0
        self.authenticate_calls = 0
        self.snapshot_calls = 0

    def authenticate(self) -> WebullAuthResult:
        self.authenticate_calls += 1
        if not self._auth_success:
            return WebullAuthResult(success=False, message="Mock auth failed")
        return WebullAuthResult(success=True, token_present=True, message="Mock authenticated")

    def get_snapshot(self, symbol: str) -> WebullQuote:
        _ensure_spy(symbol)
        self.snapshot_calls += 1
        if self._fail_snapshot:
            raise self._fail_snapshot
        if self._quote is None:
            raise WebullApiError("No mock quote configured")
        return self._quote

    def get_candles(self, symbol: str, timeframe: str) -> list[WebullCandle]:
        _ensure_spy(symbol)
        if self._fail_candles:
            raise self._fail_candles
        return list(self._candles)

    def stream_quotes(
        self,
        symbol: str,
        *,
        duration_seconds: float,
        poll_interval_seconds: float = 1.0,
    ) -> Iterator[WebullQuote]:
        _ensure_spy(symbol)
        if self._stream_quotes is not None:
            for quote in self._stream_quotes:
                self._stream_count += 1
                if self._disconnect_after is not None and self._stream_count > self._disconnect_after:
                    raise WebullApiError("Stream disconnected", error_code="DISCONNECT")
                yield quote
            return
        deadline = time.monotonic() + max(duration_seconds, 0)
        while time.monotonic() < deadline:
            yield self.get_snapshot(symbol)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_seconds, remaining))

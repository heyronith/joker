"""Shared Webull market-data loop for paper and LIVE_GATED runners.

Owns stock/options provider construction, warmup, and one poll cycle:

    quote → 0DTE surface → ingest → MarketRuntime.tick()

Callers supply the Task-1 bridge (CompatibilityLivePaperBridge). This module
does not place orders.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable

from joker.config.settings import AppSettings, EnvSettings
from joker.data.webull_capability import capability_usable_for_shadow
from joker.data.webull_market_provider import WebullMarketDataProvider
from joker.data.webull_options_provider import (
    WebullOptionsDataProvider,
    create_webull_options_provider,
)
from joker.schemas.replay import SpyQuoteEvent

logger = logging.getLogger(__name__)

LogFn = Callable[[str, dict[str, Any]], None]


def _default_log(event: str, payload: dict[str, Any]) -> None:
    logger.info(event, extra=payload)


class LiveMarketDataError(Exception):
    """Market-data loop failed closed."""


@dataclass
class LiveMarketDataLoop:
    """Real Webull observation loop shared by LivePaperRunner and LiveTradingRunner."""

    app_settings: AppSettings
    env: EnvSettings
    stock_api: Any | None = None
    options_api: Any | None = None
    require_options: bool = False
    source_label: str = "live_market"
    log: LogFn = field(default=_default_log)

    _provider: WebullMarketDataProvider | None = field(default=None, init=False, repr=False)
    _options: WebullOptionsDataProvider | None = field(default=None, init=False, repr=False)
    _http_clients: list[Any] = field(default_factory=list, init=False, repr=False)
    last_surface_complete: bool = field(default=False, init=False)
    observations_received: int = field(default=0, init=False)

    def build_providers(self) -> None:
        """Construct stock + options providers (idempotent)."""
        if self._provider is not None:
            return
        provider = WebullMarketDataProvider(
            self.env,
            api=self.stock_api,
            quote_max_age_seconds=self.app_settings.risk.quote_max_age_seconds,
            feed_max_silence_seconds=self.app_settings.risk.feed_max_silence_seconds,
            allow_delayed_quotes=self.app_settings.risk.allow_delayed_quotes,
            poll_interval_seconds=self.app_settings.data.quote_poll_interval_seconds,
        )
        self._provider = provider
        self._http_clients.append(provider)

        try:
            options = create_webull_options_provider(
                self.env,
                api=self.options_api,
                app_settings=self.app_settings,
            )
            self._http_clients.append(options)
            self._options = options
        except Exception as exc:
            self.log("options.unavailable", {"reason": str(exc)})
            if self.require_options:
                raise LiveMarketDataError(f"options_provider_failed: {exc}") from exc
            self._options = None

    def authenticate(self) -> None:
        """Authenticate stock (required) and options (best-effort) providers."""
        self.build_providers()
        assert self._provider is not None
        ok = self._provider.authenticate()
        self.log("webull.auth.result", {"success": ok, "source": self.source_label})
        if not ok:
            raise LiveMarketDataError("Webull market-data authentication failed")
        if self._options is not None:
            try:
                self._options.authenticate()
                if capability_usable_for_shadow():
                    self._options.verified = True
                self.log(
                    "options.capability",
                    {
                        "usable": capability_usable_for_shadow(),
                        "verified": getattr(self._options, "verified", False),
                    },
                )
            except Exception as exc:
                self.log("options.unavailable", {"reason": str(exc)})
                if self.require_options:
                    raise LiveMarketDataError(f"options_provider_failed: {exc}") from exc
                self._options = None

    @property
    def provider(self) -> WebullMarketDataProvider:
        if self._provider is None:
            raise LiveMarketDataError("providers not built; call authenticate() first")
        return self._provider

    @property
    def options_provider(self) -> WebullOptionsDataProvider | None:
        return self._options

    def _fetch_warmup_snapshot(self) -> Any:
        provider = self.provider
        try:
            candle_events = provider.fetch_candle_events("1m")
            self.log(
                "market.candles_loaded",
                {"count": len(candle_events), "source": "webull_stock"},
            )
        except Exception as candle_exc:
            self.log(
                "market.candles_unavailable",
                {
                    "reason": str(candle_exc),
                    "fallback": "quote_derived_candles_for_features_only",
                },
            )
        snapshot_event = provider.fetch_snapshot_event()
        snap0 = provider.get_latest_snapshot()
        if snap0 is not None and not snap0.candles:
            provider.append_quote_as_candle(snapshot_event)
        snapshot = provider.get_latest_snapshot()
        if snapshot is None:
            raise LiveMarketDataError("No SPY snapshot from Webull")
        return snapshot

    def warmup(self, bridge: Any) -> Any:
        """Sync warmup for paper runner (bridge.run_coro path)."""
        try:
            snapshot = self._fetch_warmup_snapshot()
            try:
                bridge.ingest_underlying_quote(
                    symbol=snapshot.symbol,
                    last=Decimal(str(snapshot.price)),
                    bid=Decimal(str(snapshot.bid)) if snapshot.bid is not None else None,
                    ask=Decimal(str(snapshot.ask)) if snapshot.ask is not None else None,
                    source_timestamp=snapshot.timestamp,
                    received_timestamp=datetime.now(timezone.utc),
                    source=f"{self.source_label}_warmup",
                )
            except Exception as ingest_exc:
                self.log(
                    "task1.market_ingest_failed",
                    {
                        "reason": str(ingest_exc),
                        "degraded": getattr(
                            getattr(bridge, "health", None), "degraded", None
                        ),
                    },
                )
            self.observations_received += 1
            self.log(
                "market.warmup",
                {
                    "candles": len(snapshot.candles),
                    "price": snapshot.price,
                    "delayed": self.provider.last_quote_delayed,
                    "feed_health": self.provider.feed_health,
                    "source": self.source_label,
                },
            )
            return snapshot
        except LiveMarketDataError:
            raise
        except Exception as exc:
            raise LiveMarketDataError(f"warmup_failed: {exc}") from exc

    async def awarmup(self, bridge: Any) -> Any:
        """Async warmup — await MarketRuntime directly (safe inside running loop)."""
        try:
            snapshot = self._fetch_warmup_snapshot()
            try:
                await self._a_ingest_underlying(
                    bridge,
                    symbol=snapshot.symbol,
                    last=Decimal(str(snapshot.price)),
                    bid=Decimal(str(snapshot.bid)) if snapshot.bid is not None else None,
                    ask=Decimal(str(snapshot.ask)) if snapshot.ask is not None else None,
                    source_timestamp=snapshot.timestamp,
                    source=f"{self.source_label}_warmup",
                )
            except Exception as ingest_exc:
                self.log(
                    "task1.market_ingest_failed",
                    {
                        "reason": str(ingest_exc),
                        "degraded": getattr(
                            getattr(bridge, "health", None), "degraded", None
                        ),
                    },
                )
            self.observations_received += 1
            self.log(
                "market.warmup",
                {
                    "candles": len(snapshot.candles),
                    "price": snapshot.price,
                    "delayed": self.provider.last_quote_delayed,
                    "feed_health": self.provider.feed_health,
                    "source": self.source_label,
                },
            )
            return snapshot
        except LiveMarketDataError:
            raise
        except Exception as exc:
            raise LiveMarketDataError(f"warmup_failed: {exc}") from exc

    def poll_once(self, bridge: Any) -> SpyQuoteEvent | None:
        """Sync poll for paper runner — uses bridge sync wrappers."""
        provider = self.provider
        try:
            event = provider.fetch_snapshot_event()
        except Exception as exc:
            self.log("provider.poll_error", {"reason": str(exc)})
            return None
        if not isinstance(event, SpyQuoteEvent):
            return None
        self.observations_received += 1
        try:
            bridge.ingest_underlying_quote(
                symbol=getattr(event, "symbol", None) or "SPY",
                last=Decimal(str(event.price)),
                bid=(
                    Decimal(str(event.bid))
                    if getattr(event, "bid", None) is not None
                    else None
                ),
                ask=(
                    Decimal(str(event.ask))
                    if getattr(event, "ask", None) is not None
                    else None
                ),
                source_timestamp=event.timestamp,
                received_timestamp=datetime.now(timezone.utc),
                source=f"{self.source_label}_poll",
            )
            self._sync_ingest_option_surface(bridge, float(event.price))
            bridge.tick()
        except Exception as ingest_exc:
            self.log(
                "task1.market_ingest_failed",
                {
                    "reason": str(ingest_exc),
                    "degraded": getattr(
                        getattr(bridge, "health", None), "degraded", None
                    ),
                },
            )
        try:
            provider.append_quote_as_candle(event)
        except Exception:
            pass
        return event

    async def apoll_once(self, bridge: Any) -> SpyQuoteEvent | None:
        """Async poll — await MarketRuntime directly (safe inside running loop)."""
        provider = self.provider
        try:
            event = provider.fetch_snapshot_event()
        except Exception as exc:
            self.log("provider.poll_error", {"reason": str(exc)})
            return None
        if not isinstance(event, SpyQuoteEvent):
            return None
        self.observations_received += 1
        try:
            await self._a_ingest_underlying(
                bridge,
                symbol=getattr(event, "symbol", None) or "SPY",
                last=Decimal(str(event.price)),
                bid=(
                    Decimal(str(event.bid))
                    if getattr(event, "bid", None) is not None
                    else None
                ),
                ask=(
                    Decimal(str(event.ask))
                    if getattr(event, "ask", None) is not None
                    else None
                ),
                source_timestamp=event.timestamp,
                source=f"{self.source_label}_poll",
            )
            await self._a_ingest_option_surface(bridge, float(event.price))
            market = bridge.supervisor.market_runtime
            if market is not None:
                result = await market.tick()
                if getattr(result, "snapshot", None) is not None:
                    ok = getattr(bridge, "_ingest_ok", None)
                    if callable(ok):
                        ok(result)
        except Exception as ingest_exc:
            self.log(
                "task1.market_ingest_failed",
                {
                    "reason": str(ingest_exc),
                    "degraded": getattr(
                        getattr(bridge, "health", None), "degraded", None
                    ),
                },
            )
        try:
            provider.append_quote_as_candle(event)
        except Exception:
            pass
        return event

    async def _a_ingest_underlying(
        self,
        bridge: Any,
        *,
        symbol: str,
        last: Decimal,
        bid: Decimal | None,
        ask: Decimal | None,
        source_timestamp: datetime,
        source: str,
    ) -> None:
        market = bridge.supervisor.market_runtime
        if market is None:
            raise LiveMarketDataError("market_runtime unavailable")
        await market.ingest_underlying_quote(
            symbol=symbol,
            last=last,
            bid=bid,
            ask=ask,
            source_timestamp=source_timestamp,
            received_timestamp=datetime.now(timezone.utc),
            source=source,
        )

    def _sync_ingest_option_surface(self, bridge: Any, underlying_price: float) -> None:
        rows, findings, complete = self._fetch_option_surface(bridge, underlying_price)
        if rows:
            bridge.ingest_option_quotes(rows)
        market = getattr(getattr(bridge, "supervisor", None), "market_runtime", None)
        if market is not None and findings:
            market.enqueue_quality_findings(findings)
        self.last_surface_complete = complete

    async def _a_ingest_option_surface(
        self, bridge: Any, underlying_price: float
    ) -> None:
        rows, findings, complete = self._fetch_option_surface(bridge, underlying_price)
        market = getattr(getattr(bridge, "supervisor", None), "market_runtime", None)
        if rows and market is not None:
            await market.ingest_option_quotes(rows)
        if market is not None and findings:
            market.enqueue_quality_findings(findings)
        self.last_surface_complete = complete

    def _fetch_option_surface(
        self, bridge: Any, underlying_price: float
    ) -> tuple[list[Any], list[Any], bool]:
        if self._options is None:
            return [], [], False
        try:
            from joker.runtime.option_surface_ingest import (
                convert_option_snapshots_to_surface_rows,
            )

            trading_day = None
            try:
                trading_day = bridge.supervisor.clock.trading_date()
            except Exception:
                trading_day = None
            fetch = self._options.fetch_surface_snapshots(
                underlying_price,
                trading_date=trading_day,
                max_contracts=None,
            )
            conversion = convert_option_snapshots_to_surface_rows(
                fetch.snapshots,
                trading_date=fetch.trading_date,
            )
            rows = list(conversion.rows)
            findings = list(fetch.to_data_quality_findings())
            findings.extend(conversion.to_data_quality_findings())
            if fetch.complete and conversion.converted_count != fetch.selected_count:
                from joker.market.quality import (
                    DataQualityCode,
                    DataQualityFinding,
                    DataQualitySeverity,
                )

                findings.append(
                    DataQualityFinding(
                        code=DataQualityCode.PARTIAL_OPTION_SURFACE,
                        severity=DataQualitySeverity.ERROR,
                        message=(
                            "persisted option rows differ from "
                            "selected SPY 0DTE contract count"
                        ),
                        symbol="SPY",
                        details={
                            "selected_count": fetch.selected_count,
                            "persisted_rows": len(rows),
                            "fetched_count": fetch.fetched_count,
                        },
                    )
                )
            complete = bool(fetch.complete and conversion.complete)
            self.log(
                "task1.option_surface_ingested",
                {
                    "contract_count": len(rows),
                    "complete": complete,
                    "source": self.source_label,
                },
            )
            return rows, findings, complete
        except Exception as opt_exc:
            self.log(
                "task1.option_surface_ingest_failed",
                {"reason": str(opt_exc)},
            )
            from joker.market.quality import (
                DataQualityCode,
                DataQualityFinding,
                DataQualitySeverity,
            )

            findings = [
                DataQualityFinding(
                    code=DataQualityCode.OPTION_SURFACE_UNAVAILABLE,
                    severity=DataQualitySeverity.ERROR,
                    message=(
                        "option surface fetch failed; "
                        "do not treat the previous surface as current "
                        "complete SPY 0DTE chain"
                    ),
                    symbol="SPY",
                    details={"reason": str(opt_exc)[:500]},
                )
            ]
            return [], findings, False

    async def run(
        self,
        bridge: Any,
        *,
        poll_interval_seconds: float | None = None,
        duration_seconds: float | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Async poll loop until duration, stop_event, or cancellation."""
        poll = max(
            0.5,
            float(
                poll_interval_seconds
                if poll_interval_seconds is not None
                else self.app_settings.data.quote_poll_interval_seconds
            ),
        )
        deadline = (
            time.monotonic() + max(duration_seconds, poll)
            if duration_seconds is not None
            else None
        )
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            await self.apoll_once(bridge)
            try:
                await asyncio.sleep(poll)
            except asyncio.CancelledError:
                raise

    def close(self) -> None:
        for client in self._http_clients:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._http_clients.clear()
        self._provider = None
        self._options = None

"""Data-only SPY watch runtime (shadow/paper features, no broker execution)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.config.settings import AppSettings, EnvSettings
from joker.data.provider import MarketDataProvider
from joker.data.provider_factory import ProviderKind, create_market_provider
from joker.data.webull_market_provider import WebullMarketDataProvider
from joker.features.engine import FeatureEngine
from joker.logging.event_log import EventLogWriter
from joker.runtime.run_manager import RunManager
from joker.schemas.domain import RiskConfig
from joker.compliance.data_classification import DataClassification
from joker.compliance.opra_sanitizer import snapshot_to_safe_metadata
from joker.schemas.replay import SpyCandleEvent, SpyQuoteEvent
from joker.storage.database import Database, ensure_database


@dataclass
class WatchRunConfig:
    symbol: str = "SPY"
    provider: str = "webull"
    shadow: bool = True
    duration_seconds: float | None = None
    webull_api: Any | None = None
    webull_options_api: Any | None = None
    use_options: bool = True


@dataclass
class WatchRunResult:
    run_id: str
    events_processed: int = 0
    features_updated: int = 0
    stale_warnings: int = 0
    errors: list[str] = field(default_factory=list)
    feed_health: str = "OK"
    last_price: float | None = None
    options_available: bool = False
    options_verified: bool = False
    selected_call_id: str | None = None
    selected_put_id: str | None = None
    call_bid: float | None = None
    call_ask: float | None = None
    call_mid: float | None = None
    call_spread_pct: float | None = None
    put_bid: float | None = None
    put_ask: float | None = None
    put_mid: float | None = None
    put_spread_pct: float | None = None
    option_quote_timestamp: str | None = None
    options_unavailable_fields: list[str] = field(default_factory=list)


class WatchRunner:
    """Stream real SPY data, compute features, log events — no broker orders."""

    def __init__(
        self,
        app_settings: AppSettings,
        env_settings: EnvSettings | None = None,
        db: Database | None = None,
        event_log: EventLogWriter | None = None,
    ) -> None:
        self.app_settings = app_settings
        self.env_settings = env_settings
        self.db = db or ensure_database(app_settings.db_path)
        self.event_log = event_log or EventLogWriter(
            app_settings.event_log_dir,
            redact_keys=app_settings.logging.redact_env_keys,
        )

    def _log(self, run_id: str, event_type: str, payload: dict) -> None:
        self.event_log.append(
            run_id=run_id,
            mode=self.app_settings.mode.value,
            source="watch",
            event_type=event_type,
            payload=payload,
        )

    def run(
        self,
        config: WatchRunConfig,
        *,
        on_state: Callable[[dict[str, Any]], None] | None = None,
    ) -> WatchRunResult:
        if config.symbol.upper() != "SPY":
            raise ValueError("Only SPY is supported in watch mode")

        mode = SafetyMode.SHADOW if config.shadow else self.app_settings.mode
        if mode.allows_broker_submit(self.app_settings.live_trading_enabled):
            raise RuntimeError(
                "Watch mode cannot run with live broker execution enabled"
            )

        run_manager = RunManager(self.db, self.event_log, self.app_settings)
        trading_day = date.today()
        run_id = run_manager.start_run(trading_day=trading_day)

        provider = create_market_provider(
            config.provider,
            app_settings=self.app_settings,
            env_settings=self.env_settings,
            webull_api=config.webull_api,
        )

        self._log(
            run_id,
            "provider.selected",
            {
                "provider": config.provider,
                "symbol": config.symbol,
                "shadow": config.shadow,
                "market_data_only": True,
                "broker_execution": False,
            },
        )

        if isinstance(provider, WebullMarketDataProvider):
            try:
                ok = provider.authenticate()
                self._log(
                    run_id,
                    "webull.auth.result",
                    {"success": ok, "feed_health": provider.feed_health},
                )
            except Exception as exc:
                msg = str(exc)
                self._log(run_id, "provider.error", {"error": msg})
                self._log(run_id, "replay.failure", {"error": msg})
                run_manager.end_run(run_id)
                return WatchRunResult(run_id=run_id, errors=[msg], feed_health="ERROR")

        feature_engine = FeatureEngine(
            max_age_seconds=self.app_settings.risk.quote_max_age_seconds
        )
        result = WatchRunResult(run_id=run_id)
        options_provider = None
        capability_usable = False
        if config.use_options and config.provider == ProviderKind.WEBULL.value and self.env_settings is not None:
            from joker.data.webull_capability import capability_usable_for_shadow

            capability_usable = capability_usable_for_shadow()
        if (
            config.use_options
            and config.provider == ProviderKind.WEBULL.value
            and self.env_settings is not None
            and self.env_settings.webull_market_data_enabled
        ):
            try:
                from joker.data.webull_options_provider import create_webull_options_provider

                options_provider = create_webull_options_provider(
                    self.env_settings,
                    api=config.webull_options_api,
                    app_settings=self.app_settings,
                )
                if capability_usable:
                    options_provider.verified = True
                    result.options_verified = True
                    result.options_available = True
                self._log(
                    run_id,
                    "options.capability",
                    {"usable_for_shadow": capability_usable, "source": "cache"},
                )
            except Exception as exc:
                self._log(run_id, "options.unavailable", {"reason": str(exc)})

        if config.duration_seconds and isinstance(provider, WebullMarketDataProvider):
            try:
                provider.prepare_stream(duration_seconds=config.duration_seconds)
            except Exception as exc:
                msg = str(exc)
                result.errors.append(msg)
                self._log(run_id, "provider.error", {"error": msg})
                run_manager.end_run(run_id)
                return result

        def push_state(extra: dict[str, Any]) -> None:
            if on_state is None:
                return
            base = {
                "provider": config.provider,
                "data_mode": "live" if config.provider == ProviderKind.WEBULL.value else config.provider,
                "market_price": result.last_price,
                "feed_health": result.feed_health,
                "options_available": result.options_available,
                "broker_execution_enabled": False,
                "market_data_only": True,
            }
            base.update(extra)
            on_state(base)

        try:
            for event in provider.stream_events():
                if config.duration_seconds and result.events_processed >= 10000:
                    break
                result.events_processed += 1

                if isinstance(event, SpyQuoteEvent):
                    self._log(
                        run_id,
                        "market.quote",
                        {
                            "event_id": event.event_id,
                            "symbol": event.symbol,
                            "source": event.source,
                            "data_classification": DataClassification.STOCK_MARKET_DATA.value,
                            "price": event.price,
                            "bid": event.bid,
                            "ask": event.ask,
                            "timestamp": event.timestamp.isoformat(),
                        },
                    )
                    result.last_price = event.price

                    if options_provider and result.last_price is not None and not result.options_verified:
                        try:
                            call_snap, put_snap = options_provider.fetch_atm_snapshots(
                                result.last_price
                            )
                            if call_snap and call_snap.bid and call_snap.ask:
                                result.options_available = True
                                result.options_verified = True
                                options_provider.verified = True
                                result.selected_call_id = call_snap.contract.contract_id
                                result.call_bid = call_snap.bid
                                result.call_ask = call_snap.ask
                                result.call_mid = call_snap.mid
                                result.call_spread_pct = call_snap.spread_pct
                                if call_snap.quote_timestamp:
                                    result.option_quote_timestamp = (
                                        call_snap.quote_timestamp.isoformat()
                                    )
                                result.options_unavailable_fields = (
                                    call_snap.field_availability.unavailable_fields()
                                )
                            if put_snap and put_snap.bid and put_snap.ask:
                                result.options_available = True
                                result.selected_put_id = put_snap.contract.contract_id
                                result.put_bid = put_snap.bid
                                result.put_ask = put_snap.ask
                                result.put_mid = put_snap.mid
                                result.put_spread_pct = put_snap.spread_pct
                            if result.options_verified:
                                call_safe = (
                                    snapshot_to_safe_metadata(call_snap, underlying_price=result.last_price)
                                    if call_snap
                                    else None
                                )
                                put_safe = (
                                    snapshot_to_safe_metadata(put_snap, underlying_price=result.last_price)
                                    if put_snap
                                    else None
                                )
                                self._log(
                                    run_id,
                                    "options.snapshots",
                                    {
                                        "verified": True,
                                        "call": call_safe,
                                        "put": put_safe,
                                    },
                                )
                                try:
                                    from joker.data.webull_capability import (
                                        WebullOptionsCapability,
                                        save_capability,
                                    )

                                    save_capability(
                                        WebullOptionsCapability(
                                            checked_at=datetime.now(timezone.utc),
                                            symbol=config.symbol,
                                            auth_pass=True,
                                            snapshot_succeeded=True,
                                            bid_ask_available=True,
                                            timestamp_available=True,
                                            same_day_expiration_found=True,
                                            usable_for_shadow=True,
                                            usable_for_replay_capture=True,
                                        )
                                    )
                                except Exception:
                                    pass
                            else:
                                self._log(
                                    run_id,
                                    "options.unavailable",
                                    {"reason": "required bid/ask missing"},
                                )
                        except Exception as exc:
                            self._log(run_id, "options.unavailable", {"reason": str(exc)})

                elif isinstance(event, SpyCandleEvent):
                    self._log(
                        run_id,
                        "market.candle",
                        {
                            "event_id": event.event_id,
                            "source": event.source,
                            "close": event.candle.close,
                        },
                    )

                snapshot = provider.get_latest_snapshot()
                if snapshot is None:
                    self._log(run_id, "market.missing_data", {"event_id": event.event_id})
                    continue

                features = feature_engine.compute(
                    snapshot,
                    reference_time=provider.current_time,
                )
                result.features_updated += 1
                self._log(
                    run_id,
                    "feature.updated",
                    {
                        "trend": features.trend_label,
                        "vwap_dist": features.distance_from_vwap_pct,
                    },
                )

                if isinstance(provider, WebullMarketDataProvider):
                    result.feed_health = provider.feed_health
                    if not capability_usable:
                        result.options_available = provider.options_available
                    if provider.feed_health == "STALE":
                        result.stale_warnings += 1
                        self._log(
                            run_id,
                            "market.stale_warning",
                            {"timestamp": event.timestamp.isoformat()},
                        )
                    if provider.permission_warning:
                        self._log(
                            run_id,
                            "webull.subscription.result",
                            {"warning": provider.permission_warning},
                        )

                push_state(
                    {
                        "market_bid": snapshot.bid,
                        "market_ask": snapshot.ask,
                        "last_update": event.timestamp.isoformat(),
                        "permission_warning": (
                            provider.permission_warning
                            if isinstance(provider, WebullMarketDataProvider)
                            else None
                        ),
                    }
                )

                if config.duration_seconds is None and result.events_processed >= 1:
                    break
        except Exception as exc:
            msg = str(exc)
            result.errors.append(msg)
            result.feed_health = "ERROR"
            self._log(run_id, "provider.disconnect", {"error": msg})
            self._log(run_id, "provider.error", {"error": msg})

        self._log(
            run_id,
            "watch.completed",
            {
                "events_processed": result.events_processed,
                "features_updated": result.features_updated,
                "feed_health": result.feed_health,
                "options_available": result.options_available,
            },
        )
        run_manager.end_run(run_id)
        return result

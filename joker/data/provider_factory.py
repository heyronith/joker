"""Market data provider factory."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from joker.config.settings import AppSettings, EnvSettings
from joker.data.live_mock_provider import MockMarketDataProvider
from joker.data.mock_provider import mock_spy_snapshot
from joker.data.provider import MarketDataProvider
from joker.data.replay_provider import ReplayMarketDataProvider
from joker.data.webull_config import WebullMarketConfigError, validate_webull_market_env
from joker.data.webull_market_provider import WebullMarketDataProvider


class ProviderKind(str, Enum):
    MOCK = "mock"
    REPLAY = "replay"
    WEBULL = "webull"

    @classmethod
    def from_string(cls, value: str) -> ProviderKind:
        normalized = value.strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise WebullMarketConfigError(
            f"Unknown data provider {value!r}. Choose: mock, replay, webull"
        )


PROVIDER_DESCRIPTIONS = {
    ProviderKind.MOCK: "Deterministic in-memory SPY snapshot (offline)",
    ProviderKind.REPLAY: "JSONL replay file (synthetic or recorded)",
    ProviderKind.WEBULL: "Webull OpenAPI SPY stock quotes (market-data only)",
}


def list_providers() -> list[tuple[str, str]]:
    return [(m.value, PROVIDER_DESCRIPTIONS[m]) for m in ProviderKind]


def create_market_provider(
    kind: str | ProviderKind,
    *,
    app_settings: AppSettings,
    env_settings: EnvSettings | None = None,
    replay_path: Path | None = None,
    webull_api: object | None = None,
) -> MarketDataProvider:
    provider_kind = kind if isinstance(kind, ProviderKind) else ProviderKind.from_string(kind)

    if provider_kind is ProviderKind.MOCK:
        snapshot = mock_spy_snapshot()
        return MockMarketDataProvider(snapshot=snapshot)

    if provider_kind is ProviderKind.REPLAY:
        if replay_path is None:
            raise WebullMarketConfigError("replay provider requires replay_path")
        return ReplayMarketDataProvider.from_file(str(replay_path))

    if provider_kind is ProviderKind.WEBULL:
        if env_settings is None:
            raise WebullMarketConfigError(
                "env_settings required for webull provider"
            )
        validate_webull_market_env(env_settings)
        return WebullMarketDataProvider(
            env_settings,
            api=webull_api,  # type: ignore[arg-type]
            quote_max_age_seconds=app_settings.risk.quote_max_age_seconds,
            feed_max_silence_seconds=app_settings.risk.feed_max_silence_seconds,
            allow_delayed_quotes=app_settings.risk.allow_delayed_quotes,
            poll_interval_seconds=app_settings.data.quote_poll_interval_seconds,
        )

    raise WebullMarketConfigError(f"Unsupported provider: {provider_kind}")

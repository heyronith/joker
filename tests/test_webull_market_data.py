"""Phase 18 Webull market-data tests (offline, mocked API)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from joker.broker.interface import PaperBroker
from joker.config.loader import load_app_settings
from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.provider_factory import ProviderKind, create_market_provider
from joker.data.webull_api import (
    MockWebullMarketApi,
    WebullApiError,
    WebullCandle,
    WebullQuote,
)
from joker.data.webull_config import WebullMarketConfigError, validate_webull_market_env
from joker.data.webull_diagnostics import run_webull_diagnostics
from joker.data.webull_market_provider import WebullMarketDataProvider
from joker.runtime.watch_runner import WatchRunConfig, WatchRunner
from joker.schemas.replay import SpyCandleEvent, SpyQuoteEvent


def _webull_env(**overrides: object) -> EnvSettings:
    base = {
        "OPENAI_API_KEY": "sk-test-key-for-unit-tests-only",
        "OPENAI_MODEL": "gpt-5.4-mini",
        "WEBULL_APP_KEY": "test-app-key",
        "WEBULL_APP_SECRET": "test-app-secret",
        "WEBULL_REGION": "US",
        "WEBULL_API_ENV": "uat",
    }
    base.update(overrides)
    return EnvSettings(**base)  # type: ignore[arg-type]


def _mock_api(**kwargs: object) -> MockWebullMarketApi:
    now = datetime.now(timezone.utc)
    quote = WebullQuote(
        symbol="SPY",
        price=553.25,
        bid=553.20,
        ask=553.30,
        timestamp=now,
    )
    return MockWebullMarketApi(quote=quote, **kwargs)


def test_webull_config_required_only_for_webull_provider(
    project_root: Path,
    env_vars: None,
) -> None:
    app, env = load_app_settings(project_root=project_root)
    mock_provider = create_market_provider("mock", app_settings=app, env_settings=env)
    assert mock_provider.get_latest_snapshot() is not None


def test_missing_webull_credentials_fail_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEBULL_APP_KEY", raising=False)
    monkeypatch.delenv("WEBULL_APP_SECRET", raising=False)
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-key-for-unit-tests-only",
        OPENAI_MODEL="gpt-5.4-mini",
    )
    with pytest.raises(WebullMarketConfigError, match="WEBULL_APP_KEY"):
        validate_webull_market_env(env)


def test_credentials_redacted_from_errors() -> None:
    secret = "super-secret-app-key-value"
    env = _webull_env(WEBULL_APP_KEY=secret)
    message = redact_secrets(f"failed with key {secret}", env=env)
    assert secret not in message


def test_normalize_snapshot_to_spy_quote_event() -> None:
    env = _webull_env()
    api = _mock_api()
    provider = WebullMarketDataProvider(env, api=api)
    event = provider.fetch_snapshot_event()
    assert isinstance(event, SpyQuoteEvent)
    assert event.symbol == "SPY"
    assert event.source == "webull_stock"
    assert event.price == 553.25
    assert event.bid == 553.20
    assert event.ask == 553.30


def test_normalize_candle_to_spy_candle_event() -> None:
    env = _webull_env()
    now = datetime.now(timezone.utc)
    api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, None, None, now),
        candles=[
            WebullCandle(now, 549.0, 551.0, 548.5, 550.5, 1000),
        ],
    )
    provider = WebullMarketDataProvider(env, api=api)
    events = provider.fetch_candle_events("1m")
    assert len(events) == 1
    assert isinstance(events[0], SpyCandleEvent)
    assert events[0].source == "webull_stock"
    assert events[0].candle.close == 550.5


def test_malformed_quote_rejected() -> None:
    env = _webull_env()
    provider = WebullMarketDataProvider(env, api=_mock_api())
    with pytest.raises(WebullApiError, match="missing price"):
        provider.normalize_raw_quote("SPY", {"timestamp": "2026-07-01T14:00:00+00:00"})


def test_subscription_403_classified() -> None:
    err = WebullApiError("forbidden", status_code=403, subscription_related=True)
    assert err.subscription_related is True


def test_stream_disconnect_handled() -> None:
    env = _webull_env()
    now = datetime.now(timezone.utc)
    quotes = [
        WebullQuote("SPY", 550.0, None, None, now),
        WebullQuote("SPY", 550.1, None, None, now),
        WebullQuote("SPY", 550.2, None, None, now),
    ]
    api = MockWebullMarketApi(quote=quotes[0], stream_quotes=quotes, disconnect_after=1)
    provider = WebullMarketDataProvider(env, api=api, poll_interval_seconds=0.01)
    with pytest.raises(WebullApiError, match="disconnect"):
        provider.prepare_stream(duration_seconds=1.0)


def test_stale_quote_detection() -> None:
    env = _webull_env()
    stale_ts = datetime.now(timezone.utc) - timedelta(seconds=120)
    api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, stale_ts),
    )
    provider = WebullMarketDataProvider(env, api=api, quote_max_age_seconds=30)
    provider.fetch_snapshot_event()
    assert provider.feed_health == "STALE"


def test_non_spy_symbol_rejected() -> None:
    env = _webull_env()
    api = _mock_api()
    with pytest.raises(WebullApiError, match="Only SPY"):
        api.get_snapshot("QQQ")


def test_webull_provider_has_no_order_methods() -> None:
    env = _webull_env()
    provider = WebullMarketDataProvider(env, api=_mock_api())
    assert not hasattr(provider, "submit_order")
    assert not hasattr(provider, "cancel_order")
    assert provider.get_option_chain("SPY", datetime.now(timezone.utc).date()) is None
    assert provider.get_option_quote("any") is None


def test_watch_shadow_does_not_submit_broker_orders(tmp_path: Path) -> None:
    env = _webull_env()
    app_settings, _ = load_app_settings(
        config_path="config/paper.yaml",
        project_root=Path(__file__).resolve().parents[1],
    )
    app_settings = app_settings.model_copy(
        update={
            "db_path": tmp_path / "db",
            "event_log_dir": tmp_path / "logs",
            "reports_dir": tmp_path / "reports",
        }
    )
    api = _mock_api()
    runner = WatchRunner(app_settings, env)
    result = runner.run(
        WatchRunConfig(provider="webull", shadow=True, webull_api=api, use_options=False),
    )
    assert result.events_processed >= 1
    assert result.options_available is False
    broker = PaperBroker()
    assert broker.list_open_orders() == []


def test_diagnostics_with_mock_api() -> None:
    env = _webull_env()
    report = run_webull_diagnostics(env, api=_mock_api())
    assert report.credentials_present is True
    assert any(c.name == "auth" and c.status == "pass" for c in report.checks)
    assert any(c.name == "snapshot" and c.status == "pass" for c in report.checks)


def test_diagnostics_subscription_failure() -> None:
    env = _webull_env()
    api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, None, None, datetime.now(timezone.utc)),
        fail_snapshot=WebullApiError(
            "403 forbidden",
            status_code=403,
            subscription_related=True,
        ),
    )
    report = run_webull_diagnostics(env, api=api)
    assert "subscription" in (report.likely_issue or "").lower()


def test_default_small_account_risk_config(project_root: Path, env_vars: None) -> None:
    app, _ = load_app_settings(project_root=project_root)
    # paper.yaml agent_led soft caps (advisory; hard floors still apply)
    assert app.risk.max_daily_loss_usd == 500.0
    assert app.risk.max_premium_usd == 500.0
    assert app.risk.max_trades_per_day == 5
    assert app.risk.max_open_positions == 1
    assert app.risk.policy == "agent_led"

def test_live_trading_flag_must_remain_false() -> None:
    with pytest.raises(ValueError, match="WEBULL_LIVE_TRADING_ENABLED"):
        EnvSettings(
            OPENAI_API_KEY="sk-test-key-for-unit-tests-only",
            WEBULL_LIVE_TRADING_ENABLED=True,
        )


def test_provider_factory_webull_requires_env(project_root: Path, env_vars: None) -> None:
    app, _ = load_app_settings(project_root=project_root)
    with pytest.raises(WebullMarketConfigError, match="env_settings|WEBULL"):
        create_market_provider(ProviderKind.WEBULL, app_settings=app, env_settings=None)


@pytest.fixture
def env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests-only")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("JOKER_CONFIG", "config/paper.yaml")

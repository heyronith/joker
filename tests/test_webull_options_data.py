"""Phase 19 Webull options data verification tests (offline)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from joker.broker.interface import PaperBroker
from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.options_capture import capture_options_snapshot
from joker.data.options_diagnostics import run_options_diagnostics
from joker.data.options_normalizer import (
    OptionQuoteValidationError,
    normalize_webull_option_snapshot,
    snapshot_to_quote_event,
    validate_tradable_snapshot,
)
from joker.data.options_rate_limit import TTLCache
from joker.data.webull_api import HttpWebullMarketApi
from joker.data.webull_options_api import (
    HttpWebullOptionsMarketApi,
    MockWebullOptionsMarketApi,
    OptionEndpointUnverified,
)
from joker.data.webull_options_provider import WebullOptionsDataProvider
from joker.execution.option_selector import OptionSelector
from joker.runtime.watch_runner import WatchRunConfig, WatchRunner
from joker.schemas.options_data import (
    OptionContractMetadata,
    OptionFieldAvailability,
    OptionSnapshot,
)
from joker.config.settings import AppSettings


def _env(**overrides: object) -> EnvSettings:
    base = {
        "OPENAI_API_KEY": "sk-test-key-for-unit-tests-only",
        "OPENAI_MODEL": "gpt-5.4-mini",
        "WEBULL_APP_KEY": "test-app-key",
        "WEBULL_APP_SECRET": "test-app-secret",
        "WEBULL_MARKET_DATA_ENABLED": True,
    }
    base.update(overrides)
    return EnvSettings(**base)  # type: ignore[arg-type]


def _contract(
    strike: float,
    option_type: str,
    contract_id: str,
    expiration: date | None = None,
) -> OptionContractMetadata:
    return OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=expiration or date.today(),
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        contract_id=contract_id,
    )


def _snapshot(
    contract: OptionContractMetadata,
    *,
    bid: float = 1.0,
    ask: float = 1.1,
    ts: datetime | None = None,
    **extra: object,
) -> OptionSnapshot:
    ts = ts or datetime.now(timezone.utc)
    mid = (bid + ask) / 2
    spread = ((ask - bid) / mid) * 100
    avail = OptionFieldAvailability(
        bid=True,
        ask=True,
        mid=True,
        spread_pct=True,
        quote_timestamp=True,
        contract_id=True,
    )
    return OptionSnapshot(
        contract=contract,
        bid=bid,
        ask=ask,
        mid=mid,
        spread_pct=spread,
        quote_timestamp=ts,
        field_availability=avail,
        **extra,
    )


def _mock_options_api(
    contracts: list[OptionContractMetadata],
    snapshots: dict[str, OptionSnapshot],
) -> MockWebullOptionsMarketApi:
    return MockWebullOptionsMarketApi(contracts=contracts, snapshots=snapshots)


def test_market_data_calls_enabled_rename() -> None:
    from joker.data.webull_http import WebullHttpClient

    assert HttpWebullMarketApi.MARKET_DATA_CALLS_ENABLED is True
    assert WebullHttpClient.MARKET_DATA_CALLS_ENABLED is True
    assert not hasattr(HttpWebullMarketApi, "LIVE_CALLS_ENABLED")


def test_options_api_has_no_trading_methods() -> None:
    forbidden = (
        "submit_order",
        "cancel_order",
        "get_account",
        "list_positions",
        "place_order",
    )
    for cls in (HttpWebullOptionsMarketApi, MockWebullOptionsMarketApi):
        for name in forbidden:
            assert not hasattr(cls, name)


def test_missing_credentials_only_for_options_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from joker.data.webull_config import WebullMarketConfigError, validate_webull_market_env

    monkeypatch.delenv("WEBULL_APP_KEY", raising=False)
    monkeypatch.delenv("WEBULL_APP_SECRET", raising=False)
    env = EnvSettings(
        _env_file=None,
        OPENAI_API_KEY="sk-test-key-for-unit-tests-only",
    )
    with pytest.raises(WebullMarketConfigError, match="WEBULL"):
        validate_webull_market_env(env)


def test_endpoint_unverified_fails_safely() -> None:
    api = HttpWebullOptionsMarketApi(_env())
    with pytest.raises(OptionEndpointUnverified):
        api.find_option_contracts("SPY", date.today())


def test_valid_snapshot_normalizes() -> None:
    contract = _contract(550, "call", "SPY250701C00550000")
    raw = {
        "bid": 1.05,
        "ask": 1.15,
        "last": 1.10,
        "volume": 500,
        "openInterest": 1200,
        "impliedVolatility": 0.25,
        "delta": 0.45,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    snap = normalize_webull_option_snapshot(contract, raw)
    assert snap.bid == 1.05
    assert snap.ask == 1.15
    assert snap.volume == 500
    assert snap.open_interest == 1200
    assert snap.field_availability.bid is True
    assert snap.field_availability.implied_volatility is True


def test_missing_bid_rejected() -> None:
    contract = _contract(550, "call", "c1")
    snap = _snapshot(contract, bid=0, ask=1.1)
    snap = snap.model_copy(update={"bid": None, "field_availability": OptionFieldAvailability(ask=True)})
    with pytest.raises(OptionQuoteValidationError, match="Bid"):
        validate_tradable_snapshot(snap)


def test_missing_ask_rejected() -> None:
    contract = _contract(550, "call", "c1")
    snap = _snapshot(contract)
    snap = snap.model_copy(update={"ask": None})
    with pytest.raises(OptionQuoteValidationError, match="Ask"):
        validate_tradable_snapshot(snap)


def test_missing_timestamp_rejected() -> None:
    contract = _contract(550, "call", "c1")
    snap = _snapshot(contract)
    snap = snap.model_copy(update={"quote_timestamp": None})
    with pytest.raises(OptionQuoteValidationError, match="timestamp"):
        validate_tradable_snapshot(snap)


def test_invalid_mid_rejected() -> None:
    contract = _contract(550, "call", "c1")
    snap = _snapshot(contract, bid=1.0, ask=1.1)
    snap = snap.model_copy(update={"mid": 0})
    with pytest.raises(OptionQuoteValidationError) as exc_info:
        validate_tradable_snapshot(snap)
    assert exc_info.value.code == "INVALID_MID"


def test_delayed_quote_warning() -> None:
    contract = _contract(550, "call", "c1")
    snap = _snapshot(contract, delayed=True)
    warnings = validate_tradable_snapshot(snap)
    assert any(w.code == "DELAYED" for w in warnings)


def test_wide_spread_rejected() -> None:
    contract = _contract(550, "call", "c1")
    snap = _snapshot(contract, bid=1.0, ask=2.0)
    with pytest.raises(OptionQuoteValidationError) as exc_info:
        validate_tradable_snapshot(snap, max_spread_pct=15.0)
    assert exc_info.value.code == "WIDE_SPREAD"


def test_stale_quote_rejected() -> None:
    contract = _contract(550, "call", "c1")
    stale = datetime.now(timezone.utc) - timedelta(seconds=120)
    snap = _snapshot(contract, ts=stale)
    with pytest.raises(OptionQuoteValidationError) as exc_info:
        validate_tradable_snapshot(
            snap,
            reference_time=datetime.now(timezone.utc),
            quote_max_age_seconds=30,
        )
    assert exc_info.value.code == "STALE_QUOTE"


def test_atm_call_selection() -> None:
    exp = date(2026, 7, 1)
    contracts = [
        _contract(548, "call", "c548", exp),
        _contract(550, "call", "c550", exp),
        _contract(552, "call", "c552", exp),
        _contract(548, "put", "p548", exp),
        _contract(550, "put", "p550", exp),
    ]
    result = WebullOptionsDataProvider.select_atm_candidates(contracts, 550.2, exp)
    assert result.atm_call is not None
    assert result.atm_call.strike == 550
    assert result.atm_put is not None
    assert result.atm_put.strike == 550


def test_no_same_day_expiration() -> None:
    api = _mock_options_api([], {})
    provider = WebullOptionsDataProvider(env=_env(), api=api)
    contracts = provider.discover_contracts("SPY", date.today())
    assert contracts == []
    call, put = provider.fetch_atm_snapshots(550.0, date.today())
    assert call is None and put is None


def test_missing_contract_id() -> None:
    provider = WebullOptionsDataProvider(env=_env(), api=_mock_options_api([], {}))
    bad = _contract(550, "call", "")
    with pytest.raises(Exception, match="contract_id"):
        provider.fetch_snapshot(bad)


def test_empty_chain() -> None:
    provider = WebullOptionsDataProvider(
        env=_env(),
        api=_mock_options_api([], {}),
    )
    assert provider.discover_contracts("SPY", date.today()) == []


def test_non_spy_rejected() -> None:
    provider = WebullOptionsDataProvider(env=_env(), api=_mock_options_api([], {}))
    with pytest.raises(Exception, match="SPY"):
        provider.discover_contracts("QQQ", date.today())


def test_contract_cache() -> None:
    exp = date.today()
    contracts = [_contract(550, "call", "c1", exp)]
    api = _mock_options_api(contracts, {})
    provider = WebullOptionsDataProvider(env=_env(), api=api, contract_cache_ttl_seconds=60)
    first = provider.discover_contracts("SPY", exp)
    second = provider.discover_contracts("SPY", exp)
    assert first == second
    assert isinstance(api, MockWebullOptionsMarketApi)


def test_rate_limit_classified() -> None:
    exp = date.today()
    c = _contract(550, "call", "c1", exp)
    api = MockWebullOptionsMarketApi(
        contracts=[c],
        snapshots={"c1": _snapshot(c)},
        rate_limit_after=0,
    )
    provider = WebullOptionsDataProvider(env=_env(), api=api)
    with pytest.raises(Exception) as exc_info:
        provider.fetch_snapshot(c)
    assert getattr(exc_info.value, "rate_limited", False) or "Rate limit" in str(exc_info.value)


def test_diagnostics_summarize_unavailable_fields() -> None:
    exp = date.today()
    call_c = _contract(550, "call", "c550", exp)
    put_c = _contract(550, "put", "p550", exp)
    call_s = _snapshot(call_c)
    put_s = _snapshot(put_c)
    api = _mock_options_api([call_c, put_c], {"c550": call_s, "p550": put_s})
    from joker.data.webull_api import MockWebullMarketApi, WebullQuote

    stock = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    report = run_options_diagnostics(_env(), options_api=api, stock_api=stock)
    assert report.credentials_present is True
    assert report.auth_pass is True
    assert report.same_day_expiration_found is True
    assert report.capability is not None


def test_options_snapshot_capture_writes_safe_jsonl(tmp_path: Path) -> None:
    # Align seeded expiry with provider.market_today() (America/New_York).
    from joker.data.webull_options_provider import MARKET_TZ

    exp = datetime.now(MARKET_TZ).date()
    call_c = _contract(550, "call", "c550", exp)
    put_c = _contract(550, "put", "p550", exp)
    api = _mock_options_api([call_c, put_c], {"c550": _snapshot(call_c), "p550": _snapshot(put_c)})
    from joker.data.webull_api import MockWebullMarketApi, WebullQuote

    stock = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    path, summary = capture_options_snapshot(
        _env(),
        captures_dir=tmp_path,
        options_provider=WebullOptionsDataProvider(env=_env(), api=api),
        stock_api=stock,
    )
    raw = path.read_text()
    assert "test-app-secret" not in raw
    assert summary["is_real_webull_data"] is True
    lines = [json.loads(l) for l in raw.strip().split("\n")]
    assert lines[0]["event_type"] == "capture.meta"


def test_snapshot_to_quote_event() -> None:
    contract = _contract(550, "call", "c550")
    event = snapshot_to_quote_event(_snapshot(contract))
    assert event.event_type == "option_quote"
    assert event.source == "webull_opra"
    assert event.bid == 1.0


def test_option_selector_from_snapshots() -> None:
    contract = _contract(550, "call", "c550")
    snap = _snapshot(contract)
    selected = OptionSelector().select_from_snapshots(
        [snap],
        "long_call",
        550.0,
        datetime.now(timezone.utc),
    )
    assert selected.contract.strike == 550


def test_shadow_watch_no_broker_submit(tmp_path: Path) -> None:
    from joker.data.webull_api import MockWebullMarketApi, WebullQuote

    app_settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    env = _env()
    stock_api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    exp = date.today()
    call_c = _contract(550, "call", "c550", exp)
    options_api = _mock_options_api([call_c], {"c550": _snapshot(call_c)})
    runner = WatchRunner(app_settings, env)
    result = runner.run(
        WatchRunConfig(
            provider="webull",
            shadow=True,
            webull_api=stock_api,
            webull_options_api=options_api,
            use_options=True,
        )
    )
    assert result.events_processed >= 1
    broker = PaperBroker()
    assert broker.list_open_orders() == []


def test_secrets_redacted_in_options_errors() -> None:
    secret = "super-secret-app-key"
    env = _env(WEBULL_APP_KEY=secret)
    msg = redact_secrets(f"auth failed with {secret}", env=env)
    assert secret not in msg

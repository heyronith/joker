"""Phase 20 Webull endpoint audit tests (offline)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from joker.config.settings import EnvSettings
from joker.config.validation import redact_secrets
from joker.data.webull_capability import (
    WebullOptionsCapability,
    capability_usable_for_shadow,
    load_capability,
    save_capability,
)
from joker.data.webull_endpoints import WEBULL_ENDPOINTS, get_endpoint, require_verified
from joker.data.webull_errors import OptionEndpointUnverified
from joker.data.webull_http import WebullHttpClient
from joker.data.webull_option_symbols import build_osi_symbol, metadata_from_osi
from joker.data.webull_options_api import HttpWebullOptionsMarketApi, MockWebullOptionsMarketApi
from joker.data.webull_options_provider import WebullOptionsDataProvider
from joker.data.webull_response_capture import summarize_response, write_contract_capture
from joker.data.webull_verification import build_capability_from_report, generate_verification_report
from joker.data.options_diagnostics import run_options_diagnostics
from joker.data.webull_api import MockWebullMarketApi, WebullQuote
from joker.runtime.watch_runner import WatchRunConfig, WatchRunner
from joker.config.settings import AppSettings
from joker.schemas.options_data import (
    OptionContractMetadata,
    OptionDataDiagnosticReport,
    OptionSnapshot,
    OptionFieldAvailability,
)


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


def _contract(strike: float, option_type: str, contract_id: str, expiration: date | None = None):
    return OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=expiration or date.today(),
        strike=strike,
        option_type=option_type,  # type: ignore[arg-type]
        contract_id=contract_id,
    )


def _snapshot(contract: OptionContractMetadata, **kw) -> OptionSnapshot:
    bid = kw.get("bid", 1.0)
    ask = kw.get("ask", 1.1)
    ts = kw.get("quote_timestamp", datetime.now(timezone.utc))
    return OptionSnapshot(
        contract=contract,
        bid=bid,
        ask=ask,
        mid=(bid + ask) / 2,
        quote_timestamp=ts,
        field_availability=OptionFieldAvailability(bid=True, ask=True, quote_timestamp=True, contract_id=True),
    )


def test_endpoint_registry_loads() -> None:
    assert "option_snapshot" in WEBULL_ENDPOINTS
    ep = get_endpoint("option_snapshot")
    assert ep.path == "/openapi/market-data/option/snapshot"
    assert ep.verified is True


def test_unverified_endpoint_raises() -> None:
    with pytest.raises(OptionEndpointUnverified, match="option_chain"):
        require_verified("option_chain")


def test_option_snapshot_uses_verified_path() -> None:
    ep = get_endpoint("option_snapshot")
    assert ep.path == "/openapi/market-data/option/snapshot"
    assert "symbols" in ep.required_params


def test_http_client_blocks_unverified_get() -> None:
    client = WebullHttpClient(_env())
    client.set_access_token("tok")
    WebullHttpClient.MARKET_DATA_CALLS_ENABLED = True
    with pytest.raises(OptionEndpointUnverified):
        client.request_json("option_chain", params={"symbol": "SPY"})


def test_find_option_contracts_raises_unverified() -> None:
    api = HttpWebullOptionsMarketApi(_env())
    with pytest.raises(OptionEndpointUnverified):
        api.find_option_contracts("SPY", date.today())


def test_resolve_osi_from_metadata() -> None:
    exp = date(2026, 7, 1)
    osi = build_osi_symbol("SPY", exp, 550.0, "call")
    assert osi == "SPY260701C00550000"
    meta = metadata_from_osi(osi)
    assert meta.strike == 550.0
    assert meta.option_type == "call"


def test_missing_snapshot_identifier_fails() -> None:
    api = HttpWebullOptionsMarketApi(_env())
    bad = OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=date.today(),
        strike=0,
        option_type="call",
        contract_id=None,
    )
    with pytest.raises(Exception, match="Missing required identifier"):
        api.get_option_snapshot(bad)


def test_resolve_instrument_id_for_snapshot() -> None:
    from joker.data.webull_options_api import _resolve_osi_symbol

    contract = OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=date.today(),
        strike=550.0,
        option_type="call",
        contract_id="not-osi-format",
        instrument_id="WB12345678",
    )
    assert _resolve_osi_symbol(contract) == "WB12345678"


def test_option_snapshot_uses_verified_endpoint_via_http() -> None:
    from joker.data.webull_options_api import HttpWebullOptionsMarketApi

    exp = date(2026, 7, 1)
    contract = OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=exp,
        strike=550.0,
        option_type="call",
        contract_id="SPY260701C00550000",
    )
    http = MagicMock()
    http.access_token = "tok"
    http.request_json.return_value = [
        {
            "symbol": "SPY260701C00550000",
            "bid": 1.0,
            "ask": 1.1,
            "quote_time": datetime.now(timezone.utc).isoformat(),
        }
    ]
    api = HttpWebullOptionsMarketApi(_env(), http_client=http)
    snap = api.get_option_snapshot(contract)
    http.request_json.assert_called_once_with(
        "option_snapshot",
        params={"symbols": "SPY260701C00550000", "category": "US_OPTION"},
    )
    assert snap.bid == 1.0


def test_empty_option_snapshot_response_fails() -> None:
    http = MagicMock()
    http.access_token = "tok"
    http.request_json.return_value = []
    api = HttpWebullOptionsMarketApi(_env(), http_client=http)
    contract = OptionContractMetadata(
        underlying_symbol="SPY",
        expiration=date.today(),
        strike=550.0,
        option_type="call",
        contract_id="SPY260701C00550000",
    )
    with pytest.raises(Exception, match="Empty option snapshot"):
        api.get_option_snapshot(contract)


def test_diagnostics_distinguish_unverified_vs_subscription() -> None:
    exp = date.today()
    call_id = build_osi_symbol("SPY", exp, 550, "call")
    put_id = build_osi_symbol("SPY", exp, 550, "put")
    call_c = _contract(550, "call", call_id, exp)
    put_c = _contract(550, "put", put_id, exp)
    api = MockWebullOptionsMarketApi(
        contracts=[],
        snapshots={
            call_c.contract_id: _snapshot(call_c),
            put_c.contract_id: _snapshot(put_c),
        },
    )
    stock = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    report = run_options_diagnostics(_env(), options_api=api, stock_api=stock)
    assert report.endpoint_status.get("option_chain", "").startswith("unverified") or "unverified" in str(
        report.endpoint_status.get("option_chain", "")
    )
    assert report.atm_call_snapshot_pass is True


def test_contract_discovery_false_without_api_discovery() -> None:
    provider = WebullOptionsDataProvider(
        env=_env(),
        api=MockWebullOptionsMarketApi(contracts=[], snapshots={}),
    )
    cap = provider.build_capability_report(
        None,
        None,
        contract_discovery_succeeded=False,
        auth_pass=True,
        same_day_expiration=False,
    )
    assert cap.contract_discovery is False


def test_capability_report_verified_requires_fields() -> None:
    exp = date.today()
    call_c = _contract(550, "call", "c1", exp)
    put_c = _contract(550, "put", "p1", exp)
    provider = WebullOptionsDataProvider(env=_env(), api=MockWebullOptionsMarketApi())
    cap = provider.build_capability_report(
        _snapshot(call_c),
        _snapshot(put_c),
        contract_discovery_succeeded=True,
        auth_pass=True,
        same_day_expiration=True,
    )
    assert cap.verified is True
    assert cap.contract_discovery is True


def test_response_capture_redacts_secrets(tmp_path: Path) -> None:
    env = _env()
    summary = summarize_response(
        endpoint_name="option_snapshot",
        status_code=200,
        payload={"bid": 1.0, "ask": 1.1},
    )
    path = write_contract_capture([summary], output_dir=tmp_path, env=env)
    raw = path.read_text()
    assert "test-app-secret" not in raw


def test_capability_cache_write_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from joker.data import webull_capability as cap_mod

    p = tmp_path / "cap.json"
    monkeypatch.setattr(cap_mod, "DEFAULT_CAPABILITY_PATH", p)
    cap = WebullOptionsCapability(
        checked_at=datetime.now(timezone.utc),
        usable_for_shadow=True,
        auth_pass=True,
        bid_ask_available=True,
        timestamp_available=True,
        same_day_expiration_found=True,
    )
    save_capability(cap, p)
    loaded = load_capability(p)
    assert loaded is not None
    assert loaded.usable_for_shadow is True
    assert capability_usable_for_shadow(p) is True


def test_verification_report_generated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from joker.data import webull_capability as cap_mod

    cap_path = tmp_path / "cap.json"
    monkeypatch.setattr(cap_mod, "DEFAULT_CAPABILITY_PATH", cap_path)

    exp = date.today()
    call_c = _contract(550, "call", "c550", exp)
    put_c = _contract(550, "put", "p550", exp)
    api = MockWebullOptionsMarketApi(
        contracts=[call_c, put_c],
        snapshots={"c550": _snapshot(call_c), "p550": _snapshot(put_c)},
    )
    stock = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    path, report, cap = generate_verification_report(
        _env(),
        reports_dir=tmp_path,
        stock_api=stock,
        options_api=api,
    )
    assert path.exists()
    text = path.read_text()
    assert "test-app-secret" not in text
    assert cap_path.exists()


def test_shadow_refuses_options_without_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from joker.data import webull_capability as cap_mod

    cap_path = tmp_path / "capabilities" / "webull_options_capability.json"
    cap_path.parent.mkdir(parents=True)
    cap_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "provider": "webull",
                "symbol": "SPY",
                "auth_pass": False,
                "usable_for_shadow": False,
            }
        )
    )
    monkeypatch.setattr(cap_mod, "DEFAULT_CAPABILITY_PATH", cap_path)

    app_settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    exp = date.today()
    call_c = _contract(550, "call", "c550", exp)
    options_api = MockWebullOptionsMarketApi(
        contracts=[call_c],
        snapshots={"c550": _snapshot(call_c)},
    )
    stock_api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    runner = WatchRunner(app_settings, _env())
    result = runner.run(
        WatchRunConfig(
            provider="webull",
            shadow=True,
            webull_api=stock_api,
            webull_options_api=options_api,
            use_options=True,
        )
    )
    assert result.options_verified is False or result.options_available is False


def test_shadow_allows_options_with_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from joker.data import webull_capability as cap_mod

    cap_path = tmp_path / "capabilities" / "webull_options_capability.json"
    cap_path.parent.mkdir(parents=True)
    save_capability(
        WebullOptionsCapability(
            checked_at=datetime.now(timezone.utc),
            usable_for_shadow=True,
            auth_pass=True,
            bid_ask_available=True,
            timestamp_available=True,
            same_day_expiration_found=True,
        ),
        cap_path,
    )
    monkeypatch.setattr(cap_mod, "DEFAULT_CAPABILITY_PATH", cap_path)

    app_settings = AppSettings.model_validate(
        {
            "db_path": str(tmp_path / "db"),
            "event_log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        }
    )
    exp = date.today()
    call_c = _contract(550, "call", "c550", exp)
    options_api = MockWebullOptionsMarketApi(
        contracts=[call_c],
        snapshots={"c550": _snapshot(call_c)},
    )
    stock_api = MockWebullMarketApi(
        quote=WebullQuote("SPY", 550.0, 549.9, 550.1, datetime.now(timezone.utc))
    )
    runner = WatchRunner(app_settings, _env())
    result = runner.run(
        WatchRunConfig(
            provider="webull",
            shadow=True,
            webull_api=stock_api,
            webull_options_api=options_api,
            use_options=True,
        )
    )
    assert result.options_verified is True
    assert result.options_available is True


def test_malformed_response_classified() -> None:
    client = WebullHttpClient(_env())
    err = client.classify_http_error(400, "invalid symbol", endpoint=get_endpoint("option_snapshot"))
    assert err.error_code == "MALFORMED_RESPONSE"


def test_endpoint_mismatch_classified() -> None:
    client = WebullHttpClient(_env())
    err = client.classify_http_error(404, "not found", endpoint=get_endpoint("option_snapshot"))
    assert err.error_code == "ENDPOINT_MISMATCH"


def test_market_data_calls_enabled_on_http_client() -> None:
    assert WebullHttpClient.MARKET_DATA_CALLS_ENABLED is True
    prev = WebullHttpClient.MARKET_DATA_CALLS_ENABLED
    WebullHttpClient.MARKET_DATA_CALLS_ENABLED = False
    try:
        client = WebullHttpClient(_env())
        client.set_access_token("tok")
        with pytest.raises(Exception, match="market-data calls are disabled"):
            client.request_json("stock_snapshot", params={"symbols": "SPY", "category": "US_STOCK"})
    finally:
        WebullHttpClient.MARKET_DATA_CALLS_ENABLED = prev


def test_webull_signature_matches_documented_example() -> None:
    from joker.data.webull_auth import build_signature

    body = '{"k1":123,"k2":"this is the api request body","k3":true,"k4":{"foo":[1,2]}}'
    signature = build_signature(
        app_secret="0f50a2e853334a9aae1a783bee120c1f",
        path="/trade/place_order",
        query_params={"a1": "webull", "a2": "123", "a3": "xxx", "q1": "yyy"},
        signing_headers={
            "x-app-key": "776da210ab4a452795d74e726ebd74b6",
            "x-timestamp": "2022-01-04T03:55:31Z",
            "x-signature-algorithm": "HMAC-SHA1",
            "x-signature-version": "1.0",
            "x-signature-nonce": "48ef5afed43d4d91ae514aaeafbc29ba",
            "host": "api.webull.com",
        },
        body_string=body,
    )
    assert signature == "kvlS6opdZDhEBo5jq40nHYXaLvM="

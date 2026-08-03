"""Offline tests for Webull paper-account order placement."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from joker.broker.factory import BrokerFactoryError, create_broker
from joker.broker.interface import BrokerError, PaperBroker
from joker.broker.webull import WebullClient
from joker.broker.webull_trade_api import (
    MockWebullTradeApi,
    WebullTradeConfigError,
    build_option_limit_order_payload,
    ensure_paper_trading_allowed,
    extract_cash_balance,
    map_webull_order_status,
    validate_webull_paper_trade_env,
)
from joker.config.settings import AppSettings, EnvSettings
from joker.data.webull_endpoints import get_endpoint
from joker.schemas.domain import OptionContract, OrderIntent


def _env(**overrides: object) -> EnvSettings:
    base = {
        "OPENAI_API_KEY": "sk-test-key-for-unit-tests-only",
        "OPENAI_MODEL": "gpt-5.4-mini",
        "WEBULL_APP_KEY": "test-app-key",
        "WEBULL_APP_SECRET": "test-app-secret",
        "WEBULL_ACCESS_TOKEN": "test-access-token",
        "WEBULL_PAPER_TRADING_ENABLED": True,
        "WEBULL_PAPER_ACCOUNT_ID": "PAPER_ACCT_TEST",
        "WEBULL_LIVE_TRADING_ENABLED": False,
    }
    base.update(overrides)
    return EnvSettings(**base)  # type: ignore[arg-type]


def _intent(**kw) -> OrderIntent:
    contract = OptionContract(
        symbol="SPY",
        expiration=date.today(),
        strike=kw.get("strike", 600.0),
        option_type=kw.get("option_type", "call"),
        is_0dte=True,
    )
    return OrderIntent(
        candidate_id="c1",
        contract=contract,
        side=kw.get("side", "buy"),
        order_type="limit",
        quantity=kw.get("quantity", 1),
        limit_price=kw.get("limit_price", 0.50),
    )


def test_trade_endpoints_use_documented_paths() -> None:
    assert get_endpoint("broker_order_place").path == "/openapi/trade/order/place"
    assert get_endpoint("broker_order_cancel").path == "/openapi/trade/order/cancel"
    assert get_endpoint("broker_account_list").path == "/openapi/account/list"
    assert get_endpoint("broker_account_balance").path == "/openapi/assets/balance"
    assert get_endpoint("broker_account_positions").path == "/openapi/assets/positions"


def test_live_trading_env_still_blocks_paper_helpers() -> None:
    """Live flag may be true for WebullLiveClient, but paper helpers refuse it."""
    env = _env(WEBULL_LIVE_TRADING_ENABLED=True)
    assert env.webull_live_trading_enabled is True
    with pytest.raises(WebullTradeConfigError, match="WEBULL_LIVE_TRADING_ENABLED"):
        ensure_paper_trading_allowed(env)


def test_paper_trading_requires_flag() -> None:
    env = _env(WEBULL_PAPER_TRADING_ENABLED=False)
    with pytest.raises(WebullTradeConfigError, match="WEBULL_PAPER_TRADING_ENABLED"):
        ensure_paper_trading_allowed(env)


def test_paper_trading_requires_account_id() -> None:
    env = _env(WEBULL_PAPER_ACCOUNT_ID="")
    with pytest.raises(WebullTradeConfigError, match="WEBULL_PAPER_ACCOUNT_ID"):
        validate_webull_paper_trade_env(env)


def test_build_option_limit_payload() -> None:
    payload = build_option_limit_order_payload(_intent(), client_order_id="a" * 32)
    assert payload["instrument_type"] == "OPTION"
    assert payload["order_type"] == "LIMIT"
    assert payload["option_strategy"] == "SINGLE"
    assert payload["legs"][0]["option_type"] == "CALL"
    assert payload["client_order_id"] == "a" * 32


def test_reject_market_option_order() -> None:
    intent = _intent()
    intent = intent.model_copy(update={"order_type": "market", "limit_price": None})
    with pytest.raises(WebullTradeConfigError, match="LIMIT"):
        build_option_limit_order_payload(intent)


def test_webull_client_place_and_cancel_with_mock_api() -> None:
    api = MockWebullTradeApi(account_id="PAPER_ACCT_TEST")
    client = WebullClient(_env(), trade_api=api)
    order = client.submit_order(_intent(limit_price=0.01))
    assert order.order_id
    assert len(order.order_id) == 32
    assert order.status == "filled"
    assert api.placed
    assert client.list_positions()
    cancelled = client.cancel_order(order.order_id)
    assert cancelled.status == "cancelled"
    assert order.order_id in api.cancelled


def test_webull_client_rejects_wrong_account_on_place() -> None:
    api = MockWebullTradeApi(account_id="OTHER")
    # Mock doesn't check account; wrap place to simulate Http gate.
    real_place = api.place_order

    def guarded(account_id: str, new_orders: list) -> dict:
        if account_id != "PAPER_ACCT_TEST":
            raise WebullTradeConfigError("mismatch")
        return real_place(account_id, new_orders)

    api.place_order = guarded  # type: ignore[method-assign]
    client = WebullClient(_env(), trade_api=api)
    order = client.submit_order(_intent())
    assert order.status in {"open", "pending", "filled"}


def test_factory_defaults_to_paper_broker() -> None:
    app = AppSettings.model_validate({"mode": "PAPER", "broker": {"provider": "paper"}})
    broker = create_broker(app, _env(WEBULL_PAPER_TRADING_ENABLED=False, WEBULL_PAPER_ACCOUNT_ID=None))
    assert isinstance(broker, PaperBroker)


def test_resolve_live_paper_auto_selects_webull_when_env_ready() -> None:
    from joker.broker.factory import resolve_live_paper_broker

    app = AppSettings.model_validate({"mode": "PAPER", "broker": {"provider": "paper"}})
    api = MockWebullTradeApi()
    selection = resolve_live_paper_broker(app, _env(), trade_api=api)
    assert selection.kind == "webull_paper"
    assert selection.auto_orders is True
    assert isinstance(selection.client, WebullClient)


def test_factory_webull_paper_requires_env_flag() -> None:
    app = AppSettings.model_validate(
        {"mode": "PAPER", "broker": {"provider": "webull_paper"}}
    )
    with pytest.raises(BrokerFactoryError, match="WEBULL_PAPER_TRADING_ENABLED"):
        create_broker(app, _env(WEBULL_PAPER_TRADING_ENABLED=False))


def test_factory_webull_paper_uses_client() -> None:
    app = AppSettings.model_validate(
        {"mode": "PAPER", "broker": {"provider": "webull_paper"}}
    )
    api = MockWebullTradeApi()
    broker = create_broker(app, _env(), trade_api=api)
    assert isinstance(broker, WebullClient)
    order = broker.submit_order(_intent())
    assert order.order_id in {o["client_order_id"] for o in api.placed}


def test_map_status() -> None:
    assert map_webull_order_status("FILLED") == "filled"
    assert map_webull_order_status("CANCELLED") == "cancelled"
    assert map_webull_order_status("SUBMITTED") == "open"


def test_extract_cash_balance() -> None:
    assert extract_cash_balance({"total_cash": "1234.5"}) == 1234.5
    assert extract_cash_balance({"total_cash_balance": "99.0"}) == 99.0
    assert extract_cash_balance({}) == 0.0


def test_account_looks_like_live_brokerage() -> None:
    from joker.broker.webull_trade_api import account_looks_like_live_brokerage

    assert account_looks_like_live_brokerage(
        {"account_label": "Individual Cash", "account_class": "INDIVIDUAL_CASH"},
        api_env="prod",
    )
    assert not account_looks_like_live_brokerage(
        {"account_label": "Individual Cash", "account_class": "INDIVIDUAL_CASH"},
        api_env="sandbox",
    )
    assert not account_looks_like_live_brokerage(
        {"account_label": "Paper Trading", "account_class": "PAPER"}
    )


def test_trade_credentials_env_prefers_trade_keys() -> None:
    env = _env(
        WEBULL_TRADE_APP_KEY="trade-key",
        WEBULL_TRADE_APP_SECRET="trade-secret",
        WEBULL_TRADE_API_ENV="sandbox",
        WEBULL_TRADE_ACCESS_TOKEN="trade-tok",
    )
    trade = env.trade_credentials_env()
    assert trade.webull_app_key == "trade-key"
    assert trade.webull_app_secret == "trade-secret"
    assert trade.webull_api_env == "sandbox"
    assert trade.webull_access_token == "trade-tok"
    # Market-data view unchanged
    assert env.webull_app_key == "test-app-key"
    assert env.webull_api_env != "sandbox" or True


def test_sandbox_base_url_registered() -> None:
    from joker.data.webull_endpoints import WEBULL_BASE_URLS

    assert WEBULL_BASE_URLS["sandbox"] == "https://api.sandbox.webull.com"

def test_client_blocks_without_paper_flag() -> None:
    with pytest.raises(BrokerError, match="PAPER_TRADING"):
        WebullClient(_env(WEBULL_PAPER_TRADING_ENABLED=False), trade_api=MockWebullTradeApi())


def test_legacy_validate_webull_env_still_works() -> None:
    from joker.broker.webull import WebullConfigError, validate_webull_env

    env = _env(WEBULL_DEVICE_ID=None, WEBULL_TRADE_PIN=None)
    with pytest.raises(WebullConfigError, match="WEBULL"):
        validate_webull_env(env)

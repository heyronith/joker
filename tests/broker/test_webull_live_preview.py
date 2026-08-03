"""Preview payload equivalence and rejection gates."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from joker.app.safety import SafetyMode
from joker.broker.webull_live import WebullLiveClient, create_mock_live_trade_api
from joker.broker.webull_trade_api import build_option_limit_order_payload
from joker.config.settings import AppSettings, EnvSettings
from joker.schemas.domain import OptionContract, OrderIntent


def _env() -> EnvSettings:
    return EnvSettings(  # type: ignore[call-arg]
        OPENAI_API_KEY="k",
        WEBULL_LIVE_TRADING_ENABLED=True,
        WEBULL_LIVE_APP_KEY="lk",
        WEBULL_LIVE_APP_SECRET="ls",
        WEBULL_LIVE_ACCESS_TOKEN="lt",
        WEBULL_LIVE_ACCOUNT_ID="LIVE_ACCT_1",
        WEBULL_LIVE_API_ENV="prod",
    )


def _intent() -> OrderIntent:
    return OrderIntent(
        intent_id="p" * 32,
        candidate_id="c",
        contract=OptionContract(
            symbol="SPY",
            expiration=date.today(),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        order_type="limit",
        quantity=2,
        limit_price=1.25,
        position_intent="BUY_TO_OPEN",
    )


def test_preview_place_payload_byte_equivalence() -> None:
    intent = _intent()
    a = build_option_limit_order_payload(
        intent, client_order_id="p" * 32, account_id="LIVE_ACCT_1"
    )
    b = build_option_limit_order_payload(
        intent, client_order_id="p" * 32, account_id="LIVE_ACCT_1"
    )
    assert a == b
    assert a["client_order_id"] == "p" * 32
    assert a["quantity"] == "2"
    assert a["limit_price"] == "1.25"
    assert a["position_intent"] == "BUY_TO_OPEN"


def test_preview_endpoint_registered() -> None:
    from joker.data.webull_endpoints import get_endpoint

    ep = get_endpoint("broker_order_preview")
    assert ep.path == "/openapi/trade/order/preview"
    assert ep.verified is True


def test_live_preview_uses_same_payload_as_builder(tmp_path) -> None:
    from tests.broker._live_helpers import make_live_client
    client, api, _ = make_live_client(tmp_path)
    intent = _intent()
    built = client.build_payload(intent)
    preview = client.preview_order(intent)
    assert preview.accepted is True
    assert api.previewed[0]["client_order_id"] == built["client_order_id"]
    assert api.previewed[0]["limit_price"] == built["limit_price"]
    assert Decimal(api.previewed[0]["quantity"]) == Decimal(built["quantity"])

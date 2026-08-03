"""Paper vs live capture-mode: identical pre-broker commands."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.broker.webull_live import WebullLiveClient, create_mock_live_trade_api
from joker.config.settings import AppSettings, EnvSettings
from joker.runtime.execution_runtime import ExecutionCommand
from joker.runtime.order_action_gateway import (
    OrderActionKind,
    OrderActionRequest,
    _resolve_gateway_position_intent,
)
from joker.schemas.domain import OptionContract, OrderIntent


def _intent(limit: float = 1.10, qty: int = 1) -> OrderIntent:
    return OrderIntent(
        intent_id="z" * 32,
        candidate_id="same-proposal",
        contract=OptionContract(
            symbol="SPY",
            expiration=date.today(),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        order_type="limit",
        quantity=qty,
        limit_price=limit,
        position_intent="BUY_TO_OPEN",
    )


def test_paper_live_deterministic_command_equivalence() -> None:
    """Canonical ExecutionCommand fields match before broker decoration."""
    intent = _intent()
    paper_cmd = ExecutionCommand(
        client_order_id=intent.intent_id,
        intent=intent,
        broker_account_id="paper",
    )
    live_cmd = ExecutionCommand(
        client_order_id=intent.intent_id,
        intent=intent.model_copy(),
        broker_account_id="live-hash",
    )
    # Only broker account identifier may differ.
    assert paper_cmd.intent.model_dump() == live_cmd.intent.model_dump()
    assert paper_cmd.client_order_id == live_cmd.client_order_id
    assert paper_cmd.intent.quantity == live_cmd.intent.quantity
    assert paper_cmd.intent.limit_price == live_cmd.intent.limit_price
    assert paper_cmd.intent.side == live_cmd.intent.side
    assert paper_cmd.intent.position_intent == live_cmd.intent.position_intent
    assert paper_cmd.broker_account_id != live_cmd.broker_account_id


def test_live_capture_mode_preserves_payload_without_placement() -> None:
    env = EnvSettings(  # type: ignore[call-arg]
        OPENAI_API_KEY="k",
        WEBULL_LIVE_TRADING_ENABLED=True,
        WEBULL_LIVE_APP_KEY="lk",
        WEBULL_LIVE_APP_SECRET="ls",
        WEBULL_LIVE_ACCESS_TOKEN="lt",
        WEBULL_LIVE_ACCOUNT_ID="LIVE_ACCT_1",
        WEBULL_LIVE_API_ENV="prod",
    )
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    live = WebullLiveClient(
        env,
        app_settings=AppSettings(
            mode=SafetyMode.LIVE_GATED,
            live_trading_enabled=True,
            broker={"provider": "webull_live"},
        ),
        trade_api=api,
        skip_account_list_check=True,
        capture_only=True,
    )
    intent = _intent()
    order = live.submit_order(intent)
    assert order.status == "pending"
    assert api.placed == []
    assert len(live.captured_payloads) == 1
    payload = live.captured_payloads[0]
    assert payload["limit_price"] == "1.10"
    assert payload["quantity"] == "1"
    assert payload["position_intent"] == "BUY_TO_OPEN"
    assert payload["side"] == "BUY"


def test_gateway_position_intent_entry_exit() -> None:
    entry = _resolve_gateway_position_intent(
        action=OrderActionKind.ENTRY,
        side="buy",
        contract_id=f"SPY:{date.today().isoformat()}:500.0:call",
        open_positions={},
    )
    assert entry == "BUY_TO_OPEN"

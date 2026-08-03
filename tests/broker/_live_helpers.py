"""Shared helpers for live broker unit tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from joker.app.safety import SafetyMode
from joker.broker.account_truth import hash_account_id
from joker.broker.webull_live import WebullLiveClient, create_mock_live_trade_api
from joker.broker.webull_trade_api import MockWebullTradeApi
from joker.config.settings import AppSettings, EnvSettings
from joker.persistence.broker_submission_journal import (
    BrokerSubmissionRecord,
    SyncBrokerSubmissionJournal,
    payload_hash,
)
from joker.persistence.session_pnl_baseline import SessionPnlBaselineStore
from joker.runtime.live_activation import create_live_activation
from joker.schemas.domain import OptionContract, OrderIntent


def live_env(**overrides) -> EnvSettings:
    base = {
        "OPENAI_API_KEY": "test-key",
        "WEBULL_LIVE_TRADING_ENABLED": True,
        "WEBULL_LIVE_APP_KEY": "live-key",
        "WEBULL_LIVE_APP_SECRET": "live-secret",
        "WEBULL_LIVE_ACCESS_TOKEN": "live-token",
        "WEBULL_LIVE_ACCOUNT_ID": "LIVE_ACCT_1",
        "WEBULL_LIVE_API_ENV": "prod",
        "WEBULL_PAPER_TRADING_ENABLED": False,
        "WEBULL_PAPER_ACCOUNT_ID": "PAPER_ACCT_1",
        "WEBULL_APP_KEY": "paper-key",
        "WEBULL_APP_SECRET": "paper-secret",
        "WEBULL_ACCESS_TOKEN": "paper-token",
    }
    base.update(overrides)
    return EnvSettings(**base)  # type: ignore[arg-type]


def live_app(*, live: bool = True, db_path: str | Path | None = None) -> AppSettings:
    kwargs: dict = {
        "mode": SafetyMode.LIVE_GATED if live else SafetyMode.PAPER,
        "live_trading_enabled": live,
        "broker": {"provider": "webull_live"},
    }
    if db_path is not None:
        kwargs["db_path"] = str(db_path)
    return AppSettings(**kwargs)


def live_activation(account_id: str = "LIVE_ACCT_1"):
    return create_live_activation(
        account_id_hash=hash_account_id(account_id),
        objective_id=uuid4(),
        authorized_capital_usd=Decimal("10000"),
    )


def make_live_client(
    tmp_path: Path,
    api: MockWebullTradeApi | None = None,
    *,
    capture_only: bool = False,
    account_id: str = "LIVE_ACCT_1",
    **kwargs,
):
    api = api or create_mock_live_trade_api(account_id)
    db = tmp_path / "journal.db"
    journal = SyncBrokerSubmissionJournal(db)
    baseline = SessionPnlBaselineStore(db)
    activation = kwargs.pop("activation", None)
    if activation is None and not capture_only:
        activation = live_activation(account_id)
    client = WebullLiveClient(
        live_env(WEBULL_LIVE_ACCOUNT_ID=account_id),
        app_settings=live_app(),
        activation=activation,
        trade_api=api,
        journal=journal,
        baseline_store=baseline,
        skip_account_list_check=kwargs.pop("skip_account_list_check", True),
        capture_only=capture_only,
        **kwargs,
    )
    return client, api, journal


def contract_today() -> OptionContract:
    return OptionContract(
        symbol="SPY",
        expiration=date.today(),
        strike=500.0,
        option_type="call",
        is_0dte=True,
    )


def make_intent(
    *,
    side: str = "buy",
    position_intent: str | None = "BUY_TO_OPEN",
    qty: int = 1,
    limit: float = 1.10,
    intent_id: str | None = None,
) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id or ("c" * 32),
        candidate_id="cand",
        contract=contract_today(),
        side=side,  # type: ignore[arg-type]
        order_type="limit",
        quantity=qty,
        limit_price=limit,
        position_intent=position_intent,  # type: ignore[arg-type]
    )


def prepare_journal_for_intent(client: WebullLiveClient, intent: OrderIntent) -> None:
    """Gateway-owned prepare — required before live placement."""
    assert client.journal is not None
    payload = client.build_payload(intent, client_order_id=intent.intent_id)
    cid = str(payload["client_order_id"])
    contract = intent.contract
    client.journal.prepare(
        BrokerSubmissionRecord(
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
            session_id="test-session",
            contract_id=(
                f"{contract.symbol}:{contract.expiration.isoformat()}:"
                f"{contract.strike}:{contract.option_type}"
            ),
            side=intent.side,
            position_intent=intent.position_intent,
            quantity=intent.quantity,
            limit_price=(
                f"{intent.limit_price:.2f}" if intent.limit_price is not None else None
            ),
            payload_hash=payload_hash(payload),
        )
    )

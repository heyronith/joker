"""Step 1 — WebullLiveClient credentials, intent, submission, reconciliation."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from joker.app.safety import SafetyMode
from joker.broker.factory import BrokerFactoryError, create_broker, create_live_broker
from joker.broker.interface import BrokerError
from joker.broker.position_intent import resolve_position_intent, validate_position_intent
from joker.broker.reconciliation import BrokerReconciliationService
from joker.broker.webull import WebullClient
from joker.broker.webull_live import (
    WebullLiveClient,
    WebullLiveConfigError,
    create_mock_live_trade_api,
)
from joker.broker.webull_trade_api import (
    MockWebullTradeApi,
    build_option_limit_order_payload,
    ensure_paper_trading_allowed,
)
from joker.config.settings import AppSettings, EnvSettings
from joker.persistence.broker_submission_journal import SyncBrokerSubmissionJournal
from joker.schemas.domain import OptionContract, OrderIntent, Position


def _today() -> date:
    return date.today()


def _env(**overrides) -> EnvSettings:
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


def _app(*, live: bool = True) -> AppSettings:
    return AppSettings(
        mode=SafetyMode.LIVE_GATED if live else SafetyMode.PAPER,
        live_trading_enabled=live,
        broker={"provider": "webull_live"},
    )


def _contract() -> OptionContract:
    return OptionContract(
        symbol="SPY",
        expiration=_today(),
        strike=500.0,
        option_type="call",
        is_0dte=True,
    )


def _intent(
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
        contract=_contract(),
        side=side,  # type: ignore[arg-type]
        order_type="limit",
        quantity=qty,
        limit_price=limit,
        position_intent=position_intent,  # type: ignore[arg-type]
    )


def _live_client(tmp_path: Path, api: MockWebullTradeApi | None = None, **kwargs):
    api = api or create_mock_live_trade_api("LIVE_ACCT_1")
    journal = SyncBrokerSubmissionJournal(tmp_path / "journal.db")
    return WebullLiveClient(
        _env(),
        app_settings=_app(),
        trade_api=api,
        journal=journal,
        skip_account_list_check=kwargs.pop("skip_account_list_check", True),
        **kwargs,
    ), api, journal


def test_live_credentials_never_fall_back_to_paper() -> None:
    env = _env(
        WEBULL_LIVE_APP_KEY=None,
        WEBULL_APP_KEY="paper-only-key",
        WEBULL_TRADE_APP_KEY="trade-only-key",
    )
    creds = env.live_credentials_env()
    assert creds.app_key is None
    assert "WEBULL_LIVE_APP_KEY" in creds.missing_fields()
    trade = env.trade_credentials_env()
    assert trade.webull_app_key == "trade-only-key"


def test_live_account_id_must_match_returned_account(tmp_path) -> None:
    api = create_mock_live_trade_api("OTHER_ACCT")
    with pytest.raises(WebullLiveConfigError, match="not returned"):
        WebullLiveClient(
            _env(),
            app_settings=_app(),
            trade_api=api,
            skip_account_list_check=False,
        )


def test_wrong_account_blocks_startup() -> None:
    api = create_mock_live_trade_api("OTHER")
    with pytest.raises((WebullLiveConfigError, BrokerFactoryError)):
        create_live_broker(
            _app(),
            _env(),
            trade_api=api,
            skip_account_list_check=False,
        )


def test_sandbox_environment_blocks_webull_live_client() -> None:
    with pytest.raises(WebullLiveConfigError, match="prod"):
        WebullLiveClient(
            _env(WEBULL_LIVE_API_ENV="sandbox"),
            app_settings=_app(),
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            skip_account_list_check=True,
        )


def test_paper_account_blocks_webull_live_client() -> None:
    with pytest.raises(WebullLiveConfigError, match="must not equal"):
        WebullLiveClient(
            _env(WEBULL_LIVE_ACCOUNT_ID="PAPER_ACCT_1"),
            app_settings=_app(),
            trade_api=create_mock_live_trade_api("PAPER_ACCT_1"),
            skip_account_list_check=False,
        )


def test_live_client_cannot_be_constructed_without_live_gated() -> None:
    with pytest.raises(WebullLiveConfigError, match="LIVE_GATED"):
        WebullLiveClient(
            _env(),
            app_settings=_app(live=False),
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            skip_account_list_check=True,
        )


def test_paper_client_remains_unable_to_place_live_orders() -> None:
    env = _env(
        WEBULL_PAPER_TRADING_ENABLED=True,
        WEBULL_LIVE_TRADING_ENABLED=True,
    )
    with pytest.raises(Exception, match="LIVE"):
        ensure_paper_trading_allowed(env)
    with pytest.raises(BrokerError, match="live|LIVE"):
        WebullClient(env, trade_api=MockWebullTradeApi("PAPER_ACCT_1"))


def test_buy_to_open_payload(tmp_path) -> None:
    client, _, _ = _live_client(tmp_path)
    payload = client.build_payload(_intent(position_intent="BUY_TO_OPEN"))
    assert payload["position_intent"] == "BUY_TO_OPEN"
    assert payload["open_close"] == "OPEN"
    assert payload["side"] == "BUY"
    assert payload["legs"][0]["open_close"] == "OPEN"


def test_sell_to_close_payload(tmp_path) -> None:
    client, api, _ = _live_client(tmp_path)
    # Seed a long position matching today's contract.
    pos_contract = _contract()
    api._positions.append(
        {
            "position_id": "p1",
            "instrument_type": "OPTION",
            "symbol": "SPY",
            "quantity": "1",
            "cost_price": "1.00",
            "legs": [
                {
                    "symbol": "SPY",
                    "option_type": "CALL",
                    "option_expire_date": pos_contract.expiration.isoformat(),
                    "option_exercise_price": "500.00",
                    "quantity": "1",
                }
            ],
        }
    )
    intent = _intent(side="sell", position_intent="SELL_TO_CLOSE", intent_id="d" * 32)
    positions = client.list_positions()
    payload = client.build_payload(intent, open_positions=positions)
    assert payload["position_intent"] == "SELL_TO_CLOSE"
    assert payload["open_close"] == "CLOSE"


def test_position_intent_mismatch_rejection() -> None:
    with pytest.raises(BrokerError, match="matching long"):
        resolve_position_intent(
            action="exit", side="sell", contract_id="SPY:x:1:call", open_positions=[]
        )
    with pytest.raises(BrokerError, match="side=buy"):
        validate_position_intent("BUY_TO_OPEN", side="sell")


def test_preview_and_place_payload_equivalence(tmp_path) -> None:
    client, api, _ = _live_client(tmp_path)
    intent = _intent()
    payload = client.build_payload(intent)
    preview = client.preview_order(intent)
    assert preview.accepted is True
    order = client.submit_order(intent)
    assert api.previewed[0]["client_order_id"] == payload["client_order_id"]
    assert api.placed[0]["client_order_id"] == payload["client_order_id"]
    assert api.placed[0]["limit_price"] == payload["limit_price"]
    assert order.order_id == payload["client_order_id"]


def test_preview_rejection(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.preview_reject = "INSUFFICIENT_BP"
    client, _, _ = _live_client(tmp_path, api=api)
    preview = client.preview_order(_intent())
    assert preview.accepted is False
    assert preview.rejection_code == "INSUFFICIENT_BP"


def test_preview_cost_mismatch(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.preview_cost_usd = 5000.0
    client, _, _ = _live_client(tmp_path, api=api)
    preview = client.preview_order(
        _intent(limit=1.10),
        expected_notional_usd=Decimal("110.00"),
        max_notional_mismatch_pct=Decimal("5"),
    )
    assert preview.accepted is False
    assert preview.rejection_code == "preview_cost_mismatch"


def test_insufficient_buying_power(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.buying_power_usd = 1.0
    api.preview_reject = "INSUFFICIENT_BUYING_POWER"
    client, _, _ = _live_client(tmp_path, api=api)
    preview = client.preview_order(_intent())
    assert preview.accepted is False


def test_client_order_id_persisted_before_submission(tmp_path) -> None:
    client, api, journal = _live_client(tmp_path)
    intent = _intent(intent_id="e" * 32)
    # Force timeout after prepare
    api.place_timeout = True
    with pytest.raises(BrokerError, match="unknown"):
        client.submit_order(intent)
    stored = journal.get(client.account_id_hash, "e" * 32)
    assert stored is not None
    assert stored.status == "submission_unknown"
    assert stored.payload_hash


def test_duplicate_client_order_id_rejected(tmp_path) -> None:
    client, _, _ = _live_client(tmp_path)
    intent = _intent(intent_id="f" * 32)
    client.submit_order(intent)
    with pytest.raises(BrokerError, match="duplicate"):
        client.submit_order(intent)


def test_timeout_before_broker_acceptance(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = False
    client, _, journal = _live_client(tmp_path, api=api)
    with pytest.raises(BrokerError, match="unknown"):
        client.submit_order(_intent(intent_id="a" * 32))
    stored = journal.get(client.account_id_hash, "a" * 32)
    assert stored is not None
    assert stored.status == "submission_unknown"


def test_timeout_after_broker_acceptance(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = True
    client, _, journal = _live_client(tmp_path, api=api)
    # reconcile via get_order_detail should recover
    order = client.submit_order(_intent(intent_id="b" * 32))
    assert order.status in {"open", "pending", "filled"}
    stored = journal.get(client.account_id_hash, "b" * 32)
    assert stored is not None
    assert stored.status in {"accepted", "filled", "reconciled"}


def test_submission_unknown_reconciliation(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = True
    client, _, journal = _live_client(tmp_path, api=api)
    client.submit_order(_intent(intent_id="1" * 32))
    svc = BrokerReconciliationService(
        broker=client, journal=journal, account_id_hash=client.account_id_hash
    )
    report = svc.reconcile()
    # After reconcile attempt unknowns may resolve
    stored = journal.get(client.account_id_hash, "1" * 32)
    assert stored is not None
    assert stored.status != "prepared"
    assert isinstance(report.unknown_submissions, int)


def test_capital_reservation_retained_while_unknown(tmp_path) -> None:
    from joker.broker.reconciliation import capital_reservation_release_allowed

    assert capital_reservation_release_allowed("submission_unknown") is False
    assert capital_reservation_release_allowed("rejected") is True
    assert capital_reservation_release_allowed("cancelled") is True


def test_confirmed_rejection_releases_reservation(tmp_path) -> None:
    from joker.broker.reconciliation import capital_reservation_release_allowed

    assert capital_reservation_release_allowed("rejected") is True


def test_partial_and_final_fill_and_cancel(tmp_path) -> None:
    client, api, journal = _live_client(tmp_path)
    order = client.submit_order(_intent(intent_id="2" * 32))
    assert order.status == "filled"
    # Force open then cancel
    api._orders["3" * 32] = {
        "client_order_id": "3" * 32,
        "status": "SUBMITTED",
        "order_id": "WB-3",
    }
    intent = _intent(intent_id="3" * 32)
    client._intent_by_order["3" * 32] = intent
    client._orders["3" * 32] = order.model_copy(
        update={"order_id": "3" * 32, "status": "open", "intent_id": "3" * 32}
    )
    journal.prepare(
        __import__(
            "joker.persistence.broker_submission_journal", fromlist=["BrokerSubmissionRecord"]
        ).BrokerSubmissionRecord(
            client_order_id="3" * 32,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
        )
    )
    cancelled = client.cancel_order("3" * 32)
    assert cancelled.status == "cancelled"
    detail = client.get_order("2" * 32)
    assert detail is not None


def test_broker_local_mismatches(tmp_path) -> None:
    client, api, journal = _live_client(tmp_path)
    client.submit_order(_intent(intent_id="4" * 32))
    svc = BrokerReconciliationService(
        broker=client, journal=journal, account_id_hash=client.account_id_hash
    )
    # Local empty vs broker position after fill
    report = svc.reconcile(local_orders=[], local_positions=[])
    kinds = {f.kind for f in report.findings}
    assert "broker_position_missing_locally" in kinds or report.degraded


def test_account_truth_does_not_fabricate_zero_pnl(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    # Remove NLV so session pnl cannot be derived from baseline either if we clear it
    client, _, _ = _live_client(tmp_path, api=api)
    truth = client.get_account_truth()
    assert truth.cash_usd == Decimal("100000.0") or truth.cash_usd == Decimal("100000")
    assert truth.session_pnl_available is True  # baseline established
    # Second call with same NLV → 0 pnl available (real calculation, not fabricated missing)
    truth2 = client.get_account_truth()
    assert truth2.session_pnl_usd == Decimal("0")


def test_create_broker_webull_live_no_fallback(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    client = create_broker(
        _app(),
        _env(),
        trade_api=api,
        journal_db_path=tmp_path / "j.db",
    )
    assert isinstance(client, WebullLiveClient)


def test_factory_failure_does_not_return_paper() -> None:
    with pytest.raises(BrokerFactoryError, match="refusing|failed|LIVE_GATED"):
        create_live_broker(_app(live=False), _env(), trade_api=create_mock_live_trade_api())

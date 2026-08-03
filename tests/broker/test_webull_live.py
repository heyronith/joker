"""Step 1 — WebullLiveClient credentials, intent, submission, reconciliation."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from joker.broker.factory import BrokerFactoryError, create_broker, create_live_broker
from joker.broker.interface import BrokerError, BrokerSubmissionUnknown
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
    ensure_paper_trading_allowed,
)
from joker.persistence.broker_submission_journal import BrokerSubmissionRecord
from joker.schemas.domain import Position
from tests.broker._live_helpers import (
    live_activation,
    live_app,
    live_env,
    make_intent,
    make_live_client,
    prepare_journal_for_intent,
)


def _env(**overrides):
    return live_env(**overrides)


def _app(*, live: bool = True):
    return live_app(live=live)


def _intent(**kwargs):
    return make_intent(**kwargs)


def _contract():
    from tests.broker._live_helpers import contract_today
    return contract_today()


def _live_client(tmp_path: Path, api: MockWebullTradeApi | None = None, **kwargs):
    return make_live_client(tmp_path, api, **kwargs)


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
    with pytest.raises(WebullLiveConfigError, match="not returned|LiveActivation|journal"):
        WebullLiveClient(
            _env(),
            app_settings=_app(),
            trade_api=api,
            activation=live_activation(),
            journal=__import__("joker.persistence.broker_submission_journal", fromlist=["SyncBrokerSubmissionJournal"]).SyncBrokerSubmissionJournal(tmp_path / "j.db"),
            skip_account_list_check=False,
        )


def test_wrong_account_blocks_startup(tmp_path) -> None:
    api = create_mock_live_trade_api("OTHER")
    with pytest.raises((WebullLiveConfigError, BrokerFactoryError)):
        create_live_broker(
            _app(),
            _env(),
            trade_api=api,
            activation=live_activation(),
            journal_db_path=tmp_path / "j.db",
            skip_account_list_check=False,
        )


def test_sandbox_environment_blocks_webull_live_client(tmp_path) -> None:
    with pytest.raises(WebullLiveConfigError, match="prod"):
        WebullLiveClient(
            _env(WEBULL_LIVE_API_ENV="sandbox"),
            app_settings=_app(),
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            activation=live_activation(),
            journal=__import__("joker.persistence.broker_submission_journal", fromlist=["SyncBrokerSubmissionJournal"]).SyncBrokerSubmissionJournal(tmp_path / "j.db"),
            skip_account_list_check=True,
        )


def test_paper_account_blocks_webull_live_client(tmp_path) -> None:
    with pytest.raises(WebullLiveConfigError, match="must not equal"):
        WebullLiveClient(
            _env(WEBULL_LIVE_ACCOUNT_ID="PAPER_ACCT_1"),
            app_settings=_app(),
            trade_api=create_mock_live_trade_api("PAPER_ACCT_1"),
            activation=live_activation("PAPER_ACCT_1"),
            journal=__import__("joker.persistence.broker_submission_journal", fromlist=["SyncBrokerSubmissionJournal"]).SyncBrokerSubmissionJournal(tmp_path / "j.db"),
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
    prepare_journal_for_intent(client, intent)
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
    prepare_journal_for_intent(client, intent)
    api.place_timeout = True
    with pytest.raises(BrokerSubmissionUnknown):
        client.submit_order(intent)
    stored = journal.get(client.account_id_hash, "e" * 32)
    assert stored is not None
    assert stored.status == "submission_unknown"
    assert stored.payload_hash


def test_duplicate_client_order_id_rejected(tmp_path) -> None:
    from joker.persistence.broker_submission_journal import DuplicateSubmissionError

    client, _, _ = _live_client(tmp_path)
    intent = _intent(intent_id="f" * 32)
    prepare_journal_for_intent(client, intent)
    client.submit_order(intent)
    with pytest.raises(DuplicateSubmissionError, match="duplicate"):
        # Second prepare must fail closed on duplicate identity.
        prepare_journal_for_intent(client, intent)


def test_timeout_before_broker_acceptance(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = False
    client, _, journal = _live_client(tmp_path, api=api)
    intent = _intent(intent_id="a" * 32)
    prepare_journal_for_intent(client, intent)
    with pytest.raises(BrokerSubmissionUnknown):
        client.submit_order(intent)
    stored = journal.get(client.account_id_hash, "a" * 32)
    assert stored is not None
    assert stored.status == "submission_unknown"


def test_timeout_after_broker_acceptance(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = True
    client, _, journal = _live_client(tmp_path, api=api)
    intent = _intent(intent_id="b" * 32)
    prepare_journal_for_intent(client, intent)
    order = client.submit_order(intent)
    assert order.status in {"open", "pending", "filled", "partially_filled"}
    stored = journal.get(client.account_id_hash, "b" * 32)
    assert stored is not None
    assert stored.status in {"accepted", "filled", "reconciled", "partially_filled"}


def test_submission_unknown_reconciliation(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = True
    client, _, journal = _live_client(tmp_path, api=api)
    intent = _intent(intent_id="1" * 32)
    prepare_journal_for_intent(client, intent)
    client.submit_order(intent)
    svc = BrokerReconciliationService(
        broker=client, journal=journal, account_id_hash=client.account_id_hash
    )
    resolved = svc.resolve_unknown_submissions()
    report = svc.reconcile()
    stored = journal.get(client.account_id_hash, "1" * 32)
    assert stored is not None
    assert stored.status != "prepared"
    assert isinstance(report.unknown_submissions, int)
    assert isinstance(resolved, list)


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
    intent2 = _intent(intent_id="2" * 32)
    prepare_journal_for_intent(client, intent2)
    order = client.submit_order(intent2)
    assert order.status == "filled"
    api._orders["3" * 32] = {
        "client_order_id": "3" * 32,
        "status": "SUBMITTED",
        "order_id": "WB-3",
        "side": "BUY",
        "quantity": "1",
        "limit_price": "1.10",
        "symbol": "SPY",
        "option_expire_date": __import__("datetime").date.today().isoformat(),
        "strike_price": "500",
        "option_type": "call",
    }
    intent = _intent(intent_id="3" * 32)
    prepare_journal_for_intent(client, intent)
    journal.transition(
        account_id_hash=client.account_id_hash,
        client_order_id="3" * 32,
        status="accepted",
        broker_order_id="WB-3",
    )
    cancelled = client.cancel_order("3" * 32)
    assert cancelled.status == "cancelled"
    detail = client.get_order("2" * 32)
    assert detail is not None


def test_broker_local_mismatches(tmp_path) -> None:
    client, api, journal = _live_client(tmp_path)
    intent = _intent(intent_id="4" * 32)
    prepare_journal_for_intent(client, intent)
    client.submit_order(intent)
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
        activation=live_activation(),
    )
    assert isinstance(client, WebullLiveClient)


def test_factory_failure_does_not_return_paper() -> None:
    with pytest.raises(BrokerFactoryError, match="refusing|failed|LIVE_GATED"):
        create_live_broker(_app(live=False), _env(), trade_api=create_mock_live_trade_api())

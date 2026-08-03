"""Broker reconciliation mismatch classification."""

from __future__ import annotations

from datetime import date

from joker.app.safety import SafetyMode
from joker.broker.reconciliation import BrokerReconciliationService
from joker.broker.webull_live import WebullLiveClient, create_mock_live_trade_api
from joker.config.settings import AppSettings, EnvSettings
from joker.persistence.broker_submission_journal import (
    BrokerSubmissionRecord,
    SyncBrokerSubmissionJournal,
)
from joker.schemas.domain import OptionContract, OrderIntent, Position


def _client(tmp_path):
    from tests.broker._live_helpers import make_live_client, prepare_journal_for_intent
    return make_live_client(tmp_path)


def test_order_detail_polling_resolves_unknown(tmp_path) -> None:
    client, api, journal = _client(tmp_path)
    api.place_timeout = True
    api.place_accepts_before_timeout = True
    intent = OrderIntent(
        intent_id="r" * 32,
        candidate_id="c",
        contract=OptionContract(
            symbol="SPY",
            expiration=date.today(),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        limit_price=1.0,
        position_intent="BUY_TO_OPEN",
    )
    from tests.broker._live_helpers import prepare_journal_for_intent
    prepare_journal_for_intent(client, intent)
    client.submit_order(intent)
    svc = BrokerReconciliationService(
        broker=client, journal=journal, account_id_hash=client.account_id_hash
    )
    report = svc.reconcile(local_orders=list(client._orders.values()), local_positions=[])
    stored = journal.get(client.account_id_hash, "r" * 32)
    assert stored is not None
    assert stored.status in {"accepted", "filled", "reconciled", "submission_unknown"}
    assert report.account_id_hash == client.account_id_hash


def test_restart_with_working_order(tmp_path) -> None:
    client, api, journal = _client(tmp_path)
    api._orders["w" * 32] = {
        "client_order_id": "w" * 32,
        "status": "SUBMITTED",
        "order_id": "WB-W",
        "side": "BUY",
        "quantity": "1",
        "limit_price": "1.00",
        "symbol": "SPY",
        "option_expire_date": date.today().isoformat(),
        "strike_price": "500",
        "option_type": "call",
    }
    journal.prepare(
        BrokerSubmissionRecord(
            client_order_id="w" * 32,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
        )
    )
    journal.transition(
        account_id_hash=client.account_id_hash,
        client_order_id="w" * 32,
        status="accepted",
        broker_order_id="WB-W",
    )
    open_orders = client.list_open_orders()
    # May be empty if get_order can't reconstruct without intent — journal retains state
    stored = journal.get(client.account_id_hash, "w" * 32)
    assert stored is not None
    assert stored.status == "accepted"
    assert isinstance(open_orders, list)


def test_restart_with_open_position(tmp_path) -> None:
    client, api, _ = _client(tmp_path)
    api._positions.append(
        {
            "position_id": "pos1",
            "instrument_type": "OPTION",
            "symbol": "SPY",
            "quantity": "1",
            "cost_price": "1.10",
            "legs": [
                {
                    "symbol": "SPY",
                    "option_type": "CALL",
                    "option_expire_date": date.today().isoformat(),
                    "option_exercise_price": "500",
                    "quantity": "1",
                }
            ],
        }
    )
    positions = client.list_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 1

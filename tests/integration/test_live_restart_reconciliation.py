"""Restart / ambiguous submission recovery for live journal."""

from __future__ import annotations

from datetime import date

import pytest

from joker.app.safety import SafetyMode
from joker.broker.reconciliation import (
    BrokerReconciliationService,
    capital_reservation_release_allowed,
)
from joker.broker.webull_live import WebullLiveClient, create_mock_live_trade_api
from joker.config.settings import AppSettings, EnvSettings
from joker.persistence.broker_submission_journal import (
    BrokerSubmissionRecord,
    SyncBrokerSubmissionJournal,
)
from joker.schemas.domain import OptionContract, OrderIntent


def _client(tmp_path, **api_kw):
    from tests.broker._live_helpers import make_live_client, prepare_journal_for_intent
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    for k, v in api_kw.items():
        setattr(api, k, v)
    return make_live_client(tmp_path, api)


def _intent(cid: str) -> OrderIntent:
    return OrderIntent(
        intent_id=cid,
        candidate_id="p",
        contract=OptionContract(
            symbol="SPY",
            expiration=date.today(),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        ),
        side="buy",
        limit_price=1.05,
        position_intent="BUY_TO_OPEN",
    )


def test_live_runner_handles_submission_unknown(tmp_path) -> None:
    client, api, journal = _client(
        tmp_path, place_timeout=True, place_accepts_before_timeout=True
    )
    from tests.broker._live_helpers import prepare_journal_for_intent
    intent = _intent("u" * 32)
    prepare_journal_for_intent(client, intent)
    order = client.submit_order(intent)
    stored = journal.get(client.account_id_hash, "u" * 32)
    assert stored is not None
    assert stored.status in {"accepted", "filled", "submission_unknown"}
    assert capital_reservation_release_allowed(stored.status) is False or stored.status in {
        "accepted",
        "filled",
    }
    assert order.order_id == "u" * 32


def test_duplicate_proposal_blocked_after_restart(tmp_path) -> None:
    client, _, journal = _client(tmp_path)
    from tests.broker._live_helpers import prepare_journal_for_intent
    intent = _intent("q" * 32)
    prepare_journal_for_intent(client, intent)
    client.submit_order(intent)
    client2, _, journal2 = _client(tmp_path)
    with pytest.raises(Exception, match="duplicate"):
        prepare_journal_for_intent(client2, intent)


def test_exit_sell_to_close_after_fill(tmp_path) -> None:
    client, api, _ = _client(tmp_path)
    from tests.broker._live_helpers import prepare_journal_for_intent
    intent = _intent("v" * 32)
    prepare_journal_for_intent(client, intent)
    client.submit_order(intent)
    positions = client.list_positions()
    assert positions
    exit_intent = OrderIntent(
        intent_id="x" * 32,
        candidate_id="exit",
        contract=positions[0].contract,
        side="sell",
        limit_price=1.20,
        position_intent="SELL_TO_CLOSE",
    )
    payload = client.build_payload(exit_intent, open_positions=positions)
    assert payload["position_intent"] == "SELL_TO_CLOSE"
    assert payload["open_close"] == "CLOSE"

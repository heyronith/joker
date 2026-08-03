"""Live-trading final truth, recovery, and exit correction — focused unit tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from joker.app.safety import SafetyMode
from joker.broker.factory import BrokerFactoryError, create_live_broker
from joker.broker.interface import BrokerClient, BrokerError, BrokerSubmissionUnknown, PaperBroker
from joker.broker.reconciliation import BrokerReconciliationService
from joker.broker.webull_live import WebullLiveClient, WebullLiveConfigError, create_mock_live_trade_api
from joker.broker.webull_trade_api import map_webull_order_status
from joker.config.settings import AppSettings, CognitiveGraphSettings
from joker.events.bus import InProcessAsyncEventBus
from joker.events.schemas import EventType
from joker.graph.cognitive_graph import build_cognitive_graph, initial_cycle_state
from joker.graph.graph_deps import CognitiveGraphDeps
from joker.ledger.schemas import LedgerEventType, make_ledger_event
from joker.ledger.store import SqliteLedgerStore
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.router import ModelRouter
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.objectives.repository import ObjectiveRepository, apply_objective_migrations
from joker.objectives.service import SessionObjectiveService
from joker.persistence.broker_submission_journal import BrokerSubmissionRecord
from joker.persistence.migrations import apply_task1_migrations
from joker.persistence.session_pnl_baseline import SessionPnlBaseline, SessionPnlBaselineStore
from joker.runtime.entry_permission import EntryPermissionState
from joker.runtime.execution_runtime import ExecutionCommand, ExecutionRuntime
from joker.runtime.live_trading import LiveTradingRunner
from joker.runtime.live_preflight import run_production_preflight
from joker.runtime.order_action_gateway import (
    OrderActionGateway,
    OrderActionKind,
    OrderActionRequest,
)
from joker.schemas.domain import BrokerOrder, OptionContract, OrderIntent, Position
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock
from tests.broker._live_helpers import (
    contract_today,
    live_activation,
    live_app,
    live_env,
    make_intent,
    make_live_client,
    prepare_journal_for_intent,
)
from tests.cognitive.task2_canned import CONTRACT_ID


def _router() -> ModelRouter:
    profiles = {
        n: p.model_copy(update={"provider": "fake", "model": "x"})
        for n, p in default_model_profiles().items()
    }
    return ModelRouter(
        ModelRegistry(ModelsConfig(profiles=profiles), providers={"fake": FakeModelProvider()}),
        session_id="test",
    )


async def _exec_runtime(
    tmp_path,
    broker: BrokerClient,
    *,
    session_id: str = "s-truth",
    broker_account_id: str = "live-acct",
) -> tuple[ExecutionRuntime, SqliteLedgerStore, InProcessAsyncEventBus]:
    now = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    clock = FrozenExchangeClock(now, calendar=MarketCalendar())
    ledger = SqliteLedgerStore(tmp_path / f"{session_id}.db")
    await ledger.initialize()
    bus = InProcessAsyncEventBus()
    rt = ExecutionRuntime(
        broker=broker,
        ledger_store=ledger,
        event_bus=bus,
        clock=clock,
        session_id=session_id,
        broker_account_id=broker_account_id,
    )
    return rt, ledger, bus


class _UnknownSubmitBroker(PaperBroker):
    """Raises BrokerSubmissionUnknown on first submit; optional follow-up order."""

    def __init__(self, *, resolved_order: BrokerOrder | None = None) -> None:
        super().__init__(slippage_pct=0)
        self._resolved = resolved_order
        self._raised = False

    def submit_order(self, intent: OrderIntent) -> BrokerOrder:
        if not self._raised:
            self._raised = True
            raise BrokerSubmissionUnknown(intent.intent_id)
        return super().submit_order(intent)

    def get_order(self, order_id: str) -> BrokerOrder | None:
        if self._resolved is not None and order_id == self._resolved.order_id:
            return self._resolved
        return super().get_order(order_id)


@pytest.mark.asyncio
async def test_submission_unknown_does_not_append_rejection(tmp_path) -> None:
    broker = _UnknownSubmitBroker()
    rt, ledger, bus = await _exec_runtime(tmp_path, broker)
    intent = make_intent(intent_id="s" * 32)
    try:
        with pytest.raises(BrokerSubmissionUnknown):
            await rt.submit_execution_command(
                ExecutionCommand(client_order_id=intent.intent_id, intent=intent)
            )
        events = await ledger.get_by_session("s-truth")
        types = {e.event_type for e in events}
        assert LedgerEventType.SUBMISSION_UNKNOWN in types
        assert LedgerEventType.REJECTION not in types
    finally:
        await ledger.close()
        await bus.close()


@pytest.mark.asyncio
async def test_submission_unknown_does_not_publish_order_rejected(tmp_path) -> None:
    broker = _UnknownSubmitBroker()
    rt, ledger, bus = await _exec_runtime(tmp_path, broker)
    rejected: list[str] = []
    unknown: list[str] = []

    async def on_rejected(event) -> None:
        rejected.append(str(event.event_type))

    async def on_unknown(event) -> None:
        unknown.append(str(event.event_type))

    bus.subscribe(EventType.ORDER_REJECTED, on_rejected)
    bus.subscribe(EventType.ORDER_SUBMISSION_UNKNOWN, on_unknown)
    intent = make_intent(intent_id="r" * 32)
    try:
        with pytest.raises(BrokerSubmissionUnknown):
            await rt.submit_execution_command(
                ExecutionCommand(client_order_id=intent.intent_id, intent=intent)
            )
        await bus.drain()
        assert unknown == [EventType.ORDER_SUBMISSION_UNKNOWN]
        assert rejected == []
    finally:
        await ledger.close()
        await bus.close()


@pytest.mark.asyncio
async def test_submission_unknown_retains_reservation(tmp_path) -> None:
    from joker.broker.reconciliation import capital_reservation_release_allowed

    client, api, journal = make_live_client(tmp_path)
    api.place_timeout = True
    api.place_accepts_before_timeout = False
    intent = make_intent(intent_id="k" * 32)
    prepare_journal_for_intent(client, intent)
    with pytest.raises(BrokerSubmissionUnknown):
        client.submit_order(intent)
    stored = journal.get(client.account_id_hash, "k" * 32)
    assert stored is not None
    assert stored.status == "submission_unknown"
    assert capital_reservation_release_allowed(stored.status) is False


@pytest.mark.asyncio
async def test_reconciled_unknown_appends_acceptance(tmp_path) -> None:
    contract = contract_today()
    cid = f"SPY:{contract.expiration.isoformat()}:{contract.strike}:{contract.option_type}"
    resolved = BrokerOrder(
        order_id="a" * 32,
        intent_id="a" * 32,
        status="open",
        contract=contract,
        side="buy",
        quantity=1,
        limit_price=1.10,
    )
    broker = _UnknownSubmitBroker(resolved_order=resolved)
    rt, ledger, bus = await _exec_runtime(tmp_path, broker, broker_account_id="paper")
    intent = make_intent(intent_id="a" * 32)
    try:
        with pytest.raises(BrokerSubmissionUnknown):
            await rt.submit_execution_command(
                ExecutionCommand(client_order_id="a" * 32, intent=intent)
            )
        event = await rt.resolve_submission_unknown(
            "a" * 32,
            side="buy",
            quantity=Decimal("1"),
            contract_id=cid,
        )
        assert event is not None
        assert event.event_type == LedgerEventType.BROKER_ORDER_ACCEPTED
        events = await ledger.get_by_session("s-truth")
        assert any(e.event_type == LedgerEventType.BROKER_ORDER_ACCEPTED for e in events)
        assert not any(e.event_type == LedgerEventType.REJECTION for e in events)
    finally:
        await ledger.close()
        await bus.close()


@pytest.mark.asyncio
async def test_confirmed_unknown_rejection_appends_rejection(tmp_path) -> None:
    contract = contract_today()
    cid = f"SPY:{contract.expiration.isoformat()}:{contract.strike}:{contract.option_type}"
    rejected = BrokerOrder(
        order_id="b" * 32,
        intent_id="b" * 32,
        status="rejected",
        contract=contract,
        side="buy",
        quantity=1,
        limit_price=1.10,
    )
    broker = _UnknownSubmitBroker(resolved_order=rejected)
    rt, ledger, bus = await _exec_runtime(tmp_path, broker, broker_account_id="paper")
    intent = make_intent(intent_id="b" * 32)
    try:
        with pytest.raises(BrokerSubmissionUnknown):
            await rt.submit_execution_command(
                ExecutionCommand(client_order_id="b" * 32, intent=intent)
            )
        event = await rt.resolve_submission_unknown(
            "b" * 32,
            side="buy",
            quantity=Decimal("1"),
            contract_id=cid,
        )
        assert event is not None
        assert event.event_type == LedgerEventType.REJECTION
    finally:
        await ledger.close()
        await bus.close()


def test_live_broker_requires_activation(tmp_path) -> None:
    with pytest.raises((BrokerFactoryError, WebullLiveConfigError)):
        create_live_broker(
            live_app(db_path=tmp_path / "j.db"),
            live_env(),
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            capture_only=False,
            activation=None,
            journal_db_path=tmp_path / "j.db",
            skip_account_list_check=True,
        )


def test_live_broker_requires_durable_journal(tmp_path) -> None:
    with pytest.raises((BrokerFactoryError, WebullLiveConfigError)):
        create_live_broker(
            live_app(db_path=tmp_path / "j.db"),
            live_env(),
            trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
            capture_only=False,
            activation=live_activation(),
            journal_db_path=None,
            skip_account_list_check=True,
        )


def test_expired_activation_blocks_placement(tmp_path) -> None:
    activation = live_activation()
    expired = replace(
        activation,
        activated_at=activation.expires_at - timedelta(hours=2),
        expires_at=activation.expires_at - timedelta(hours=1),
    )
    with pytest.raises(WebullLiveConfigError, match="expired|inactive"):
        make_live_client(tmp_path, activation=expired)


def test_activation_objective_mismatch_blocks(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    with pytest.raises(WebullLiveConfigError, match="objective_id mismatch"):
        WebullLiveClient(
            live_env(),
            app_settings=live_app(),
            trade_api=api,
            activation=live_activation(),
            journal=__import__(
                "joker.persistence.broker_submission_journal",
                fromlist=["SyncBrokerSubmissionJournal"],
            ).SyncBrokerSubmissionJournal(tmp_path / "j.db"),
            skip_account_list_check=True,
            objective_id=str(uuid4()),
        )


def test_missing_journal_transition_fails_closed(tmp_path) -> None:
    client, _, _ = make_live_client(tmp_path)
    intent = make_intent(intent_id="m" * 32)
    with pytest.raises(BrokerError, match="missing journal"):
        client.submit_order(intent)


@pytest.mark.asyncio
async def test_startup_reconciles_actual_persisted_projection(tmp_path) -> None:
    client, api, journal = make_live_client(tmp_path)
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
                    "option_expire_date": date.today().isoformat(),
                    "option_exercise_price": "500",
                    "quantity": "1",
                }
            ],
        }
    )
    svc = BrokerReconciliationService(
        broker=client, journal=journal, account_id_hash=client.account_id_hash
    )
    empty_report = svc.reconcile(local_orders=[], local_positions=[])
    assert any(f.kind == "broker_position_missing_locally" for f in empty_report.findings)
    positions = client.list_positions()
    local_positions = [
        SimpleNamespace(
            contract_id=(
                f"{p.contract.symbol}:{p.contract.expiration.isoformat()}:"
                f"{p.contract.strike}:{p.contract.option_type}"
            ),
            quantity=Decimal(p.quantity),
        )
        for p in positions
    ]
    populated_report = svc.reconcile(local_orders=[], local_positions=local_positions)
    assert not any(
        f.kind == "broker_position_missing_locally" for f in populated_report.findings
    )


def test_restart_restores_working_order(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    client, _, journal = make_live_client(tmp_path, api=api)
    cid = "w" * 32
    api._orders[cid] = {
        "client_order_id": cid,
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
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
            contract_id=f"SPY:{date.today().isoformat()}:500.0:call",
            side="buy",
            quantity=1,
            limit_price="1.00",
            position_intent="BUY_TO_OPEN",
        )
    )
    journal.transition(
        account_id_hash=client.account_id_hash,
        client_order_id=cid,
        status="accepted",
        broker_order_id="WB-W",
    )
    client2, _, journal2 = make_live_client(tmp_path, api=api)
    stored = journal2.get(client2.account_id_hash, cid)
    assert stored is not None
    assert stored.status == "accepted"
    order = client2.get_order(cid)
    assert order is not None
    assert order.status in {"open", "pending", "partially_filled"}


def test_restart_restores_partial_fill(tmp_path) -> None:
    client, api, journal = make_live_client(tmp_path)
    cid = "p" * 32
    api._orders[cid] = {
        "client_order_id": cid,
        "status": "PARTIAL_FILLED",
        "order_id": "WB-P",
        "side": "BUY",
        "quantity": "2",
        "filled_quantity": "1",
        "limit_price": "1.10",
        "symbol": "SPY",
        "option_expire_date": date.today().isoformat(),
        "strike_price": "500",
        "option_type": "call",
        "avg_filled_price": "1.10",
    }
    journal.prepare(
        BrokerSubmissionRecord(
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
            contract_id=f"SPY:{date.today().isoformat()}:500.0:call",
            side="buy",
            quantity=2,
            limit_price="1.10",
            position_intent="BUY_TO_OPEN",
        )
    )
    journal.transition(
        account_id_hash=client.account_id_hash,
        client_order_id=cid,
        status="partially_filled",
        broker_order_id="WB-P",
    )
    order = client.get_order(cid)
    assert order is not None
    assert order.status == "partially_filled"
    assert order.filled_quantity == 1


def test_restart_restores_open_position(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
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
    client, _, _ = make_live_client(tmp_path, api=api)
    client2, _, _ = make_live_client(tmp_path, api=api)
    positions = client2.list_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 1


def test_restart_restores_working_exit(tmp_path) -> None:
    client, api, journal = make_live_client(tmp_path)
    entry = make_intent(intent_id="e" * 32)
    prepare_journal_for_intent(client, entry)
    client.submit_order(entry)
    cid = "x" * 32
    api._orders[cid] = {
        "client_order_id": cid,
        "status": "SUBMITTED",
        "order_id": "WB-X",
        "side": "SELL",
        "quantity": "1",
        "limit_price": "1.20",
        "symbol": "SPY",
        "option_expire_date": date.today().isoformat(),
        "strike_price": "500",
        "option_type": "call",
        "position_intent": "SELL_TO_CLOSE",
    }
    journal.prepare(
        BrokerSubmissionRecord(
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
            contract_id=f"SPY:{date.today().isoformat()}:500.0:call",
            side="sell",
            quantity=1,
            limit_price="1.20",
            position_intent="SELL_TO_CLOSE",
        )
    )
    journal.transition(
        account_id_hash=client.account_id_hash,
        client_order_id=cid,
        status="accepted",
        broker_order_id="WB-X",
    )
    order = client.get_order(cid)
    assert order is not None
    assert order.side == "sell"


def test_resolved_unknown_reruns_reconciliation(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.place_timeout = True
    api.place_accepts_before_timeout = False
    client, _, journal = make_live_client(tmp_path, api=api)
    intent = make_intent(intent_id="1" * 32)
    prepare_journal_for_intent(client, intent)
    with pytest.raises(BrokerSubmissionUnknown):
        client.submit_order(intent)
    svc = BrokerReconciliationService(
        broker=client, journal=journal, account_id_hash=client.account_id_hash
    )
    before = svc.reconcile()
    api._orders["1" * 32] = {
        "client_order_id": "1" * 32,
        "status": "SUBMITTED",
        "order_id": "WB-1",
        "side": "BUY",
        "quantity": "1",
        "limit_price": "1.10",
        "symbol": "SPY",
        "option_expire_date": date.today().isoformat(),
        "strike_price": "500",
        "option_type": "call",
    }
    resolved = svc.resolve_unknown_submissions()
    assert resolved
    after = svc.reconcile()
    assert after.unknown_submissions <= before.unknown_submissions


@pytest.mark.asyncio
async def test_unresolved_reconciliation_blocks_graph_entry() -> None:
    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="s",
        run_id="r",
        entry_permission=EntryPermissionState(
            permitted=False, reasons=("reconciliation_unresolved",)
        ),
    )
    graph = build_cognitive_graph(deps)
    state = initial_cycle_state(
        session_id="s",
        run_id="r",
        cycle_id="c",
        trigger_event_id="t",
        trigger_event_type="bar_closed",
        snapshot_id="snap",
    )
    result = await graph.ainvoke(state)
    errors = result.get("errors") or []
    assert any(getattr(e, "error_code", None) == "entry_permission_blocked" for e in errors)
    assert result.get("_block_new_entries") is True


@pytest.mark.asyncio
async def test_unresolved_reconciliation_still_allows_exit(tmp_path, monkeypatch) -> None:
    from joker.agents.cognitive.execution import CognitiveValidationError
    from joker.runtime import order_action_gateway as gw_mod

    contract_id = f"SPY:{date(2026, 7, 1).isoformat()}:500.0:call"
    runtime = MagicMock()
    runtime.submit_execution_command = AsyncMock(
        return_value=BrokerOrder(
            order_id="exit-1",
            intent_id="exit-1",
            status="pending",
            contract=OptionContract(
                symbol="SPY",
                expiration=date(2026, 7, 1),
                strike=500.0,
                option_type="call",
            ),
            side="sell",
            quantity=1,
            limit_price=1.10,
        )
    )
    runtime._broker = MagicMock()

    async def _fake_load(deps, snapshot_id):
        return (
            SimpleNamespace(
                snapshot_id=snapshot_id,
                trading_date=date(2026, 7, 1),
                option_surface_id=uuid4(),
            ),
            SimpleNamespace(usable_for_execution=False, quote_age_seconds=999),
            SimpleNamespace(surface_id=uuid4(), contracts=()),
            (),
        )

    def _fail_dq(self, request, **kwargs):
        raise CognitiveValidationError("data quality unusable")

    monkeypatch.setattr(gw_mod, "load_snapshot_truth", _fake_load)
    monkeypatch.setattr(OrderActionGateway, "_validate_and_compile", _fail_dq)

    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="gw",
        run_id="gw",
        execution_runtime=runtime,
        entry_permission=EntryPermissionState(permitted=False, reasons=("reconciliation",)),
        projection_loader=AsyncMock(
            return_value=SimpleNamespace(
                positions={
                    contract_id: SimpleNamespace(quantity=Decimal("1"), contract_id=contract_id)
                },
                orders=[],
            )
        ),
    )
    gateway = OrderActionGateway(deps)
    request = OrderActionRequest(
        action=OrderActionKind.EXIT,
        client_order_id="exit-1",
        contract_id=contract_id,
        side="sell",
        quantity=1,
        order_type="limit",
        limit_price=1.10,
        snapshot_id="snap",
        allow_degraded_exit=True,
        degraded_exit_reason="data quality",
    )
    result = await gateway.submit(request)
    assert result.submitted is True
    assert result.degraded_exit is True


@pytest.mark.asyncio
async def test_degraded_live_exit_sets_sell_to_close(tmp_path, monkeypatch) -> None:
    from joker.agents.cognitive.execution import CognitiveValidationError
    from joker.runtime import order_action_gateway as gw_mod

    contract_id = f"SPY:{date(2026, 7, 1).isoformat()}:500.0:call"
    captured: list[ExecutionCommand] = []

    async def _capture(cmd: ExecutionCommand) -> BrokerOrder:
        captured.append(cmd)
        return BrokerOrder(
            order_id=cmd.client_order_id,
            intent_id=cmd.intent.intent_id,
            status="pending",
            contract=cmd.intent.contract,
            side=cmd.intent.side,
            quantity=cmd.intent.quantity,
            limit_price=cmd.intent.limit_price,
        )

    runtime = MagicMock()
    runtime.submit_execution_command = _capture
    runtime._broker = MagicMock()

    async def _fake_load(deps, snapshot_id):
        return (
            SimpleNamespace(
                snapshot_id=snapshot_id,
                trading_date=date(2026, 7, 1),
                option_surface_id=uuid4(),
            ),
            SimpleNamespace(usable_for_execution=False),
            SimpleNamespace(surface_id=uuid4(), contracts=()),
            (),
        )

    def _fail_dq(self, request, **kwargs):
        raise CognitiveValidationError("option surface unusable")

    monkeypatch.setattr(gw_mod, "load_snapshot_truth", _fake_load)
    monkeypatch.setattr(OrderActionGateway, "_validate_and_compile", _fail_dq)

    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="gw",
        run_id="gw",
        execution_runtime=runtime,
        projection_loader=AsyncMock(
            return_value=SimpleNamespace(
                positions={
                    contract_id: SimpleNamespace(quantity=Decimal("2"), contract_id=contract_id)
                },
                orders=[],
            )
        ),
    )
    gateway = OrderActionGateway(deps)
    request = OrderActionRequest(
        action=OrderActionKind.EXIT,
        client_order_id="d" * 32,
        contract_id=contract_id,
        side="sell",
        quantity=2,
        order_type="limit",
        limit_price=1.10,
        snapshot_id="snap",
        allow_degraded_exit=True,
        degraded_exit_reason="option surface unusable",
    )
    result = await gateway.submit(request)
    assert result.submitted is True
    assert captured[0].intent.position_intent == "SELL_TO_CLOSE"
    assert captured[0].intent.side == "sell"


@pytest.mark.asyncio
async def test_degraded_live_exit_reaches_capture_broker(tmp_path, monkeypatch) -> None:
    from joker.agents.cognitive.execution import CognitiveValidationError
    from joker.runtime import order_action_gateway as gw_mod

    client, api, _ = make_live_client(tmp_path, capture_only=True)
    contract_id = f"SPY:{date.today().isoformat()}:500.0:call"
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
                    "option_expire_date": date.today().isoformat(),
                    "option_exercise_price": "500",
                    "quantity": "1",
                }
            ],
        }
    )
    rt, ledger, bus = await _exec_runtime(tmp_path, client, session_id="live-cap")
    runtime = rt

    async def _fake_load(deps, snapshot_id):
        return (
            SimpleNamespace(
                snapshot_id=snapshot_id,
                trading_date=date.today(),
                option_surface_id=uuid4(),
            ),
            SimpleNamespace(usable_for_execution=False),
            SimpleNamespace(surface_id=uuid4(), contracts=()),
            (),
        )

    def _fail_dq(self, request, **kwargs):
        raise CognitiveValidationError("data quality stale")

    monkeypatch.setattr(gw_mod, "load_snapshot_truth", _fake_load)
    monkeypatch.setattr(OrderActionGateway, "_validate_and_compile", _fail_dq)

    deps = CognitiveGraphDeps(
        router=_router(),
        config=CognitiveGraphSettings(),
        session_id="live-cap",
        run_id="gw",
        execution_runtime=runtime,
        projection_loader=AsyncMock(
            return_value=SimpleNamespace(
                positions={
                    contract_id: SimpleNamespace(quantity=Decimal("1"), contract_id=contract_id)
                },
                orders=[],
            )
        ),
    )
    gateway = OrderActionGateway(deps)
    request = OrderActionRequest(
        action=OrderActionKind.EXIT,
        client_order_id="c" * 32,
        contract_id=contract_id,
        side="sell",
        quantity=1,
        order_type="limit",
        limit_price=1.10,
        snapshot_id="snap",
        allow_degraded_exit=True,
        degraded_exit_reason="data quality stale",
    )
    try:
        result = await gateway.submit(request)
        assert result.submitted is True
        assert client.captured_payloads
        assert client.captured_payloads[-1]["position_intent"] == "SELL_TO_CLOSE"
        assert api.placed == []
    finally:
        await ledger.close()
        await bus.close()


def test_open_orders_reconstruct_without_memory_cache(tmp_path) -> None:
    client, api, journal = make_live_client(tmp_path)
    cid = "o" * 32
    api._orders[cid] = {
        "client_order_id": cid,
        "status": "SUBMITTED",
        "order_id": "WB-O",
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
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="prepared",
            contract_id=f"SPY:{date.today().isoformat()}:500.0:call",
            side="buy",
            quantity=1,
            limit_price="1.00",
            position_intent="BUY_TO_OPEN",
        )
    )
    journal.transition(
        account_id_hash=client.account_id_hash,
        client_order_id=cid,
        status="accepted",
        broker_order_id="WB-O",
    )
    client._orders.clear()
    open_orders = client.list_open_orders()
    assert len(open_orders) == 1
    assert open_orders[0].order_id == cid


def test_order_detail_failure_is_not_local_truth(tmp_path) -> None:
    client, api, journal = make_live_client(tmp_path)
    cid = "f" * 32
    journal.prepare(
        BrokerSubmissionRecord(
            client_order_id=cid,
            broker_mode="webull_live",
            account_id_hash=client.account_id_hash,
            status="accepted",
            contract_id=f"SPY:{date.today().isoformat()}:500.0:call",
            side="buy",
            quantity=1,
        )
    )
    api._orders.pop(cid, None)

    def _fail_detail(account_id: str, client_order_id: str) -> dict:
        raise TimeoutError("detail unavailable")

    api.get_order_detail = _fail_detail  # type: ignore[method-assign]
    assert client.get_order(cid) is None


def test_partial_fill_is_not_mapped_to_open() -> None:
    assert map_webull_order_status("PARTIAL_FILLED") == "partially_filled"
    assert map_webull_order_status("SUBMITTED") == "open"


@pytest.mark.asyncio
async def test_partial_fill_records_incremental_fill(tmp_path) -> None:
    contract = contract_today()
    broker = PaperBroker(slippage_pct=0)
    rt, ledger, bus = await _exec_runtime(tmp_path, broker)
    cid = "i" * 32
    order = BrokerOrder(
        order_id=cid,
        intent_id=cid,
        status="partially_filled",
        contract=contract,
        side="buy",
        quantity=2,
        limit_price=1.10,
        filled_quantity=1,
        remaining_quantity=1,
        average_fill_price=1.10,
    )
    broker._orders[cid] = order
    try:
        partial = order.model_copy(
            update={"filled_quantity": 1, "status": "partially_filled", "average_fill_price": 1.10}
        )
        await rt.on_broker_update(partial, client_order_id=cid)
        more = order.model_copy(
            update={
                "filled_quantity": 2,
                "remaining_quantity": 0,
                "status": "filled",
                "average_fill_price": 1.10,
            }
        )
        await rt.on_broker_update(more, client_order_id=cid)
        events = await ledger.get_by_session("s-truth")
        partials = [e for e in events if e.event_type == LedgerEventType.PARTIAL_FILL]
        finals = [e for e in events if e.event_type == LedgerEventType.FINAL_FILL]
        assert len(partials) == 1
        assert len(finals) == 1
        assert partials[0].quantity == Decimal("1")
        assert finals[0].quantity == Decimal("1")
    finally:
        await ledger.close()
        await bus.close()


@pytest.mark.asyncio
async def test_partial_fill_converts_partial_reservation(tmp_path) -> None:
    apply_objective_migrations(tmp_path / "obj.db")
    repo = ObjectiveRepository(tmp_path / "obj.db")
    svc = SessionObjectiveService(repo, require_positive_expected_value=False)
    definition = await svc.create_objective(
        session_id="s-partial",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(timezone.utc) + timedelta(hours=4),
        max_concurrent_positions=2,
        accepted_total_loss_risk=True,
    )
    await svc.confirm_objective(definition.objective_id)
    state = await svc.get_state()
    await svc.reserve_for_order(
        client_order_id="cl-1",
        contract_id=CONTRACT_ID,
        quantity=2,
        premium_per_contract_usd=Decimal("1.10"),
        objective_state_version=state.version,
    )
    state = await svc.apply_verified_fill(
        client_order_id="cl-1",
        fill_quantity=1,
        fill_price=Decimal("1.10"),
        remaining_working_quantity=1,
        contract_id=CONTRACT_ID,
    )
    exposure = repo.get_exposure_by_client_order("cl-1")
    assert exposure is not None
    assert exposure.status == "partial"
    assert exposure.filled_quantity == 1
    assert exposure.working_quantity == 1
    assert state.working_order_reservation_usd > Decimal("0")


@pytest.mark.asyncio
async def test_polling_event_source_starts_and_stops_with_runner(tmp_path) -> None:
    from joker.broker.account_truth import hash_account_id
    from joker.runtime.live_activation import create_live_activation

    app = AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(tmp_path / "poll.db"),
        broker={"provider": "webull_live"},
        evolution={"enabled": True},
        objective={"enabled": True, "require_positive_expected_value": False},
        cognitive_graph={"enabled": True},
    )
    apply_task1_migrations(tmp_path / "poll.db")
    apply_objective_migrations(tmp_path / "poll.db")
    obj_repo = ObjectiveRepository(tmp_path / "poll.db")
    objective_service = SessionObjectiveService(
        obj_repo, require_positive_expected_value=False
    )
    definition = await objective_service.create_objective(
        session_id="poll-session",
        authorised_capital_usd=500,
        target_profit_pct=10,
        deadline_exchange_time=datetime.now(tz=timezone.utc) + timedelta(hours=4),
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    await objective_service.confirm_objective(definition.objective_id)
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=definition.objective_id,
        authorized_capital_usd=Decimal("500"),
    )
    runner = LiveTradingRunner(
        app_settings=app,
        env=live_env(),
        objective_service=objective_service,
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        capture_only=True,
        db_path=tmp_path / "poll.db",
        poll_interval_seconds=0.05,
    )
    session = await runner.start(
        start_cognitive_agent=False,
        start_evolution_workers=False,
        start_market_loop=False,
        session_id="poll-session",
        fake_model_provider=__import__(
            "joker.models.fake_provider", fromlist=["FakeModelProvider"]
        ).FakeModelProvider(available=True),
    )
    assert runner._poll_task is not None
    assert runner._poller is not None
    poll_task = runner._poll_task
    await runner.shutdown()
    assert poll_task.done() or poll_task.cancelled()
    assert session is not None


def test_session_pnl_baseline_survives_restart(tmp_path) -> None:
    from joker.broker.account_truth import hash_account_id

    db = tmp_path / "journal.db"
    account_hash = hash_account_id("LIVE_ACCT_1")
    trading_date = date.today().isoformat()
    session_id = "sess-restart"
    store = SessionPnlBaselineStore(db)
    baseline = SessionPnlBaseline(
        account_id_hash=account_hash,
        trading_date=trading_date,
        session_id=session_id,
        starting_nlv=Decimal("100000"),
        starting_cash=Decimal("50000"),
        captured_at=datetime.now(timezone.utc),
    )
    store.put(baseline)
    client, _, _ = make_live_client(tmp_path, account_id="LIVE_ACCT_1")
    assert client._baseline_store is not None
    client._session_id = session_id
    loaded = client._baseline_store.get(
        account_id_hash=account_hash,
        trading_date=trading_date,
        session_id=session_id,
    )
    assert loaded is not None
    assert loaded.starting_nlv == Decimal("100000")
    client2, _, _ = make_live_client(tmp_path, account_id="LIVE_ACCT_1")
    client2._session_id = session_id
    loaded2 = client2._baseline_store.get(
        account_id_hash=account_hash,
        trading_date=trading_date,
        session_id=session_id,
    )
    assert loaded2 is not None
    assert loaded2.starting_nlv == Decimal("100000")


def test_unavailable_session_pnl_is_not_zero(tmp_path) -> None:
    api = create_mock_live_trade_api("LIVE_ACCT_1")
    api.net_liquidation_usd = None
    client, _, _ = make_live_client(tmp_path, api=api)
    if client._baseline_store is not None:
        conn = __import__("sqlite3").connect(client._baseline_store._db_path)
        conn.execute("DELETE FROM session_pnl_baseline")
        conn.commit()
        conn.close()
    truth = client.get_account_truth()
    assert truth.session_pnl_available is False
    assert truth.session_pnl_usd is None
    available, value = client.get_daily_pnl_available()
    assert available is False
    assert value is None


def test_health_does_not_hardcode_market_state(tmp_path) -> None:
    from joker.broker.account_truth import hash_account_id
    from joker.runtime.live_activation import create_live_activation

    app = AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(tmp_path / "health.db"),
        broker={"provider": "webull_live"},
        evolution={"enabled": True},
        objective={"enabled": True},
        cognitive_graph={"enabled": True},
    )
    activation = create_live_activation(
        account_id_hash=hash_account_id("LIVE_ACCT_1"),
        objective_id=uuid4(),
        authorized_capital_usd=Decimal("1000"),
    )
    runner = LiveTradingRunner(
        app_settings=app,
        env=live_env(),
        objective_service=MagicMock(),
        activation=activation,
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        capture_only=True,
        db_path=tmp_path / "health.db",
    )
    health = runner.health()
    assert health.market_data_healthy is False
    assert health.option_surface_healthy is False


def test_preflight_checks_actual_database(tmp_path) -> None:
    db = tmp_path / "preflight.db"
    apply_task1_migrations(db)
    app = AppSettings(
        mode=SafetyMode.LIVE_GATED,
        live_trading_enabled=True,
        db_path=str(db),
        broker={"provider": "webull_live"},
    )
    report = run_production_preflight(
        app_settings=app,
        env=live_env(),
        trade_api=create_mock_live_trade_api("LIVE_ACCT_1"),
        skip_network=False,
        check_market_data=False,
    )
    assert any("database: ok" in c for c in report.checks)
    assert report.database_ok is True
    assert report.operational_ready is False

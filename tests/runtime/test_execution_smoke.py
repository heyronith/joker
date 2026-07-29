"""Spy-based execution-smoke tests — no live Webull contact."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.broker.webull import WebullClient
from joker.config.settings import AppSettings, EnvSettings
from joker.ledger.schemas import LedgerEventType
from joker.runtime.execution_runtime import ExecutionCommand
from joker.runtime.execution_smoke import (
    ExecutionSmokeError,
    ExecutionSmokeRunner,
    build_smoke_execution_command,
    require_non_live_account,
    require_odte_open_window,
    require_sandbox_trade_environment,
    require_webull_paper_selection,
    select_smoke_contract,
)
from joker.schemas.domain import BrokerOrder, OptionContract
from joker.schemas.options_data import OptionContractMetadata, OptionSnapshot


def _env(**overrides: object) -> EnvSettings:
    data = {
        "OPENAI_API_KEY": "test-key",
        "WEBULL_PAPER_TRADING_ENABLED": "true",
        "WEBULL_PAPER_ACCOUNT_ID": "PAPER_ACCT_TEST",
        "WEBULL_LIVE_TRADING_ENABLED": "false",
        "WEBULL_TRADE_API_ENV": "sandbox",
        "WEBULL_TRADE_APP_KEY": "trade-key",
        "WEBULL_TRADE_APP_SECRET": "trade-secret",
        "WEBULL_TRADE_ACCESS_TOKEN": "trade-token",
    }
    data.update({k: str(v) for k, v in overrides.items()})
    return EnvSettings.model_validate(data)


def _app() -> AppSettings:
    return AppSettings(
        mode=SafetyMode.PAPER,
        live_trading_enabled=False,
        broker={"provider": "webull_paper", "webull_paper_trading_enabled": True},
        cognitive_graph={"enabled": True},
        evolution={"enabled": True},
    )


def _snap(*, ask: float = 0.25, age_s: float = 1.0, option_type: str = "call") -> OptionSnapshot:
    now = datetime.now(timezone.utc)
    return OptionSnapshot(
        contract=OptionContractMetadata(
            underlying_symbol="SPY",
            expiration=date.today(),
            strike=500.0,
            option_type=option_type,  # type: ignore[arg-type]
            contract_id="SPY_TEST_CONTRACT",
        ),
        bid=0.20,
        ask=ask,
        quote_timestamp=now - timedelta(seconds=age_s),
    )


class SpyWebullPaperBroker(WebullClient):
    LIVE_CALLS_ENABLED = False

    def __init__(self) -> None:
        self._submit_calls: list = []
        self._cancel_calls: list[str] = []
        self._orders: dict[str, BrokerOrder] = {}
        self._intent_by_order: dict[str, object] = {}
        self._account_id = "PAPER_ACCT_TEST"
        self._positions: list = []
        self.open_orders: list[BrokerOrder] = []

    def list_accounts_raw(self) -> list[dict]:
        return [
            {
                "account_id": "PAPER_ACCT_TEST",
                "account_label": "Sandbox Paper",
                "account_class": "CASH",
            }
        ]

    def list_open_orders(self) -> list[BrokerOrder]:
        return list(self.open_orders)

    def list_positions(self) -> list:
        return list(self._positions)

    def submit_order(self, intent) -> BrokerOrder:
        self._submit_calls.append(intent)
        order = BrokerOrder(
            order_id=f"wb-{len(self._submit_calls)}",
            intent_id=intent.intent_id,
            status="open",
            side=intent.side,
            quantity=intent.quantity,
            limit_price=intent.limit_price,
            contract=intent.contract,
        )
        self._orders[order.order_id] = order
        self.open_orders = [order]
        return order

    def cancel_order(self, order_id: str) -> BrokerOrder:
        self._cancel_calls.append(order_id)
        order = self._orders[order_id]
        cancelled = order.model_copy(update={"status": "cancelled"})
        self._orders[order_id] = cancelled
        self.open_orders = []
        return cancelled

    def get_order(self, order_id: str) -> BrokerOrder | None:
        return self._orders.get(order_id)

    def get_account_balance(self) -> float:
        return 50.0

    def get_daily_pnl(self) -> float:
        return 0.0

    def get_daily_pnl_available(self) -> tuple[bool, float | None]:
        return True, 0.0

    def close(self) -> None:
        self._closed = True


def test_prod_trade_env_rejected_before_broker() -> None:
    env = _env(WEBULL_TRADE_API_ENV="prod")
    with pytest.raises(ExecutionSmokeError, match="sandbox"):
        require_sandbox_trade_environment(env)


def test_uat_trade_env_rejected() -> None:
    env = _env(WEBULL_TRADE_API_ENV="uat")
    with pytest.raises(ExecutionSmokeError, match="sandbox"):
        require_sandbox_trade_environment(env)


def test_live_classified_account_rejected() -> None:
    env = _env().model_copy(update={"webull_trade_api_env": "prod"})
    with pytest.raises(ExecutionSmokeError, match="live brokerage"):
        require_non_live_account(
            env,
            accounts=[
                {
                    "account_id": "PAPER_ACCT_TEST",
                    "account_label": "Individual Cash",
                    "account_class": "INDIVIDUAL_CASH",
                }
            ],
        )


def test_missing_confirmation_rejected() -> None:
    runner = ExecutionSmokeRunner(_app(), _env(), require_sandbox=True, confirm_place=False)
    with pytest.raises(ExecutionSmokeError, match="confirm-place"):
        runner.run()


def test_paper_broker_never_selected() -> None:
    app = _app()
    env = _env()
    with pytest.raises(ExecutionSmokeError, match="PaperBroker|webull_paper"):
        require_webull_paper_selection(
            app, env, broker=PaperBroker()  # type: ignore[arg-type]
        )


def test_select_smoke_contract_requires_fresh_quote() -> None:
    with pytest.raises(ExecutionSmokeError, match="quote"):
        select_smoke_contract(call_snap=_snap(age_s=30), put_snap=None)


def test_select_smoke_contract_requires_min_ask() -> None:
    with pytest.raises(ExecutionSmokeError, match="quote"):
        select_smoke_contract(call_snap=_snap(ask=0.05), put_snap=None)


def test_routes_through_execution_runtime(tmp_path: Path, monkeypatch) -> None:
    broker = SpyWebullPaperBroker()
    submitted: list[ExecutionCommand] = []

    class FakeBridge:
        def __init__(self) -> None:
            self.session_id = "smoke-sess"
            self.supervisor = SimpleNamespace(
                market_runtime=object(),
                execution_runtime=SimpleNamespace(
                    poll_order_status=_async_order("cancelled"),
                    run_reconciliation=_async_const(SimpleNamespace(is_consistent=True)),
                ),
                ledger_store=SimpleNamespace(get_by_session=_async_ledger()),
                data_quality_repository=None,
                event_bus=None,
            )
            self._closed = False

        def submit_execution_command(self, command: ExecutionCommand) -> BrokerOrder:
            submitted.append(command)
            return broker.submit_order(command.intent)

        def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
            order_id = list(broker._orders.keys())[-1]
            return broker.cancel_order(order_id)

        def run_coro(self, coro):
            import asyncio

            if asyncio.iscoroutine(coro):
                return asyncio.run(coro)
            return coro

        def shutdown(self) -> None:
            self._closed = True

    fake_bridge = FakeBridge()

    async def _health():
        return SimpleNamespace(status="healthy", local_provider_available=True)

    def fake_start(self, broker_arg):
        cognitive = SimpleNamespace(health=_health)
        evolution = SimpleNamespace(
            settings=SimpleNamespace(enabled=True),
            _prepared=True,
            shutdown=_async_const(None),
        )
        return fake_bridge, cognitive, evolution

    market = MagicMock()
    market.authenticate.return_value = True
    market.get_latest_snapshot.return_value = SimpleNamespace(price=500.0)
    options = MagicMock()
    options.authenticate.return_value = True
    options.fetch_atm_snapshots.return_value = (_snap(), None)

    monkeypatch.setattr(ExecutionSmokeRunner, "_start_runtimes", fake_start)
    monkeypatch.setattr(
        "joker.runtime.execution_smoke.require_webull_paper_selection",
        lambda *a, **k: SimpleNamespace(kind="webull_paper", client=broker),
    )
    monkeypatch.setattr("joker.runtime.execution_smoke.require_odte_open_window", lambda **_: None)
    result = ExecutionSmokeRunner(
        _app(),
        _env(),
        require_sandbox=True,
        confirm_place=True,
        broker=broker,
        market_provider=market,
        options_provider=options,
        db_path=tmp_path / "smoke.db",
    ).run()
    assert result.passed is True, result.errors
    assert submitted, "must submit via ExecutionRuntime/bridge path"
    assert isinstance(submitted[0], ExecutionCommand)
    assert broker._submit_calls
    assert broker._cancel_calls
    assert result.final_open_orders == 0
    assert result.final_positions == 0
    assert result.fill_detected is False
    assert getattr(broker, "_closed", False) is True
    assert fake_bridge._closed is True


def test_fill_fails_smoke(tmp_path: Path, monkeypatch) -> None:
    broker = SpyWebullPaperBroker()

    class FillingBridge:
        session_id = "s"

        def __init__(self) -> None:
            self.supervisor = SimpleNamespace(
                market_runtime=object(),
                execution_runtime=SimpleNamespace(
                    poll_order_status=_async_order("filled"),
                    run_reconciliation=_async_const(SimpleNamespace(is_consistent=True)),
                ),
                ledger_store=SimpleNamespace(get_by_session=_async_ledger()),
            )

        def submit_execution_command(self, command: ExecutionCommand) -> BrokerOrder:
            if command.intent.candidate_id == "execution-smoke-flatten":
                broker._positions = []
                return BrokerOrder(
                    order_id="flat-1",
                    intent_id=command.intent.intent_id,
                    status="filled",
                    side="sell",
                    quantity=1,
                    limit_price=0.01,
                    contract=command.intent.contract,
                )
            order = broker.submit_order(command.intent)
            filled = order.model_copy(update={"status": "filled"})
            broker._orders[order.order_id] = filled
            broker._positions = [object()]
            broker.open_orders = []
            return filled

        def cancel_order(self, *, client_order_id: str) -> BrokerOrder:
            pytest.fail("cancel should not be required after immediate fill path")

        def run_coro(self, coro):
            import asyncio

            if asyncio.iscoroutine(coro):
                return asyncio.run(coro)
            return coro

        def shutdown(self) -> None:
            return None

    bridge = FillingBridge()

    async def _health():
        return SimpleNamespace(status="healthy", local_provider_available=True)

    def fake_start(self, broker_arg):
        return (
            bridge,
            SimpleNamespace(health=_health),
            SimpleNamespace(
                settings=SimpleNamespace(enabled=True),
                _prepared=True,
                shutdown=_async_const(None),
            ),
        )

    market = MagicMock()
    market.authenticate.return_value = True
    market.get_latest_snapshot.return_value = SimpleNamespace(price=500.0)
    options = MagicMock()
    options.authenticate.return_value = True
    options.fetch_atm_snapshots.return_value = (_snap(), None)
    monkeypatch.setattr(ExecutionSmokeRunner, "_start_runtimes", fake_start)
    monkeypatch.setattr(
        "joker.runtime.execution_smoke.require_webull_paper_selection",
        lambda *a, **k: SimpleNamespace(kind="webull_paper", client=broker),
    )
    monkeypatch.setattr("joker.runtime.execution_smoke.require_odte_open_window", lambda **_: None)

    result = ExecutionSmokeRunner(
        _app(),
        _env(),
        require_sandbox=True,
        confirm_place=True,
        broker=broker,
        market_provider=market,
        options_provider=options,
        db_path=tmp_path / "fill.db",
    ).run()
    assert result.passed is False
    assert result.fill_detected is True
    assert result.flattened is True


def test_odte_open_cutoff_fails_closed() -> None:
    from zoneinfo import ZoneInfo

    late = datetime(2026, 7, 29, 15, 41, tzinfo=ZoneInfo("America/New_York"))
    with pytest.raises(ExecutionSmokeError, match="15:40"):
        require_odte_open_window(now_et=late)


def test_build_smoke_command_unique_client_id() -> None:
    contract = OptionContract(
        symbol="SPY",
        expiration=date.today(),
        strike=500.0,
        option_type="call",
        is_0dte=True,
    )
    a = build_smoke_execution_command(contract)
    b = build_smoke_execution_command(contract)
    assert a.client_order_id != b.client_order_id
    assert a.client_order_id.startswith("smk")
    assert len(a.client_order_id) <= 32
    assert a.intent.limit_price == 0.01


def _async_const(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


def _async_order(status: str):
    async def _inner(client_order_id: str):
        return BrokerOrder(
            order_id="wb-1",
            intent_id=client_order_id,
            status=status,  # type: ignore[arg-type]
            side="buy",
            quantity=1,
            limit_price=0.01,
            contract=OptionContract(
                symbol="SPY",
                expiration=date.today(),
                strike=500.0,
                option_type="call",
                is_0dte=True,
            ),
        )

    return _inner


def _async_ledger():
    async def get_by_session(session_id: str):
        class AnyId:
            def __eq__(self, other: object) -> bool:
                return True

            def __ne__(self, other: object) -> bool:
                return False

        types = [
            LedgerEventType.ORDER_SUBMISSION_REQUESTED,
            LedgerEventType.BROKER_ORDER_ACCEPTED,
            LedgerEventType.CANCELLATION,
        ]
        return [
            SimpleNamespace(event_type=et, client_order_id=AnyId()) for et in types
        ]

    return get_by_session

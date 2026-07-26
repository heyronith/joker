"""Live order lifecycle: pending exit, verified entry, cancellation via ExecutionRuntime."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from joker.app.safety import SafetyMode
from joker.broker.interface import PaperBroker
from joker.execution.exit_manager import ExitManager, OpenTradeContext
from joker.execution.option_selector import OptionSelector
from joker.features.engine import FeatureEngine
from joker.risk.capital import CapitalBudget, CapitalPlan
from joker.risk.governor import RiskGovernor
from joker.runtime.compatibility import CompatibilityLivePaperBridge, ExecutionDelegatingBroker
from joker.runtime.execution_runtime import contract_id_for
from joker.runtime.market_handler import MarketEventHandler, PendingEntry, PendingExit
from joker.runtime.market_runtime import MarketTickResult
from joker.runtime.reactive_engine import ReactiveEngine
from joker.schemas.domain import (
    OptionContract,
    OptionQuote,
    OrderIntent,
    PlaybookSetup,
    RiskConfig,
    TradeCandidate,
)
from joker.schemas.replay import OptionQuoteEvent

ET = ZoneInfo("America/New_York")


def _risk() -> RiskConfig:
    return RiskConfig(
        max_daily_loss_usd=500,
        max_trades_per_day=10,
        max_open_positions=1,
        max_premium_usd=500,
        max_spread_pct=0.5,
        quote_max_age_seconds=60,
        allowed_symbol="SPY",
        kill_switch=False,
        allow_delayed_quotes=True,
        feed_max_silence_seconds=120,
        delayed_quote_max_age_seconds=300,
        policy="strict",
    )


def _contract() -> OptionContract:
    return OptionContract(
        symbol="SPY",
        expiration=date(2026, 7, 1),
        strike=500.0,
        option_type="call",
        is_0dte=True,
    )


def _setup() -> PlaybookSetup:
    return PlaybookSetup(
        setup_id="s1",
        name="t",
        direction="long_call",
        enabled=True,
        stop_rule="0.5",
        take_profit_rule="1.0",
        stop_pct=0.3,
        take_profit_pct=0.5,
        time_stop_minutes=30,
    )


def _candidate(contract: OptionContract, now: datetime, quantity: int = 1) -> TradeCandidate:
    quote = OptionQuote(contract=contract, bid=0.9, ask=1.0, timestamp=now)
    return TradeCandidate(
        run_id="life",
        setup_id="s1",
        contract=contract,
        quote=quote,
        direction="long_call",
        entry_limit_price=1.0,
        stop_price=0.7,
        take_profit_price=1.5,
        quantity=quantity,
    )


def _handler(broker: PaperBroker, bridge: CompatibilityLivePaperBridge):
    exec_broker = ExecutionDelegatingBroker(inner=broker, bridge=bridge)
    risk = _risk()
    reactive = ReactiveEngine(
        RiskGovernor(risk, SafetyMode.PAPER, live_enabled=False), exec_broker
    )
    capital = CapitalBudget(
        plan=CapitalPlan(
            authorized_usd=1000,
            target_profit_pct=10,
            max_concurrent_positions=1,
            max_contracts_per_trade=5,
            min_contracts_per_trade=1,
        )
    )

    class _Prov:
        current_time = datetime(2026, 7, 1, 11, 0, tzinfo=ET)

        def get_latest_snapshot(self):
            return None

    handler = MarketEventHandler(
        provider=_Prov(),  # type: ignore[arg-type]
        reactive_engine=reactive,
        risk_governor=reactive.risk_governor,
        broker=exec_broker,
        feature_engine=FeatureEngine(),
        option_selector=OptionSelector(),
        exit_manager=ExitManager(),
        mode=SafetyMode.PAPER,
        run_id="life-1",
        capital_budget=capital,
        task1_bridge=bridge,
    )
    return handler, capital, exec_broker


def test_unfilled_exit_keeps_position_and_capital(tmp_path: Path) -> None:
    class OpenExitBroker(PaperBroker):
        def _should_fill(self, intent: OrderIntent, fill_price: float) -> bool:
            if intent.side == "sell":
                return False
            return super()._should_fill(intent, fill_price)

    broker = OpenExitBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "u.db", session_id="u1"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        contract = _contract()
        buy = OrderIntent(
            candidate_id="b",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.0,
        )
        buy_order = exec_broker.submit_order(buy)
        assert buy_order.status == "filled"
        cid = contract_id_for(contract)
        handler.state.open_trade = OpenTradeContext(
            position_id="p1",
            entry_price=1.0,
            stop_price=0.5,
            take_profit_price=2.0,
            entry_time=handler.provider.current_time,
            time_stop_minutes=30,
            quantity=1,
            reserved_notional_usd=100.0,
        )
        capital.reserve(100.0)
        reserved_before = capital.reserved_usd
        realized_before = capital.realized_pnl_usd

        sell = OrderIntent(
            candidate_id="s",
            contract=contract,
            side="sell",
            order_type="limit",
            quantity=1,
            limit_price=1.5,
        )
        sell_order = exec_broker.submit_order(sell)
        assert sell_order.status == "open"
        handler.state.pending_exit = PendingExit(
            client_order_id=sell.intent_id,
            broker_order_id=sell_order.order_id,
            position_id="p1",
            contract_id=cid,
            requested_quantity=1,
            reason="stop_loss",
            submitted_at=handler.provider.current_time,
        )
        handler._reconcile_pending_exit()
        assert handler.state.open_trade is not None
        assert handler.state.trades_exited == 0
        assert capital.reserved_usd == reserved_before
        assert capital.realized_pnl_usd == realized_before
        assert handler.state.pending_exit is not None
        proj = bridge.project_session()
        assert proj.positions[cid].quantity == Decimal("1")
        assert proj.positions[cid].open is True
    finally:
        bridge.shutdown()


def test_rejected_exit_keeps_position(tmp_path: Path) -> None:
    """Degraded truth layer must not leave a stuck PendingExit via _try_exit."""
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "r.db", session_id="r1"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        contract = _contract()
        buy = OrderIntent(
            candidate_id="b",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.0,
        )
        exec_broker.submit_order(buy)
        now = handler.provider.current_time
        handler.state.open_trade = OpenTradeContext(
            position_id="p1",
            entry_price=1.0,
            stop_price=0.5,
            take_profit_price=1.5,
            entry_time=now,
            time_stop_minutes=30,
            quantity=1,
            reserved_notional_usd=100.0,
        )
        capital.reserve(100.0)
        logs: list[tuple[str, dict]] = []
        handler._on_log = lambda et, p=None: logs.append((et, p or {}))

        bridge.health.degraded = True
        quote = OptionQuoteEvent(
            timestamp=now,
            contract_id=contract_id_for(contract),
            expiration=contract.expiration,
            strike=contract.strike,
            option_type="call",
            bid=1.6,
            ask=1.7,
            mid=1.65,
            spread_pct=6.0,
            quote_timestamp=now,
        )
        decision = handler._try_exit(quote)
        assert decision is None
        assert handler.state.pending_exit is None
        assert handler.state.open_trade is not None
        assert handler.state.trades_exited == 0
        assert capital.reserved_usd == 100.0
        assert any(et == "exit.order_failed" for et, _ in logs)
        assert not any(et == "exit.executed" for et, _ in logs)
        assert not any(et == "exit.order_submitted" for et, _ in logs)
    finally:
        bridge.shutdown()


def test_partial_exit_leaves_one_contract(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "p.db", session_id="p1"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        contract = _contract()
        buy = OrderIntent(
            candidate_id="b",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=2,
            limit_price=1.0,
        )
        exec_broker.submit_order(buy)
        cid = contract_id_for(contract)
        handler.state.open_trade = OpenTradeContext(
            position_id="p1",
            entry_price=1.0,
            stop_price=0.5,
            take_profit_price=2.0,
            entry_time=handler.provider.current_time,
            time_stop_minutes=30,
            quantity=2,
            reserved_notional_usd=200.0,
        )
        capital.reserve(200.0)
        sell = OrderIntent(
            candidate_id="s",
            contract=contract,
            side="sell",
            order_type="limit",
            quantity=1,
            limit_price=1.2,
        )
        sell_order = exec_broker.submit_order(sell)
        handler.state.pending_exit = PendingExit(
            client_order_id=sell.intent_id,
            broker_order_id=sell_order.order_id,
            position_id="p1",
            contract_id=cid,
            requested_quantity=1,
            reason="take_profit",
            submitted_at=handler.provider.current_time,
        )
        handler._reconcile_pending_exit()
        assert handler.state.open_trade is not None
        assert handler.state.open_trade.quantity == 1
        assert handler.state.trades_exited == 0
        proj = bridge.project_session()
        assert proj.positions[cid].quantity == Decimal("1")
        assert proj.positions[cid].realized_pnl > 0
    finally:
        bridge.shutdown()


def test_completed_exit_uses_ledger_pnl(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "c.db", session_id="c1"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        contract = _contract()
        buy = OrderIntent(
            candidate_id="b",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.0,
        )
        exec_broker.submit_order(buy)
        cid = contract_id_for(contract)
        handler.state.open_trade = OpenTradeContext(
            position_id="p1",
            entry_price=1.0,
            stop_price=0.5,
            take_profit_price=2.0,
            entry_time=handler.provider.current_time,
            time_stop_minutes=30,
            quantity=1,
            reserved_notional_usd=100.0,
        )
        capital.reserve(100.0)
        sell = OrderIntent(
            candidate_id="s",
            contract=contract,
            side="sell",
            order_type="limit",
            quantity=1,
            limit_price=1.5,
        )
        sell_order = exec_broker.submit_order(sell)
        handler.state.pending_exit = PendingExit(
            client_order_id=sell.intent_id,
            broker_order_id=sell_order.order_id,
            position_id="p1",
            contract_id=cid,
            requested_quantity=1,
            reason="take_profit",
            submitted_at=handler.provider.current_time,
        )
        logs: list[str] = []
        handler._on_log = lambda et, p=None: logs.append(et)
        handler._reconcile_pending_exit()
        assert handler.state.open_trade is None
        assert handler.state.pending_exit is None
        assert handler.state.trades_exited == 1
        assert capital.reserved_usd == 0.0
        assert capital.realized_pnl_usd == 50.0
        assert "exit.executed" in logs
        proj = bridge.project_session()
        assert proj.positions[cid].quantity == Decimal("0")
        assert proj.positions[cid].realized_pnl == Decimal("50")
    finally:
        bridge.shutdown()


def test_verified_entry_waits_for_fill_price(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "e.db", session_id="e1"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        setup = _setup()
        contract = _contract()
        candidate = _candidate(contract, handler.provider.current_time)
        # Pending entry with client id not in ledger → stay pending.
        handler.state.pending_entry = PendingEntry(
            order_id="orphan",
            client_order_id="missing-in-ledger",
            candidate=candidate,
            setup=setup,
        )
        handler._reconcile_pending_entry()
        assert handler.state.open_trade is None
        assert handler.state.pending_entry is not None

        buy = OrderIntent(
            candidate_id="b2",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.0,
        )
        order = exec_broker.submit_order(buy)
        handler.state.pending_entry = PendingEntry(
            order_id=order.order_id,
            client_order_id=buy.intent_id,
            candidate=candidate,
            setup=setup,
        )
        handler._reconcile_pending_entry()
        assert handler.state.open_trade is not None
        assert handler.state.open_trade.entry_price == 1.0
        assert handler.state.pending_entry is None
    finally:
        bridge.shutdown()


def test_cancellation_writes_ledger_event(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=50)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "x.db", session_id="x1"
    )
    bridge.start()
    try:
        exec_broker = ExecutionDelegatingBroker(inner=broker, bridge=bridge)
        contract = _contract()
        intent = OrderIntent(
            candidate_id="c",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=0.01,
        )
        order = exec_broker.submit_order(intent)
        assert order.status == "open"
        cancelled = exec_broker.cancel_order(order.order_id)
        assert cancelled.status == "cancelled"
        events = bridge.run_coro(
            bridge.supervisor.ledger_store.get_by_session("x1")  # type: ignore[union-attr]
        )
        assert any(e.event_type.value == "cancellation" for e in events)
    finally:
        bridge.shutdown()


def test_full_cli_session_buy_sell_via_runtimes(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "f.db", session_id="f1"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        bridge.ingest_underlying_quote(
            symbol="SPY",
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            source_timestamp=start,
            received_timestamp=start,
        )
        contract = _contract()
        buy = OrderIntent(
            candidate_id="b",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.0,
        )
        buy_order = exec_broker.submit_order(buy)
        assert buy_order.status == "filled"
        assert handler.state.open_trade is None
        handler.state.pending_entry = PendingEntry(
            order_id=buy_order.order_id,
            client_order_id=buy.intent_id,
            candidate=_candidate(contract, start),
            setup=_setup(),
        )
        handler._reconcile_pending_entry()
        assert handler.state.open_trade is not None
        capital.reserve(handler.state.open_trade.reserved_notional_usd)

        sell = OrderIntent(
            candidate_id="s",
            contract=contract,
            side="sell",
            order_type="limit",
            quantity=1,
            limit_price=1.4,
        )
        sell_order = exec_broker.submit_order(sell)
        handler.state.pending_exit = PendingExit(
            client_order_id=sell.intent_id,
            broker_order_id=sell_order.order_id,
            position_id=handler.state.open_trade.position_id,
            contract_id=contract_id_for(contract),
            requested_quantity=1,
            reason="take_profit",
            submitted_at=start,
        )
        assert handler.state.trades_exited == 0
        handler._reconcile_pending_exit()
        assert handler.state.open_trade is None
        assert handler.state.trades_exited == 1
        proj = bridge.project_session()
        cid = contract_id_for(contract)
        assert proj.positions[cid].quantity == Decimal("0")
        assert proj.positions[cid].realized_pnl == Decimal("40")
    finally:
        bridge.shutdown()


def test_tick_failure_degrades_and_snapshot_recovers(tmp_path: Path) -> None:
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "h.db", session_id="h1"
    )
    bridge.start()
    try:
        market = bridge.market_runtime
        assert market is not None

        async def boom_tick(now=None):
            raise RuntimeError("tick boom")

        market.tick = boom_tick  # type: ignore[method-assign]
        try:
            bridge.tick()
            assert False, "expected tick failure"
        except RuntimeError as exc:
            assert "tick boom" in str(exc)

        assert bridge.health.degraded is True
        assert bridge.health.allows_execution is False

        # Quote success alone must not restore health.
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        bridge.ingest_underlying_quote(
            symbol="SPY",
            last=Decimal("500"),
            bid=Decimal("499.9"),
            ask=Decimal("500.1"),
            source_timestamp=start,
            received_timestamp=start,
        )
        assert bridge.health.degraded is True

        contract = _contract()
        exec_broker = ExecutionDelegatingBroker(inner=broker, bridge=bridge)
        blocked = exec_broker.submit_order(
            OrderIntent(
                candidate_id="blocked",
                contract=contract,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=1.0,
            )
        )
        assert blocked.status == "rejected"
        assert blocked.order_id.startswith("blocked-")

        async def ok_tick(now=None):
            return MarketTickResult(snapshot=object())  # type: ignore[arg-type]

        market.tick = ok_tick  # type: ignore[method-assign]
        bridge.tick()
        assert bridge.health.degraded is False
        assert bridge.health.allows_execution is True

        filled = exec_broker.submit_order(
            OrderIntent(
                candidate_id="ok",
                contract=contract,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=1.0,
            )
        )
        assert filled.status == "filled"
    finally:
        bridge.shutdown()


def test_same_contract_two_round_trips_via_try_exit(tmp_path: Path) -> None:
    """Baseline must be captured before submit inside _try_exit (immediate paper fills)."""
    broker = PaperBroker(slippage_pct=0)
    bridge = CompatibilityLivePaperBridge(
        broker=broker, db_path=tmp_path / "2x.db", session_id="2x"
    )
    bridge.start()
    try:
        handler, capital, exec_broker = _handler(broker, bridge)
        contract = _contract()
        cid = contract_id_for(contract)
        outcomes: list[dict] = []
        handler._on_trade_outcome = lambda payload: outcomes.append(dict(payload))

        def _tp_quote(mid: float) -> OptionQuoteEvent:
            now = handler.provider.current_time
            return OptionQuoteEvent(
                timestamp=now,
                contract_id=cid,
                expiration=contract.expiration,
                strike=contract.strike,
                option_type="call",
                bid=mid - 0.05,
                ask=mid + 0.05,
                mid=mid,
                spread_pct=((0.1) / mid) * 100.0,
                quote_timestamp=now,
            )

        # Trade 1: buy @ 1.0, exit via take-profit at mid 1.5 → +50
        exec_broker.submit_order(
            OrderIntent(
                candidate_id="b1",
                contract=contract,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=1.0,
            )
        )
        handler.state.open_trade = OpenTradeContext(
            position_id="t1",
            entry_price=1.0,
            stop_price=0.5,
            take_profit_price=1.5,
            entry_time=handler.provider.current_time,
            time_stop_minutes=30,
            quantity=1,
            reserved_notional_usd=100.0,
        )
        capital.reserve(100.0)
        decision1 = handler._try_exit(_tp_quote(1.5))
        assert decision1 is not None
        assert handler.state.pending_exit is None
        assert handler.state.open_trade is None
        assert handler.state.trades_exited == 1
        assert outcomes[-1]["trade_pnl_usd"] == 50.0
        assert capital.realized_pnl_usd == 50.0
        assert bridge.project_session().positions[cid].realized_pnl == Decimal("50")

        # Trade 2: same contract, exit at mid 1.2 → +20 (not cumulative 70)
        exec_broker.submit_order(
            OrderIntent(
                candidate_id="b2",
                contract=contract,
                side="buy",
                order_type="limit",
                quantity=1,
                limit_price=1.0,
            )
        )
        handler.state.open_trade = OpenTradeContext(
            position_id="t2",
            entry_price=1.0,
            stop_price=0.5,
            take_profit_price=1.2,
            entry_time=handler.provider.current_time,
            time_stop_minutes=30,
            quantity=1,
            reserved_notional_usd=100.0,
        )
        capital.reserve(100.0)
        decision2 = handler._try_exit(_tp_quote(1.2))
        assert decision2 is not None
        assert handler.state.pending_exit is None
        assert handler.state.open_trade is None
        assert handler.state.trades_exited == 2
        assert outcomes[-1]["trade_pnl_usd"] == 20.0
        assert capital.realized_pnl_usd == 70.0
        assert bridge.project_session().positions[cid].realized_pnl == Decimal("70")
        assert outcomes[-1]["realized_pnl_usd"] == 70.0
    finally:
        bridge.shutdown()

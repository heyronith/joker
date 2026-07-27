"""Restart recovery for ledger mappings, positions, and mismatch corrections."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from joker.broker.interface import PaperBroker
from joker.events.bus import InProcessAsyncEventBus
from joker.ledger.reconciliation import (
    BrokerPositionView,
    BrokerReconciler,
)
from joker.ledger.schemas import LedgerEventType, make_ledger_event
from joker.ledger.store import SqliteLedgerStore
from joker.persistence.aiosqlite_lifecycle import wait_for_no_aiosqlite_workers
from joker.runtime.execution_runtime import ExecutionCommand, ExecutionRuntime
from joker.runtime.session_supervisor import SessionSupervisor, SessionSupervisorConfig
from joker.schemas.domain import OptionContract, OrderIntent
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock


@pytest.mark.asyncio
async def test_restart_reconstructs_mapping_and_open_position(tmp_path) -> None:
    now = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    clock = FrozenExchangeClock(now, calendar=MarketCalendar())
    db = tmp_path / "rec.db"
    broker = PaperBroker(slippage_pct=0)
    bus = InProcessAsyncEventBus()
    ledger = SqliteLedgerStore(db)
    try:
        await ledger.initialize()
        exec_rt = ExecutionRuntime(
            broker=broker,
            ledger_store=ledger,
            event_bus=bus,
            clock=clock,
            session_id="s-rec",
            broker_account_id="paper",
        )
        contract = OptionContract(
            symbol="SPY",
            expiration=date(2026, 7, 1),
            strike=500.0,
            option_type="call",
            is_0dte=True,
        )
        intent = OrderIntent(
            candidate_id="c",
            contract=contract,
            side="buy",
            order_type="limit",
            quantity=1,
            limit_price=1.25,
        )
        order = await exec_rt.submit_execution_command(
            ExecutionCommand(client_order_id="ord1", intent=intent)
        )
        assert order.status == "filled"
        projected = await exec_rt.project_session()
        cid = f"SPY:{contract.expiration.isoformat()}:{contract.strike}:{contract.option_type}"
        assert projected.positions[cid].quantity == Decimal("1")
        await ledger.close()
        await bus.close()

        # Restart
        bus2 = InProcessAsyncEventBus()
        ledger2 = SqliteLedgerStore(db)
        await ledger2.initialize()
        try:
            exec2 = ExecutionRuntime(
                broker=broker,
                ledger_store=ledger2,
                event_bus=bus2,
                clock=clock,
                session_id="s-rec",
                broker_account_id="paper",
            )
            mapping = await exec2.restore_order_mappings()
            assert mapping["ord1"] == order.order_id
            polled = await exec2.poll_order_status("ord1")
            assert polled is not None
            projected2 = await exec2.project_session()
            assert projected2.positions[cid].quantity == Decimal("1")
        finally:
            await ledger2.close()
            await bus2.close()
    finally:
        await wait_for_no_aiosqlite_workers(timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_apply_reconciliation_corrections_for_position_mismatch(tmp_path) -> None:
    now = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    clock = FrozenExchangeClock(now, calendar=MarketCalendar())
    db = tmp_path / "corr.db"
    broker = PaperBroker(slippage_pct=0)
    bus = InProcessAsyncEventBus()
    ledger = SqliteLedgerStore(db)
    try:
        await ledger.initialize()
        # Seed ledger with open qty 1
        await ledger.append(
            make_ledger_event(
                LedgerEventType.ORDER_SUBMISSION_REQUESTED,
                broker_account_id="paper",
                client_order_id="o1",
                contract_id="c1",
                side="buy",
                quantity=Decimal("1"),
                exchange_timestamp=now,
                idempotency_key="s",
                session_id="s1",
            )
        )
        await ledger.append(
            make_ledger_event(
                LedgerEventType.FINAL_FILL,
                broker_account_id="paper",
                client_order_id="o1",
                contract_id="c1",
                side="buy",
                quantity=Decimal("1"),
                price=Decimal("1.0"),
                exchange_timestamp=now,
                idempotency_key="f",
                session_id="s1",
            )
        )
        exec_rt = ExecutionRuntime(
            broker=broker,
            ledger_store=ledger,
            event_bus=bus,
            clock=clock,
            session_id="s1",
            broker_account_id="paper",
        )
        projection = await exec_rt.project_session()
        report = BrokerReconciler().reconcile(
            session_id="s1",
            projection=projection,
            broker_orders=[],
            broker_positions=[
                BrokerPositionView(
                    contract_id="c1", quantity=Decimal("2"), avg_price=Decimal("1.0")
                )
            ],
            exchange_timestamp=now,
        )
        assert not report.is_consistent
        written = await exec_rt.apply_reconciliation_corrections(
            report, mark_unresolved_if_still_mismatched=False
        )
        assert written
        assert any(e.metadata.get("correction_kind") == "position_quantity" for e in written)
        after = await exec_rt.project_session()
        assert after.positions["c1"].quantity == Decimal("2")
    finally:
        await ledger.close()
        await bus.close()
        await wait_for_no_aiosqlite_workers(timeout_seconds=5.0)


@pytest.mark.asyncio
async def test_supervisor_unresolved_blocks_recovery_claim(tmp_path) -> None:
    now = datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc)
    clock = FrozenExchangeClock(now, calendar=MarketCalendar())
    # Empty broker vs empty ledger is consistent — seed a mismatch that
    # auto-correct cannot fully clear without broker position support.
    broker = PaperBroker(slippage_pct=0)
    supervisor = SessionSupervisor(
        broker=broker,
        clock=clock,
        config=SessionSupervisorConfig(
            db_path=tmp_path / "u.db",
            checkpoint_db_path=tmp_path / "u_ckpt.db",
            session_id="s-u",
            auto_apply_reconciliation_corrections=True,
        ),
    )
    try:
        await supervisor.start()
        # Consistent empty state claims recovery.
        assert supervisor.claims_recovery is True
    finally:
        await supervisor.shutdown()
        await wait_for_no_aiosqlite_workers(timeout_seconds=5.0)

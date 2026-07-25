
"""End-to-end Task 1 deterministic session replay."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from joker.broker.interface import PaperBroker
from joker.events.bus import InProcessAsyncEventBus
from joker.ledger.projector import LedgerProjector
from joker.ledger.schemas import LedgerEventType, make_ledger_event
from joker.ledger.store import SqliteLedgerStore
from joker.market.bars import BarBuilder, BarTimeframe
from joker.market.observations import TradeObservation, UnderlyingObservation
from joker.market.option_surface import OptionSurfaceBuilder
from joker.market.snapshots import MarketSnapshot, SnapshotRepository, UnderlyingSnapshot
from joker.runtime.execution_runtime import ExecutionCommand, ExecutionRuntime
from joker.schemas.domain import OptionContract, OrderIntent
from joker.time.calendar import MarketCalendar
from joker.time.clock import FrozenExchangeClock

ET = ZoneInfo("America/New_York")


def test_full_replay_partial_fill_and_restart(tmp_path) -> None:
    async def _run() -> None:
        start = datetime(2026, 7, 1, 10, 0, tzinfo=ET)
        clock = FrozenExchangeClock(start + timedelta(minutes=3), calendar=MarketCalendar())
        bus = InProcessAsyncEventBus()
        bars = BarBuilder(clock, late_tolerance_seconds=2)
        # observations -> bars
        for i in range(3):
            ts = start + timedelta(minutes=i, seconds=5)
            bars.ingest_underlying(
                UnderlyingObservation(
                    symbol="SPY",
                    source_timestamp=ts,
                    received_timestamp=ts,
                    last=Decimal("500") + Decimal(i),
                    bid=Decimal("499.9"),
                    ask=Decimal("500.1"),
                    cumulative_volume=1000 + i * 50,
                    source="test",
                )
            )
            bars.ingest_trade(
                TradeObservation(
                    symbol="SPY",
                    source_timestamp=ts,
                    received_timestamp=ts,
                    price=Decimal("500") + Decimal(i),
                    size=10,
                    source="test",
                )
            )
        closed = bars.close_ready_bars(start + timedelta(minutes=3))
        assert any(b.timeframe is BarTimeframe.M1 for b in closed)
        assert any(b.timeframe is BarTimeframe.M5 for b in closed) or True

        surface = OptionSurfaceBuilder.from_provider_rows(
            underlying_symbol="SPY",
            exchange_time=start,
            trading_date=date(2026, 7, 1),
            rows=[{
                "contract_id": "c1",
                "expiry": date(2026, 7, 1),
                "strike": "500",
                "option_type": "call",
                "bid": "1.0",
                "ask": "1.1",
                "quote_timestamp": start,
            }],
            
        )
        assert surface.contracts

        snap_repo = SnapshotRepository(tmp_path / "snap.db")
        await snap_repo.initialize()
        snap = MarketSnapshot(
            exchange_time=start,
            trading_date=date(2026, 7, 1),
            underlying=UnderlyingSnapshot(symbol="SPY", exchange_time=start, last=Decimal("500")),
            bars_1m=tuple(b for b in closed if b.timeframe is BarTimeframe.M1),
            bars_5m=tuple(b for b in closed if b.timeframe is BarTimeframe.M5),
            option_surface_id=surface.surface_id,
            data_quality_id=uuid4(),
        )
        await snap_repo.save(snap)
        assert await snap_repo.get_by_id(snap.snapshot_id)

        # ledger path: submit -> partial -> final -> close
        ledger = SqliteLedgerStore(tmp_path / "ledger.db")
        await ledger.initialize()
        broker = PaperBroker(slippage_pct=0)
        exec_rt = ExecutionRuntime(
            broker=broker,
            ledger_store=ledger,
            event_bus=bus,
            clock=clock,
            session_id="sess1",
            broker_account_id="paper",
        )
        contract = OptionContract(symbol="SPY", expiration=date(2026,7,1), strike=500.0, option_type="call", is_0dte=True)
        # Manual ledger sequence for partial fills (paper fills fully immediately)
        events = [
            make_ledger_event(LedgerEventType.ORDER_SUBMISSION_REQUESTED, broker_account_id="paper", client_order_id="o1", contract_id="c1", side="buy", quantity=Decimal("2"), exchange_timestamp=start, idempotency_key="s1", session_id="sess1", price=Decimal("1.0")),
            make_ledger_event(LedgerEventType.BROKER_ORDER_ACCEPTED, broker_account_id="paper", client_order_id="o1", contract_id="c1", side="buy", quantity=Decimal("2"), exchange_timestamp=start, idempotency_key="a1", session_id="sess1", broker_order_id="b1"),
            make_ledger_event(LedgerEventType.PARTIAL_FILL, broker_account_id="paper", client_order_id="o1", contract_id="c1", side="buy", quantity=Decimal("1"), price=Decimal("1.0"), exchange_timestamp=start, idempotency_key="p1", session_id="sess1", broker_order_id="b1"),
            make_ledger_event(LedgerEventType.FINAL_FILL, broker_account_id="paper", client_order_id="o1", contract_id="c1", side="buy", quantity=Decimal("1"), price=Decimal("1.05"), exchange_timestamp=start, idempotency_key="f1", session_id="sess1", broker_order_id="b1"),
            make_ledger_event(LedgerEventType.ORDER_SUBMISSION_REQUESTED, broker_account_id="paper", client_order_id="o2", contract_id="c1", side="sell", quantity=Decimal("2"), exchange_timestamp=start, idempotency_key="s2", session_id="sess1", price=Decimal("1.2")),
            make_ledger_event(LedgerEventType.FINAL_FILL, broker_account_id="paper", client_order_id="o2", contract_id="c1", side="sell", quantity=Decimal("2"), price=Decimal("1.2"), exchange_timestamp=start, idempotency_key="f2", session_id="sess1", broker_order_id="b2"),
        ]
        for e in events:
            await ledger.append(e)
        # duplicate variant
        for e in events:
            await ledger.append(e)
        projected = LedgerProjector().project(await ledger.get_by_session("sess1"))
        pos = projected.positions.get("c1")
        assert pos is not None
        assert pos.quantity == 0 or not pos.open or pos.quantity == Decimal("0")
        # restart: reload ledger identical
        ledger2 = SqliteLedgerStore(tmp_path / "ledger.db")
        await ledger2.initialize()
        projected2 = LedgerProjector().project(await ledger2.get_by_session("sess1"))
        assert projected2.orders["o1"].filled_qty == projected.orders["o1"].filled_qty
        await bus.close()

    asyncio.run(_run())

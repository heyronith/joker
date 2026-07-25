
"""Ledger store append + idempotency."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from joker.ledger.schemas import LedgerEventType, make_ledger_event
from joker.ledger.store import SqliteLedgerStore


def test_ledger_append_idempotent(tmp_path) -> None:
    async def _run() -> None:
        store = SqliteLedgerStore(tmp_path / "ledger.db")
        await store.initialize()
        now = datetime.now(timezone.utc)
        evt = make_ledger_event(
            LedgerEventType.ORDER_SUBMISSION_REQUESTED,
            broker_account_id="acct",
            client_order_id="c1",
            contract_id="SPY:2026-07-01:500:call",
            side="buy",
            quantity=Decimal("2"),
            exchange_timestamp=now,
            idempotency_key="k1",
            session_id="s1",
            price=Decimal("1.25"),
        )
        assert await store.append(evt) is True
        assert await store.append(evt) is False
        rows = await store.get_by_session("s1")
        assert len(rows) == 1

    asyncio.run(_run())

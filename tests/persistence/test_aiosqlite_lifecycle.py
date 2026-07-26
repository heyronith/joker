"""Lifecycle tests for aiosqlite worker teardown before event-loop close."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import aiosqlite
import pytest

from joker.graph.checkpoints import SqliteCheckpointStore
from joker.graph.langgraph_checkpointer import CognitiveCheckpointer
from joker.ledger.store import SqliteLedgerStore
from joker.persistence.aiosqlite_lifecycle import (
    close_aiosqlite_connection,
    drain_aiosqlite_workers,
    iter_aiosqlite_worker_threads,
    join_aiosqlite_workers,
)


@pytest.mark.asyncio
async def test_close_aiosqlite_connection_joins_worker(tmp_path: Path) -> None:
    conn = await aiosqlite.connect(tmp_path / "t.db")
    worker = conn._thread
    assert worker.is_alive()
    await close_aiosqlite_connection(conn)
    assert not worker.is_alive()
    assert not iter_aiosqlite_worker_threads()


@pytest.mark.asyncio
async def test_owned_stores_leave_no_workers_after_close(tmp_path: Path) -> None:
    ledger = SqliteLedgerStore(tmp_path / "ledger.db")
    await ledger.initialize()
    checkpoints = SqliteCheckpointStore(tmp_path / "ckpt_store.db")
    await checkpoints.initialize()
    cognitive = CognitiveCheckpointer(tmp_path / "lg.db")
    await cognitive.open()

    assert iter_aiosqlite_worker_threads()

    await cognitive.close()
    await ledger.close()
    await checkpoints.close()
    await drain_aiosqlite_workers()
    assert not iter_aiosqlite_worker_threads()


def test_loop_close_after_drain_does_not_raise_thread_exception(tmp_path: Path) -> None:
    """Regression: closing the loop must not leave aiosqlite workers mid-callback."""
    caught: list[BaseException] = []

    def _hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not None and args.exc_value is not None:
            caught.append(args.exc_value)

    previous = threading.excepthook
    threading.excepthook = _hook
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _body() -> None:
            ledger = SqliteLedgerStore(tmp_path / "loop.db")
            await ledger.initialize()
            await ledger.close()
            checkpointer = CognitiveCheckpointer(tmp_path / "loop_lg.db")
            await checkpointer.open()
            await checkpointer.close()
            await drain_aiosqlite_workers()

        loop.run_until_complete(_body())
        join_aiosqlite_workers()
        loop.close()
        # Give any late worker callbacks a moment to surface.
        join_aiosqlite_workers(timeout=0.5)
    finally:
        threading.excepthook = previous
        try:
            asyncio.set_event_loop(None)
        except Exception:
            pass

    assert not caught, f"unexpected thread exceptions: {caught!r}"

"""Safe aiosqlite connection teardown for event-loop lifecycle boundaries.

aiosqlite 0.22+ runs SQL on a background ``_connection_worker_thread``.  Closing
the owning asyncio loop while that worker still has a Future to complete raises
``RuntimeError: Event loop is closed`` inside the worker and surfaces as
``PytestUnhandledThreadExceptionWarning`` (seen intermittently on CPython 3.12).

Callers that own a long-lived ``aiosqlite.Connection`` must close it through
``close_aiosqlite_connection`` (or the store helpers that use it) and, before
destroying an event loop, call ``drain_aiosqlite_workers``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_SECONDS = 1.0


def _worker_thread(conn: aiosqlite.Connection) -> threading.Thread | None:
    worker = getattr(conn, "_thread", None)
    if isinstance(worker, threading.Thread):
        return worker
    return None


def _is_aiosqlite_worker(thread: threading.Thread) -> bool:
    target = getattr(thread, "_target", None)
    name = thread.name or ""
    if target is not None and getattr(target, "__name__", "") == "_connection_worker_thread":
        return True
    return "connection_worker" in name


def iter_aiosqlite_worker_threads() -> list[threading.Thread]:
    """Return live aiosqlite worker threads (best-effort introspection)."""
    return [
        thread
        for thread in threading.enumerate()
        if thread is not threading.current_thread()
        and thread.is_alive()
        and _is_aiosqlite_worker(thread)
    ]


def join_aiosqlite_workers(*, timeout: float = _JOIN_TIMEOUT_SECONDS) -> None:
    """Block until known aiosqlite workers exit (or overall timeout elapses).

    Only call after every owned connection has been closed; joining a live
    worker that is still serving an open connection simply burns the timeout.
    Uses a single overall deadline so many leaked workers cannot stall CI for
    ``N * timeout`` seconds.
    """
    import time

    deadline = time.monotonic() + max(0.0, timeout)
    for thread in iter_aiosqlite_worker_threads():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            thread.join(remaining)
        except Exception:  # noqa: BLE001 — best-effort teardown
            logger.debug(
                "aiosqlite_worker_join_failed",
                extra={"thread": thread.name},
                exc_info=True,
            )


async def drain_aiosqlite_workers(*, timeout: float = _JOIN_TIMEOUT_SECONDS) -> None:
    """Yield to the loop so close callbacks can finish, then briefly join workers.

    Must not block the running loop inside a long ``thread.join``: aiosqlite
    workers exit only after the loop processes their stop Future. Joining from
    ``asyncio.to_thread`` while the loop waits on that thread deadlocks CI.
    """
    import time

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        await asyncio.sleep(0)
        if not iter_aiosqlite_worker_threads():
            return
        # Short sync joins only — keep yielding so stop callbacks can run.
        join_aiosqlite_workers(timeout=0.05)


async def wait_for_no_aiosqlite_workers(*, timeout_seconds: float = 5.0) -> None:
    """Drain/join until no aiosqlite workers remain, or raise with their names."""
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while True:
        remaining = max(0.0, deadline - asyncio.get_running_loop().time())
        if remaining <= 0:
            break
        await drain_aiosqlite_workers(timeout=min(0.25, remaining))
        if not iter_aiosqlite_worker_threads():
            return
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.02)
    names = [t.name for t in iter_aiosqlite_worker_threads()]
    if names:
        raise AssertionError(f"aiosqlite workers still alive: {names}")


async def close_aiosqlite_connection(
    conn: aiosqlite.Connection | None,
    *,
    join_timeout: float = _JOIN_TIMEOUT_SECONDS,
) -> None:
    """Commit (best-effort), await ``close()``, and join *this* connection's worker.

    Joining is required even after ``await conn.close()``: aiosqlite signals the
    stop Future via ``call_soon_threadsafe`` and only then exits the worker; a
    subsequent ``loop.close()`` can race the thread's final bookkeeping on 3.12.

    Does not wait on sibling connections — call ``drain_aiosqlite_workers`` at
    event-loop teardown boundaries after every owned connection is closed.
    """
    if conn is None:
        return

    worker = _worker_thread(conn)
    try:
        if getattr(conn, "_connection", None) is not None:
            try:
                await conn.commit()
            except Exception:  # noqa: BLE001 — closing anyway
                pass
            await conn.close()
        else:
            # Connection object exists but sqlite handle is gone — still stop worker.
            stop_future = conn.stop()
            if stop_future is not None:
                try:
                    await stop_future
                except Exception:  # noqa: BLE001 — best-effort
                    pass
    except Exception as exc:  # noqa: BLE001 — force-stop without loop callbacks
        logger.warning(
            "aiosqlite_close_failed_forcing_stop",
            extra={"error": str(exc)},
        )
        _force_stop_without_loop_callback(conn)
    finally:
        if worker is not None and worker.is_alive():
            # Sync join is safe here: close() already awaited the stop Future, so
            # the worker is exiting and does not need the event loop.
            worker.join(join_timeout)
        await asyncio.sleep(0)



def _force_stop_without_loop_callback(conn: aiosqlite.Connection) -> None:
    """Enqueue stop with ``future=None`` so the worker never touches a closed loop."""
    try:
        from aiosqlite.core import _STOP_RUNNING_SENTINEL
    except Exception:  # noqa: BLE001
        return

    def close_and_stop() -> Any:
        underlying = getattr(conn, "_connection", None)
        if underlying is not None:
            try:
                underlying.close()
            except Exception:
                pass
            try:
                conn._connection = None
            except Exception:
                pass
        return _STOP_RUNNING_SENTINEL

    try:
        conn._running = False
        conn._tx.put_nowait((None, close_and_stop))
    except Exception:
        logger.debug("aiosqlite_force_stop_enqueue_failed", exc_info=True)

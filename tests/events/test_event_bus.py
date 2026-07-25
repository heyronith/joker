"""Event bus drain waits for slow handlers; shutdown leaves no workers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from joker.events.bus import InProcessAsyncEventBus
from joker.events.schemas import EventType, make_event


def test_event_idempotency_and_handler() -> None:
    async def _run() -> None:
        bus = InProcessAsyncEventBus(handler_timeout_seconds=2.0)
        seen: list[str] = []

        async def handler(event) -> None:
            seen.append(str(event.event_id))

        bus.subscribe(EventType.QUOTE_RECEIVED, handler)
        evt = make_event(
            EventType.QUOTE_RECEIVED,
            session_id="s1",
            exchange_timestamp=datetime.now(timezone.utc),
            source="test",
        )
        assert await bus.publish(evt) is True
        assert await bus.publish(evt) is False
        await bus.drain()
        assert seen == [str(evt.event_id)]
        await bus.close()
        assert bus.active_worker_count == 0
        assert bus.is_idle

    asyncio.run(_run())


def test_drain_waits_for_slow_handler() -> None:
    async def _run() -> None:
        bus = InProcessAsyncEventBus(handler_timeout_seconds=5.0)
        started = asyncio.Event()
        finished = asyncio.Event()

        async def slow_handler(event) -> None:
            started.set()
            await asyncio.sleep(0.2)
            finished.set()

        bus.subscribe(EventType.BAR_CLOSED, slow_handler)
        evt = make_event(
            EventType.BAR_CLOSED,
            session_id="s1",
            exchange_timestamp=datetime.now(timezone.utc),
            source="test",
        )
        await bus.publish(evt)
        await started.wait()
        # Handler still active — drain must not return early.
        assert not finished.is_set()
        await bus.drain(timeout=5.0)
        assert finished.is_set()
        assert bus.is_idle
        await bus.close()
        assert bus.active_worker_count == 0

    asyncio.run(_run())

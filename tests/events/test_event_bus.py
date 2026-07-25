
"""Event bus idempotency and drain."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

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

    asyncio.run(_run())

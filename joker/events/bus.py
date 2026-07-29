"""In-process asynchronous event bus with per-correlation ordering."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from joker.events.schemas import DomainEvent, EventType

logger = logging.getLogger(__name__)

EventHandlerFn = Callable[[DomainEvent], Awaitable[None]]


class InProcessAsyncEventBus:
    """Async event bus: idempotent publish, ordered per correlation_id, isolated handlers.

    No Redis/Kafka — process-local only. Handler failures are logged and isolated.
    """

    def __init__(self, *, handler_timeout_seconds: float | None = 10.0) -> None:
        self._handler_timeout = handler_timeout_seconds
        self._subs: dict[EventType | None, list[EventHandlerFn]] = defaultdict(list)
        self._seen: set[UUID] = set()
        self._queues: dict[UUID, asyncio.Queue[DomainEvent | None]] = {}
        self._workers: dict[UUID, asyncio.Task[Any]] = {}
        self._inflight: set[asyncio.Task[Any]] = set()
        self._dispatching: int = 0
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: EventType | None, handler: EventHandlerFn) -> None:
        """Subscribe to an event type, or None for all events."""
        self._subs[event_type].append(handler)

    async def publish(self, event: DomainEvent) -> bool:
        """Publish event. Returns False if duplicate event_id (idempotent skip)."""
        async with self._lock:
            if event.event_id in self._seen:
                logger.debug(
                    "event_duplicate_skipped",
                    extra={
                        "event_id": str(event.event_id),
                        "event_type": event.event_type.value,
                        "session_id": event.session_id,
                        "correlation_id": str(event.correlation_id),
                    },
                )
                return False
            self._seen.add(event.event_id)
            corr = event.correlation_id
            if corr not in self._queues:
                queue: asyncio.Queue[DomainEvent | None] = asyncio.Queue()
                self._queues[corr] = queue
                worker = asyncio.create_task(
                    self._run_stream(corr, queue),
                    name=f"event-stream-{corr}",
                )
                self._workers[corr] = worker
                self._inflight.add(worker)
                worker.add_done_callback(self._inflight.discard)
            await self._queues[corr].put(event)
            return True

    async def _run_stream(
        self,
        corr: UUID,
        queue: asyncio.Queue[DomainEvent | None],
    ) -> None:
        while True:
            event = await queue.get()
            if event is None:
                break
            self._dispatching += 1
            try:
                await self._dispatch(event)
            finally:
                self._dispatching -= 1

    async def _dispatch(self, event: DomainEvent) -> None:
        handlers = list(self._subs.get(event.event_type, [])) + list(self._subs.get(None, []))
        for handler in handlers:
            name = getattr(handler, "__name__", handler.__class__.__name__)
            try:
                if self._handler_timeout is None:
                    await handler(event)
                else:
                    # Shield is intentionally NOT used: timeouts must cancel the
                    # handler so shutdown can drain. Handlers must therefore be
                    # fast (enqueue) and leave durable work to workers.
                    await asyncio.wait_for(handler(event), timeout=self._handler_timeout)
            except asyncio.TimeoutError:
                logger.error(
                    "event_handler_timeout handler=%s event_type=%s "
                    "timeout_seconds=%s handler_cancelled=True "
                    "work_may_be_incomplete=True",
                    name,
                    event.event_type.value,
                    self._handler_timeout,
                    extra={
                        "handler": name,
                        "timeout_seconds": self._handler_timeout,
                        "event_id": str(event.event_id),
                        "event_type": event.event_type.value,
                        "session_id": event.session_id,
                        "correlation_id": str(event.correlation_id),
                        "handler_cancelled": True,
                        "work_may_be_incomplete": True,
                    },
                )
            except Exception:
                logger.exception(
                    "event_handler_error",
                    extra={
                        "handler": name,
                        "event_id": str(event.event_id),
                        "event_type": event.event_type.value,
                        "session_id": event.session_id,
                        "correlation_id": str(event.correlation_id),
                    },
                )

    async def drain(self, *, timeout: float = 30.0) -> None:
        """Wait until all queues are empty and all dispatched handlers completed."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            queues_empty = all(q.empty() for q in self._queues.values())
            if queues_empty and self._dispatching == 0:
                # Yield once more to catch races where a worker just dequeued.
                await asyncio.sleep(0)
                queues_empty = all(q.empty() for q in self._queues.values())
                if queues_empty and self._dispatching == 0:
                    return
            await asyncio.sleep(0.01)
        raise TimeoutError("event bus drain timed out with pending work")

    async def close(self) -> None:
        """Stop correlation workers cleanly; leaves no worker tasks running."""
        await self.drain(timeout=max(self._handler_timeout or 10.0, 5.0))
        for queue in list(self._queues.values()):
            await queue.put(None)
        for task in list(self._workers.values()):
            await task
        self._queues.clear()
        self._workers.clear()
        self._dispatching = 0

    @property
    def seen_event_ids(self) -> set[UUID]:
        return set(self._seen)

    def clear_seen(self) -> None:
        """Clear idempotency cache (tests / new session only)."""
        self._seen.clear()

    @property
    def active_worker_count(self) -> int:
        return sum(1 for t in self._workers.values() if not t.done())

    @property
    def is_idle(self) -> bool:
        return self._dispatching == 0 and all(q.empty() for q in self._queues.values())

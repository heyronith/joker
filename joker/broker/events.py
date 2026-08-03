"""Broker event ingestion — push when available, order-detail polling fallback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, Protocol

from joker.broker.interface import BrokerClient

BrokerEventKind = Literal[
    "submitted",
    "accepted",
    "partially_filled",
    "filled",
    "cancel_pending",
    "cancelled",
    "rejected",
    "replace_pending",
    "replaced",
    "unknown",
]


@dataclass(frozen=True)
class BrokerEvent:
    event_kind: BrokerEventKind
    client_order_id: str
    broker_order_id: str | None
    captured_at: datetime
    raw: dict[str, Any] | None = None


class BrokerEventSource(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class OrderDetailPollingEventSource:
    """Mandatory fallback — polls get_order for journaled client_order_ids."""

    def __init__(
        self,
        broker: BrokerClient,
        *,
        client_order_ids: Callable[[], list[str]],
        on_event: Callable[[BrokerEvent], Awaitable[None] | None],
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self._broker = broker
        self._client_order_ids = client_order_ids
        self._on_event = on_event
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._last_status: dict[str, str] = {}

    async def start(self) -> None:
        import asyncio

        self._running = True
        try:
            while self._running:
                await self.poll_once()
                if not self._running:
                    break
                try:
                    await asyncio.sleep(self._poll_interval)
                except asyncio.CancelledError:
                    break
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False

    async def poll_once(self) -> list[BrokerEvent]:
        events: list[BrokerEvent] = []
        for cid in self._client_order_ids():
            order = self._broker.get_order(cid)
            if order is None:
                continue
            status = str(order.status)
            prev = self._last_status.get(cid)
            if prev == status:
                continue
            self._last_status[cid] = status
            kind = _map_status(status)
            event = BrokerEvent(
                event_kind=kind,
                client_order_id=cid,
                broker_order_id=order.order_id,
                captured_at=datetime.now(timezone.utc),
                raw={"status": status},
            )
            events.append(event)
            result = self._on_event(event)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        return events


def _map_status(status: str) -> BrokerEventKind:
    value = status.strip().lower()
    mapping: dict[str, BrokerEventKind] = {
        "pending": "submitted",
        "open": "accepted",
        "partially_filled": "partially_filled",
        "filled": "filled",
        "cancelled": "cancelled",
        "rejected": "rejected",
    }
    return mapping.get(value, "unknown")

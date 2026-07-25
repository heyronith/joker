"""Event handler protocol and logging wrapper (no trading decisions)."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from joker.events.schemas import DomainEvent

logger = logging.getLogger(__name__)


@runtime_checkable
class EventHandler(Protocol):
    """Async handler for a single domain event. Must not place orders or decide trades."""

    async def __call__(self, event: DomainEvent) -> None:
        ...


class LoggingEventHandler:
    """Wraps an EventHandler and logs entry/exit with structured context."""

    def __init__(
        self,
        handler: EventHandler,
        *,
        name: str | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self._handler = handler
        self._name = name or getattr(handler, "__name__", handler.__class__.__name__)
        self._log = log or logger

    async def __call__(self, event: DomainEvent) -> None:
        self._log.info(
            "event_handler_start",
            extra={
                "handler": self._name,
                "event_id": str(event.event_id),
                "event_type": event.event_type.value,
                "session_id": event.session_id,
                "correlation_id": str(event.correlation_id),
            },
        )
        try:
            await self._handler(event)
        except Exception:
            self._log.exception(
                "event_handler_failed",
                extra={
                    "handler": self._name,
                    "event_id": str(event.event_id),
                    "event_type": event.event_type.value,
                    "session_id": event.session_id,
                    "correlation_id": str(event.correlation_id),
                },
            )
            raise
        self._log.info(
            "event_handler_complete",
            extra={
                "handler": self._name,
                "event_id": str(event.event_id),
                "event_type": event.event_type.value,
                "session_id": event.session_id,
                "correlation_id": str(event.correlation_id),
            },
        )


async def log_event_handler(event: DomainEvent) -> None:
    """Structured log-only handler (no trading decisions)."""
    logger.info(
        "domain_event",
        extra={
            "event_id": str(event.event_id),
            "event_type": event.event_type.value,
            "session_id": event.session_id,
            "correlation_id": str(event.correlation_id),
            "source": event.source,
        },
    )

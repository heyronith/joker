"""Agent runtime protocol for SessionSupervisor injection."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from joker.events.schemas import DomainEvent


@runtime_checkable
class AgentRuntime(Protocol):
    """Structural protocol for Task 2 agent runtime injection."""

    async def start(self) -> None: ...

    async def on_event(self, event: DomainEvent) -> None: ...

    async def shutdown(self) -> None: ...

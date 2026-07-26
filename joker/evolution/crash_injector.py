"""Crash injection for Task 3 graph recovery tests (never enabled in production)."""

from __future__ import annotations

from typing import Any, Mapping, Protocol


class EvolutionCrashInjector(Protocol):
    async def after_node(self, node_name: str, state: Mapping[str, Any]) -> None:
        ...


class CrashAfterNode:
    """Raise after the named node has persisted its output."""

    def __init__(self, node_name: str, *, message: str = "injected_crash") -> None:
        self.node_name = node_name
        self.message = message
        self.hits = 0

    async def after_node(self, node_name: str, state: Mapping[str, Any]) -> None:
        if node_name == self.node_name:
            self.hits += 1
            raise RuntimeError(f"{self.message}:{node_name}")


class NoopCrashInjector:
    async def after_node(self, node_name: str, state: Mapping[str, Any]) -> None:
        return None

"""Authoritative shared entry-permission gate for paper and live runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntryPermissionState:
    """Shared mutable gate — health, validate_trigger, and gateway all read this."""

    permitted: bool = True
    reasons: tuple[str, ...] = ()

    def block(self, *reasons: str) -> None:
        merged = tuple(dict.fromkeys((*self.reasons, *reasons)))
        self.permitted = False
        self.reasons = merged

    def allow(self) -> None:
        self.permitted = True
        self.reasons = ()

    def as_tuple(self) -> tuple[bool, tuple[str, ...]]:
        return self.permitted, self.reasons

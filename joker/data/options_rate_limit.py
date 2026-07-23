"""Rate limiting and TTL caching for Webull options data."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class RateLimiter:
    """Simple sliding-window rate limiter."""

    max_requests: int = 60
    window_seconds: float = 60.0
    _timestamps: list[float] = field(default_factory=list)

    def acquire(self) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_requests:
            raise RateLimitExceeded(
                f"Rate limit exceeded: {self.max_requests} requests per "
                f"{self.window_seconds:.0f}s"
            )
        self._timestamps.append(now)


class RateLimitExceeded(Exception):
    pass


@dataclass
class TTLCache(Generic[T]):
    ttl_seconds: float
    _entries: dict[str, tuple[float, T]] = field(default_factory=dict)

    def get(self, key: str) -> T | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._entries[key]
            return None
        return value

    def set(self, key: str, value: T) -> None:
        self._entries[key] = (time.monotonic() + self.ttl_seconds, value)

    def clear(self) -> None:
        self._entries.clear()

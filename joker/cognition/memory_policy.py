"""Apply pinned memory policy to lesson/memory selection."""

from __future__ import annotations

from typing import Any, Sequence, TypeVar

from joker.cognition.prompt_overrides import get_active_memory_policy

T = TypeVar("T")


def select_memories(
    memories: Sequence[T],
    *,
    regime: str | None = None,
    regime_attr: str = "regime",
    contradiction_attr: str = "is_contradiction",
) -> list[T]:
    """Filter/limit memories according to the active memory policy."""
    policy = get_active_memory_policy() or {}
    max_memories = int(policy.get("max_memories", len(memories) or 8))
    include_contradictions = bool(policy.get("include_contradictions", True))
    require_regime_match = bool(policy.get("regime_matching", False))

    selected: list[T] = []
    for item in memories:
        if not include_contradictions and bool(getattr(item, contradiction_attr, False)):
            continue
        if require_regime_match and regime is not None:
            item_regime = getattr(item, regime_attr, None)
            if item_regime is not None and item_regime != regime:
                continue
        selected.append(item)
        if len(selected) >= max_memories:
            break
    return selected


def memory_policy_snapshot() -> dict[str, Any]:
    return dict(get_active_memory_policy() or {})

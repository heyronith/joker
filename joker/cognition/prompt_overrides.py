"""Cycle-scoped prompt/profile/policy overrides for pinned Task 3 configurations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from joker.cognition.schemas import AgentRole, PromptSpec

_prompt_overrides: ContextVar[dict[str, PromptSpec] | None] = ContextVar(
    "joker_prompt_overrides", default=None
)
_profile_overrides: ContextVar[dict[str, str] | None] = ContextVar(
    "joker_profile_overrides", default=None
)
_configuration_version_id: ContextVar[str | None] = ContextVar(
    "joker_configuration_version_id", default=None
)
_context_policy: ContextVar[dict[str, Any] | None] = ContextVar(
    "joker_context_policy", default=None
)
_memory_policy: ContextVar[dict[str, Any] | None] = ContextVar(
    "joker_memory_policy", default=None
)
_debate_policy: ContextVar[dict[str, Any] | None] = ContextVar(
    "joker_debate_policy", default=None
)
_routing_policy: ContextVar[dict[str, Any] | None] = ContextVar(
    "joker_routing_policy", default=None
)
_escalation_policy: ContextVar[dict[str, Any] | None] = ContextVar(
    "joker_escalation_policy", default=None
)


def get_override_prompt(role: AgentRole) -> PromptSpec | None:
    overrides = _prompt_overrides.get()
    if not overrides:
        return None
    return overrides.get(role.value)


def get_override_profile(role: str) -> str | None:
    overrides = _profile_overrides.get()
    if not overrides:
        return None
    return overrides.get(role)


def get_active_configuration_version_id() -> str | None:
    return _configuration_version_id.get()


def get_active_context_policy() -> dict[str, Any] | None:
    return _context_policy.get()


def get_active_memory_policy() -> dict[str, Any] | None:
    return _memory_policy.get()


def get_active_debate_policy() -> dict[str, Any] | None:
    return _debate_policy.get()


def get_active_routing_policy() -> dict[str, Any] | None:
    return _routing_policy.get()


def get_active_escalation_policy() -> dict[str, Any] | None:
    return _escalation_policy.get()


@contextmanager
def pinned_configuration_overrides(
    *,
    configuration_version_id: str,
    prompt_overrides: dict[str, PromptSpec],
    role_profiles: dict[str, str],
    context_policy: dict[str, Any] | None = None,
    memory_policy: dict[str, Any] | None = None,
    debate_policy: dict[str, Any] | None = None,
    routing_policy: dict[str, Any] | None = None,
    escalation_policy: dict[str, Any] | None = None,
) -> Iterator[None]:
    """Apply champion/challenger artefacts for the duration of one Task 2 cycle."""
    tokens = (
        _prompt_overrides.set(dict(prompt_overrides)),
        _profile_overrides.set(dict(role_profiles)),
        _configuration_version_id.set(configuration_version_id),
        _context_policy.set(dict(context_policy) if context_policy else None),
        _memory_policy.set(dict(memory_policy) if memory_policy else None),
        _debate_policy.set(dict(debate_policy) if debate_policy else None),
        _routing_policy.set(dict(routing_policy) if routing_policy else None),
        _escalation_policy.set(dict(escalation_policy) if escalation_policy else None),
    )
    try:
        yield
    finally:
        _prompt_overrides.reset(tokens[0])
        _profile_overrides.reset(tokens[1])
        _configuration_version_id.reset(tokens[2])
        _context_policy.reset(tokens[3])
        _memory_policy.reset(tokens[4])
        _debate_policy.reset(tokens[5])
        _routing_policy.reset(tokens[6])
        _escalation_policy.reset(tokens[7])


@contextmanager
def pinned_applied_configuration(applied: Any) -> Iterator[None]:
    """Accept a full AppliedConfiguration and pin all artefacts."""
    with pinned_configuration_overrides(
        configuration_version_id=str(applied.configuration_version_id),
        prompt_overrides=dict(applied.prompt_overrides),
        role_profiles=dict(applied.role_profiles),
        context_policy=getattr(applied, "context_policy", None),
        memory_policy=getattr(applied, "memory_policy", None),
        debate_policy=getattr(applied, "debate_policy", None),
        routing_policy=getattr(applied, "routing_policy", None),
        escalation_policy=getattr(applied, "escalation_policy", None),
    ):
        yield

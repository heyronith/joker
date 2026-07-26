"""Cycle-scoped prompt/profile overrides for pinned Task 3 configurations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

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


@contextmanager
def pinned_configuration_overrides(
    *,
    configuration_version_id: str,
    prompt_overrides: dict[str, PromptSpec],
    role_profiles: dict[str, str],
) -> Iterator[None]:
    """Apply champion/challenger artefacts for the duration of one Task 2 cycle."""
    tok_p = _prompt_overrides.set(dict(prompt_overrides))
    tok_r = _profile_overrides.set(dict(role_profiles))
    tok_c = _configuration_version_id.set(configuration_version_id)
    try:
        yield
    finally:
        _prompt_overrides.reset(tok_p)
        _profile_overrides.reset(tok_r)
        _configuration_version_id.reset(tok_c)

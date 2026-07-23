"""Replay run failure types and fail-closed handling."""

from __future__ import annotations


class ReplayRunError(Exception):
    """Base replay failure — safe for display, no secrets."""


class ReplayLoadFailure(ReplayRunError):
    pass


class OpenAICouncilFailure(ReplayRunError):
    pass


class PlaybookValidationFailure(ReplayRunError):
    pass


class PlaybookArmFailure(ReplayRunError):
    pass


class EmptyReplayFailure(ReplayRunError):
    pass


class NoActivePlaybookFailure(ReplayRunError):
    pass

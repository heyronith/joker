"""Typed exceptions for the cognitive layer."""

from __future__ import annotations


class CognitiveError(Exception):
    """Base error for cognitive graph, context, and persistence failures."""


class ArtifactNotFoundError(CognitiveError):
    """Raised when a cognitive artifact cannot be located by ID."""


class ArtifactPersistenceError(CognitiveError):
    """Raised when append-only artifact persistence fails."""


class ArtifactConflictError(CognitiveError):
    """Raised when an artifact ID or idempotency key already exists with different data."""


class ContextAssemblyError(CognitiveError):
    """Raised when role-specific context cannot be assembled safely."""


class ContextBudgetExceededError(ContextAssemblyError):
    """Raised when context exceeds configured character/token budgets."""


class CognitiveValidationError(CognitiveError):
    """Raised when cognitive schema validation fails outside Pydantic."""


class ModelCallNotFoundError(CognitiveError):
    """Raised when a model call record cannot be located."""


class ModelCallStateError(CognitiveError):
    """Raised when a model call is in an unexpected state for the requested operation."""

"""Typed exceptions for the model provider layer."""

from __future__ import annotations


class ModelError(Exception):
    """Base error for model provider failures. Messages are safe for display."""


class ModelProviderUnavailable(ModelError):
    """The configured provider is offline or not configured."""


class ModelTimeout(ModelError):
    """The model call exceeded its timeout."""


class StructuredOutputFailure(ModelError):
    """The model returned output that failed schema validation."""


class ModelRefusal(ModelError):
    """The model refused to complete the request."""


class ModelConfigurationError(ModelError):
    """Provider or profile configuration is invalid."""


class ModelBudgetExceeded(ModelError):
    """A configured model budget or concurrency limit was exceeded."""


class ModelResponseEmpty(ModelError):
    """The model returned an empty response."""

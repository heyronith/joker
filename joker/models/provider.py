"""Provider-neutral model interface."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from joker.models.schemas import ModelRequest, ModelResult, ProviderHealth

T = TypeVar("T", bound=BaseModel)


class ModelProvider(Protocol):
    """Async structured completion interface for cognitive agents."""

    @property
    def provider_name(self) -> str:
        """Stable provider identifier (e.g. ``ollama``, ``openai``, ``fake``)."""
        ...

    async def healthcheck(self) -> ProviderHealth:
        """Return current provider availability and installed models."""
        ...

    async def complete_structured(
        self,
        *,
        request: ModelRequest,
        output_type: type[T],
    ) -> ModelResult[T]:
        """Complete a request and return schema-validated output."""
        ...

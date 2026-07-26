"""Base cognitive agent — structured model calls with idempotent requests."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Generic, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel

from joker.cognition.artifacts import build_model_call_idempotency_key
from joker.cognition.context import ContextPackage
from joker.cognition.exceptions import CognitiveValidationError
from joker.cognition.prompts import get_prompt
from joker.cognition.schemas import AgentRole, PromptSpec
from joker.models.router import ModelRouter
from joker.models.schemas import ModelRequest

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_OUTPUT_TOKENS = 1200


class CognitiveAgent(ABC, Generic[T]):
    """Role-specific agent that routes a context package through the model layer.

  Tests using :class:`~joker.models.fake_provider.FakeModelProvider` should register
  canned structured outputs per agent role via ``set_canned_for_role``. When no canned
  output is registered the fake provider raises; production providers must return schema-
  valid JSON for the agent's ``output_type``.
    """

    role: ClassVar[AgentRole]
    output_type: ClassVar[type[T]]

    def __init__(self, *, node_name: str | None = None) -> None:
        self._node_name = node_name

    @property
    def node_name(self) -> str:
        return self._node_name or self.role.value

    @property
    def prompt_spec(self) -> PromptSpec:
        return get_prompt(self.role)

    @property
    def prompt_version(self) -> str:
        return self.prompt_spec.version

    async def run(
        self,
        context: ContextPackage,
        router: ModelRouter,
        *,
        attempt_level: int = 0,
        extra_payload: dict[str, Any] | None = None,
    ) -> T:
        """Invoke the model router and return a validated, metadata-enriched artefact."""
        request = self.build_request(
            context,
            attempt_level=attempt_level,
            extra_payload=extra_payload,
        )
        result = await router.route_and_complete(request, self.output_type)
        return self.enrich_output(
            result.output,
            context=context,
            model_call_id=result.request_id,
        )

    def build_request(
        self,
        context: ContextPackage,
        *,
        attempt_level: int = 0,
        extra_payload: dict[str, Any] | None = None,
        request_id: UUID | None = None,
    ) -> ModelRequest:
        """Build an idempotent :class:`ModelRequest` for this agent invocation."""
        payload = context.to_payload()
        if extra_payload:
            payload.update(extra_payload)

        idempotency_key = build_model_call_idempotency_key(
            session_id=context.session_id,
            cycle_id=context.cycle_id,
            snapshot_id=context.snapshot_id,
            node_name=self.node_name,
            agent_role=self.role,
            prompt_version=self.prompt_version,
            context_hash=context.context_hash,
            attempt_level=attempt_level,
        )

        return ModelRequest(
            request_id=request_id or uuid4(),
            idempotency_key=idempotency_key,
            role=self.role.value,
            prompt_id=self.prompt_spec.prompt_id,
            prompt_version=self.prompt_version,
            model_profile="",
            context_payload=payload,
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            snapshot_id=context.snapshot_id,
            cycle_id=context.cycle_id,
        )

    def enrich_output(
        self,
        output: T,
        *,
        context: ContextPackage,
        model_call_id: UUID,
    ) -> T:
        """Attach cycle metadata from context; subclasses may extend."""
        updates: dict[str, Any] = {
            "session_id": context.session_id,
            "snapshot_id": context.snapshot_id,
            "cycle_id": context.cycle_id,
            "prompt_version": self.prompt_version,
            "model_call_id": model_call_id,
        }
        if "agent_role" in output.model_fields:
            updates["agent_role"] = self.role
        return output.model_copy(update=updates)


def require_fields(output: BaseModel, *field_names: str) -> None:
    """Raise when required string fields are empty after enrichment."""
    for name in field_names:
        value = getattr(output, name, None)
        if value is None or (isinstance(value, str) and not value.strip()):
            raise CognitiveValidationError(f"{output.__class__.__name__}.{name} is required")

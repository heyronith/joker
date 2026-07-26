"""Smoke tests for the model provider layer."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from joker.models import (
    FakeModelProvider,
    ModelProviderUnavailable,
    ModelRegistry,
    ModelRequest,
    ModelRouter,
    ModelsConfig,
    StructuredOutputFailure,
)


class SampleOutput(BaseModel):
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


def _request(*, role: str = "market_structure", profile: str = "general_reasoning") -> ModelRequest:
    return ModelRequest(
        request_id=uuid4(),
        idempotency_key=f"idem-{uuid4()}",
        role=role,
        prompt_id="perception.market_structure",
        prompt_version="1.0.0",
        model_profile=profile,
        context_payload={"user_prompt": "analyze"},
        timeout_seconds=5.0,
        max_output_tokens=500,
        snapshot_id=uuid4(),
        cycle_id="cycle-1",
    )


@pytest.mark.asyncio
async def test_fake_provider_returns_canned_role_output() -> None:
    provider = FakeModelProvider()
    provider.set_canned_for_role("market_structure", {"summary": "range bound", "confidence": 0.6})
    request = _request()
    result = await provider.complete_structured(request=request, output_type=SampleOutput)
    assert result.output.summary == "range bound"
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_fake_provider_idempotent_reuse() -> None:
    provider = FakeModelProvider()
    provider.set_canned_for_role("market_structure", {"summary": "stable", "confidence": 0.5})
    request = _request()
    first = await provider.complete_structured(request=request, output_type=SampleOutput)
    second = await provider.complete_structured(request=request, output_type=SampleOutput)
    assert first.output.summary == second.output.summary
    assert second.latency_ms == 0
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_router_escalates_after_schema_failure() -> None:
    local = FakeModelProvider(provider_name="ollama", simulate_malformed=True)
    remote = FakeModelProvider(provider_name="openai")
    remote.set_canned_for_role("falsifier", {"summary": "escalated", "confidence": 0.7})

    config = ModelsConfig(
        ollama={"enabled": True},
        openai={"enabled": True},
    )
    registry = ModelRegistry(config, providers={"ollama": local, "openai": remote})
    router = ModelRouter(registry)

    request = _request(role="falsifier", profile="independent_critic")
    result = await router.route_and_complete(request, SampleOutput)
    assert result.output.summary == "escalated"
    assert result.escalated_from == "independent_critic"
    assert any(log["selected_profile"] == "remote_escalation" for log in router.routing_logs)


@pytest.mark.asyncio
async def test_fake_provider_simulated_failure() -> None:
    provider = FakeModelProvider(simulate_failure=True)
    with pytest.raises(ModelProviderUnavailable):
        await provider.complete_structured(request=_request(), output_type=SampleOutput)


@pytest.mark.asyncio
async def test_fake_provider_malformed_raises_structured_output_failure() -> None:
    provider = FakeModelProvider(simulate_malformed=True)
    provider.set_canned_for_role("market_structure", {"summary": "x", "confidence": 0.5})
    with pytest.raises(StructuredOutputFailure):
        await provider.complete_structured(request=_request(), output_type=SampleOutput)

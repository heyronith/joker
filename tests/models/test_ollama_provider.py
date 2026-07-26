"""Ollama provider tests with mocked HTTP transport."""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel, Field

from joker.models.ollama_provider import OllamaModelProvider
from joker.models.schemas import ModelRequest, OllamaProviderConfig


class SampleOutput(BaseModel):
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


def _request() -> ModelRequest:
    return ModelRequest(
        request_id=uuid4(),
        idempotency_key="idem-1",
        role="market_structure",
        prompt_id="perception.market_structure",
        prompt_version="1.0.0",
        model_profile="qwen3.5:9b",
        context_payload={
            "resolved_model": "qwen3.5:9b",
            "user_prompt": "analyze",
        },
        timeout_seconds=5.0,
        max_output_tokens=500,
        snapshot_id=uuid4(),
        cycle_id="cycle-1",
    )


@pytest.mark.asyncio
async def test_ollama_healthcheck_lists_tags() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"models": [{"name": "qwen3.5:9b"}, {"name": "gemma4:12b"}]},
        )
    )
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaModelProvider(OllamaProviderConfig(base_url="http://ollama.test"), client=client)
    health = await provider.healthcheck()
    assert health.status == "healthy"
    assert "qwen3.5:9b" in health.available_models


@pytest.mark.asyncio
async def test_ollama_complete_structured_parses_json() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": '{"summary": "trend up", "confidence": 0.8}',
                },
                "prompt_eval_count": 12,
                "eval_count": 18,
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaModelProvider(OllamaProviderConfig(base_url="http://ollama.test"), client=client)
    result = await provider.complete_structured(request=_request(), output_type=SampleOutput)
    assert result.output.summary == "trend up"
    assert result.input_tokens == 12
    assert result.output_tokens == 18

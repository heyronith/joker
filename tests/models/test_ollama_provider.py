"""Ollama provider tests with mocked HTTP transport."""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
import pytest
from pydantic import BaseModel, Field

from joker.models.exceptions import ModelResponseEmpty
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
        temperature=0.0,
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
    provider = OllamaModelProvider(
        OllamaProviderConfig(base_url="http://ollama.test"), client=client
    )
    try:
        health = await provider.healthcheck()
        assert health.status == "healthy"
        assert "qwen3.5:9b" in health.available_models
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_complete_structured_sends_think_false_and_schema() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": '{"summary": "trend up", "confidence": 0.8}',
                    "thinking": "ignored",
                },
                "prompt_eval_count": 12,
                "eval_count": 18,
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaModelProvider(
        OllamaProviderConfig(base_url="http://ollama.test"), client=client
    )
    try:
        result = await provider.complete_structured(
            request=_request(), output_type=SampleOutput
        )
        assert result.output.summary == "trend up"
        assert result.input_tokens == 12
        assert result.output_tokens == 18
        body = captured["body"]
        assert isinstance(body, dict)
        assert body["think"] is False
        assert body["stream"] is False
        assert body["options"]["num_predict"] == 500
        assert body["options"]["temperature"] == 0.0
        fmt = body["format"]
        assert isinstance(fmt, dict)
        assert "summary" in (fmt.get("properties") or {})
        assert "confidence" in (fmt.get("properties") or {})
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_empty_content_fails_closed_ignoring_thinking() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3.5:9b"}]})
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "thinking": '{"summary": "from thinking", "confidence": 0.9}',
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="http://ollama.test")
    provider = OllamaModelProvider(
        OllamaProviderConfig(base_url="http://ollama.test"), client=client
    )
    try:
        with pytest.raises(ModelResponseEmpty):
            await provider.complete_structured(
                request=_request(), output_type=SampleOutput
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_ollama_owned_client_closes_cleanly() -> None:
    provider = OllamaModelProvider(
        OllamaProviderConfig(base_url="http://127.0.0.1:9")
    )
    # Force client construction then close without leaking transports.
    _ = provider._get_client()
    assert provider._client is not None
    await provider.aclose()
    assert provider._client is None

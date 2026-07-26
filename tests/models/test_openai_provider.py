"""OpenAI provider tests with mocked SDK client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from joker.models.openai_provider import OpenAIModelProvider
from joker.models.schemas import ModelRequest, OpenAIProviderConfig


class SampleOutput(BaseModel):
    summary: str
    confidence: float = Field(ge=0.0, le=1.0)


def _request() -> ModelRequest:
    return ModelRequest(
        request_id=uuid4(),
        idempotency_key="idem-1",
        role="meta_decision",
        prompt_id="decision.meta",
        prompt_version="1.0.0",
        model_profile="gpt-test",
        context_payload={
            "resolved_model": "gpt-test",
            "user_prompt": "decide",
        },
        timeout_seconds=5.0,
        max_output_tokens=500,
        snapshot_id=uuid4(),
        cycle_id="cycle-1",
    )


@pytest.mark.asyncio
async def test_openai_complete_structured_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-unit-tests-only")

    parsed = SampleOutput(summary="execute", confidence=0.9)
    message = MagicMock(refusal=None, parsed=parsed, content=None)
    choice = MagicMock(message=message)
    usage = MagicMock(prompt_tokens=7, completion_tokens=11, total_tokens=18)
    response = MagicMock(choices=[choice], usage=usage)

    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse = AsyncMock(return_value=response)

    provider = OpenAIModelProvider(OpenAIProviderConfig(), client=mock_client)
    result = await provider.complete_structured(request=_request(), output_type=SampleOutput)
    assert result.output.summary == "execute"
    mock_client.beta.chat.completions.parse.assert_awaited_once()
    call_kwargs = mock_client.beta.chat.completions.parse.await_args.kwargs
    assert call_kwargs["store"] is False

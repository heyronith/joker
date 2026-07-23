"""Phase 15 LLM client tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from joker.agents.llm_client import (
    LLMAPIError,
    LLMTimeoutError,
    LLMValidationError,
    MockLLMClient,
    OpenAILLMClient,
)
from joker.config.validation import redact_secrets
from joker.schemas.domain import AgentOpinion


def _valid_opinion_json() -> str:
    return AgentOpinion(
        agent_name="TestAgent",
        summary="Market is choppy",
        confidence=0.6,
    ).model_dump_json()


def test_agent_opinion_json_schema_strict_metadata() -> None:
    """OpenAI structured outputs require additionalProperties: false on objects."""
    schema = AgentOpinion.model_json_schema()
    metadata_schema = schema["$defs"]["AgentOpinionMetadata"]
    assert metadata_schema.get("additionalProperties") is False
    assert schema.get("additionalProperties") is False


def test_agent_opinion_coerces_legacy_metadata_dict() -> None:
    opinion = AgentOpinion.model_validate(
        {
            "agent_name": "a",
            "summary": "s",
            "confidence": 0.5,
            "metadata": {"notes": "legacy"},
        }
    )
    assert opinion.metadata.notes == "legacy"


def test_mock_llm_parses_valid_structured_response() -> None:
    client = MockLLMClient()
    client.set_response(AgentOpinion, _valid_opinion_json())
    result = client.complete_structured("analyze", AgentOpinion)
    assert result.result.agent_name == "TestAgent"
    assert result.total_tokens == 30


def test_mock_llm_rejects_invalid_structured_response() -> None:
    client = MockLLMClient()
    client.set_response(AgentOpinion, '{"agent_name": "x"}')
    with pytest.raises(LLMValidationError):
        client.complete_structured("analyze", AgentOpinion)


def test_mock_llm_handles_timeout() -> None:
    client = MockLLMClient(delay_seconds=5.0)
    client.set_response(AgentOpinion, _valid_opinion_json())
    with pytest.raises(LLMTimeoutError):
        client.complete_structured("analyze", AgentOpinion, timeout_seconds=1.0)


def test_openai_client_parses_mocked_response() -> None:
    opinion = AgentOpinion(agent_name="MarketRegimeAgent", summary="up", confidence=0.8)
    mock_message = MagicMock()
    mock_message.refusal = None
    mock_message.parsed = opinion
    mock_message.content = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_usage = MagicMock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response

    client = OpenAILLMClient(
        api_key="sk-test-key-for-unit-tests-only",
        model="gpt-5.4-mini",
        client=mock_client,
    )
    result = client.complete_structured("test", AgentOpinion)
    assert result.result.summary == "up"
    assert result.prompt_tokens == 5


def test_openai_client_rejects_invalid_parsed_output() -> None:
    mock_message = MagicMock()
    mock_message.refusal = None
    mock_message.parsed = None
    mock_message.content = '{"agent_name":"x"}'
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response

    client = OpenAILLMClient(
        api_key="sk-test-key-for-unit-tests-only",
        model="gpt-5.4-mini",
        client=mock_client,
    )
    with pytest.raises(LLMValidationError):
        client.complete_structured("test", AgentOpinion)


def test_openai_client_handles_timeout() -> None:
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = TimeoutError("timed out")
    client = OpenAILLMClient(
        api_key="sk-test-key-for-unit-tests-only",
        model="gpt-5.4-mini",
        max_retries=0,
        client=mock_client,
    )
    with pytest.raises(LLMTimeoutError):
        client.complete_structured("test", AgentOpinion, timeout_seconds=1.0)


def test_openai_client_handles_api_error() -> None:
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = RuntimeError("rate limit")
    client = OpenAILLMClient(
        api_key="sk-test-key-for-unit-tests-only",
        model="gpt-5.4-mini",
        max_retries=0,
        client=mock_client,
    )
    with pytest.raises(LLMAPIError):
        client.complete_structured("test", AgentOpinion)


def test_openai_client_redacts_secrets_in_logs() -> None:
    secret = "sk-test-key-for-unit-tests-only"
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.side_effect = RuntimeError(f"auth failed {secret}")
    client = OpenAILLMClient(
        api_key=secret,
        model="gpt-5.4-mini",
        max_retries=0,
        client=mock_client,
    )
    with pytest.raises(LLMAPIError):
        client.complete_structured("test", AgentOpinion)
    assert client.request_log
    logged_error = client.request_log[-1].error or ""
    assert secret not in logged_error
    assert "[REDACTED]" in redact_secrets(logged_error) or secret not in logged_error

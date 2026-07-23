"""Agent council factory and shared utilities."""

from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from joker.agents.communicator import CommunicatorAgent
from joker.agents.llm_client import LLMClient, OpenAILLMClient
from joker.agents.mock_agents import AgentCouncilProtocol, MockAgentCouncil
from joker.agents.openai_agents import OpenAIAgentCouncil
from joker.config.settings import AgentSettings, AppSettings, EnvSettings
from joker.schemas.domain import AgentOpinion

T = TypeVar("T", bound=BaseModel)


class AgentError(Exception):
    pass


class BaseAgent:
    """Shared parsing helper for legacy tests."""

    name: str = "BaseAgent"

    def parse_output(self, raw: str, model: Type[T]) -> T:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgentError(f"{self.name}: invalid JSON output") from exc
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise AgentError(f"{self.name}: schema validation failed: {exc}") from exc


def create_agent_council(
    app_settings: AppSettings,
    env_settings: EnvSettings | None = None,
    llm_client: LLMClient | None = None,
) -> AgentCouncilProtocol:
    """Create mock or OpenAI-backed council based on config."""
    if app_settings.agents.mock_agents:
        return MockAgentCouncil()

    if env_settings is None:
        raise AgentError("env_settings required when mock_agents is false")

    client = llm_client or OpenAILLMClient(
        api_key=env_settings.openai_api_key,
        model=env_settings.openai_model,
        max_retries=app_settings.agents.max_retries,
        default_timeout_seconds=float(app_settings.agents.council_timeout_seconds),
    )
    return OpenAIAgentCouncil(client, app_settings.agents)


def create_communicator(
    app_settings: AppSettings,
    env_settings: EnvSettings | None = None,
    llm_client: LLMClient | None = None,
) -> CommunicatorAgent:
    if app_settings.agents.mock_agents:
        return CommunicatorAgent(llm_client=None)
    if env_settings is None:
        raise AgentError("env_settings required when mock_agents is false")
    client = llm_client or OpenAILLMClient(
        api_key=env_settings.openai_api_key,
        model=env_settings.openai_model,
        max_retries=app_settings.agents.max_retries,
        default_timeout_seconds=float(app_settings.agents.council_timeout_seconds),
    )
    return CommunicatorAgent(llm_client=client)


# Backward-compatible alias
AgentCouncil = MockAgentCouncil

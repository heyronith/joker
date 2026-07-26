"""Cognitive startup provider validation tests."""

from __future__ import annotations

import os

import pytest

from joker.cognition.exceptions import CognitiveRuntimeConfigurationError
from joker.models.fake_provider import FakeModelProvider
from joker.models.registry import ModelRegistry
from joker.models.schemas import ModelsConfig, default_model_profiles
from joker.runtime.cognitive_startup import validate_cognitive_providers


@pytest.mark.asyncio
async def test_cognitive_graph_no_providers_fails_clearly(monkeypatch) -> None:
    monkeypatch.delenv("JOKER_COGNITIVE_USE_FAKE_MODELS", raising=False)
    monkeypatch.delenv("JOKER_COGNITIVE_ALLOW_UNHEALTHY_PROVIDERS", raising=False)
    cfg = ModelsConfig(
        profiles=default_model_profiles(),
    )
    cfg = cfg.model_copy(
        update={
            "ollama": cfg.ollama.model_copy(update={"enabled": False}),
            "openai": cfg.openai.model_copy(update={"enabled": False}),
        }
    )
    with pytest.raises(CognitiveRuntimeConfigurationError, match="cognitive"):
        await validate_cognitive_providers(cfg, mock_agents=False)


@pytest.mark.asyncio
async def test_cognitive_graph_healthy_ollama_mock_transport_starts() -> None:
    class _HealthyOllama:
        provider_name = "ollama"

        async def healthcheck(self):
            from joker.models.schemas import ProviderHealth, utc_now

            return ProviderHealth(
                status="healthy",
                provider_name="ollama",
                available_models=("qwen3.5:9b",),
                checked_at=utc_now(),
            )

    cfg = ModelsConfig(profiles=default_model_profiles())
    cfg = cfg.model_copy(
        update={
            "ollama": cfg.ollama.model_copy(update={"enabled": True}),
            "openai": cfg.openai.model_copy(update={"enabled": False}),
        }
    )
    registry = ModelRegistry.with_defaults(cfg)
    registry.register_provider("ollama", _HealthyOllama())  # type: ignore[arg-type]
    result = await validate_cognitive_providers(
        cfg, mock_agents=False, registry=registry
    )
    assert result.availability.usable
    assert result.availability.ollama_healthy
    assert not result.remapped_to_fake
    for name, profile in result.registry.profiles.items():
        if name in {"fast_structured", "general_reasoning", "independent_critic"}:
            assert profile.provider == "ollama"


@pytest.mark.asyncio
async def test_mock_session_all_profiles_map_explicitly_to_fake() -> None:
    cfg = ModelsConfig(profiles=default_model_profiles())
    result = await validate_cognitive_providers(cfg, mock_agents=True)
    assert result.mock_session
    assert result.remapped_to_fake
    for profile in result.registry.profiles.values():
        assert profile.provider == "fake"
    provider = result.registry.get_provider("fake")
    assert isinstance(provider, FakeModelProvider)


@pytest.mark.asyncio
async def test_fake_env_override_without_mock_agents(monkeypatch) -> None:
    monkeypatch.setenv("JOKER_COGNITIVE_USE_FAKE_MODELS", "1")
    cfg = ModelsConfig(profiles=default_model_profiles())
    cfg = cfg.model_copy(
        update={"ollama": cfg.ollama.model_copy(update={"enabled": False})}
    )
    result = await validate_cognitive_providers(cfg, mock_agents=False)
    assert result.remapped_to_fake
    assert all(p.provider == "fake" for p in result.registry.profiles.values())

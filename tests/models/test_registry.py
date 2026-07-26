"""Model registry default profile tests."""

from __future__ import annotations

from joker.models.registry import ModelRegistry
from joker.models.schemas import ModelsConfig, default_model_profiles


def test_default_profiles_match_task2_yaml() -> None:
    profiles = default_model_profiles()
    assert profiles["fast_structured"].model == "qwen3.5:9b"
    assert profiles["general_reasoning"].model == "gemma4:12b"
    assert profiles["independent_critic"].model == "ministral-3:14b"
    assert profiles["deep_local"].enabled is False
    assert profiles["remote_escalation"].provider == "openai"


def test_registry_resolves_openai_model_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test-model")
    registry = ModelRegistry(ModelsConfig())
    profile = registry.get_profile("remote_escalation")
    assert registry.resolve_model_name(profile) == "gpt-test-model"

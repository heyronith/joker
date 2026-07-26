"""Model profile registry and provider resolution."""

from __future__ import annotations

import os
from typing import Mapping

from joker.models.exceptions import ModelConfigurationError
from joker.models.fake_provider import FakeModelProvider
from joker.models.ollama_provider import OllamaModelProvider
from joker.models.openai_provider import OpenAIModelProvider
from joker.models.provider import ModelProvider
from joker.models.schemas import (
    ModelProfileConfig,
    ModelsConfig,
    OllamaProviderConfig,
    OpenAIProviderConfig,
    default_model_profiles,
)


class ModelRegistry:
    """Registry of named model profiles and provider instances."""

    def __init__(
        self,
        config: ModelsConfig | None = None,
        *,
        providers: Mapping[str, ModelProvider] | None = None,
    ) -> None:
        self._config = config or ModelsConfig()
        self._providers: dict[str, ModelProvider] = dict(providers or {})
        self._ensure_default_providers()

    @property
    def config(self) -> ModelsConfig:
        return self._config

    @property
    def profiles(self) -> dict[str, ModelProfileConfig]:
        return dict(self._config.profiles)

    def register_provider(self, name: str, provider: ModelProvider) -> None:
        """Register or replace a provider instance."""
        self._providers[name] = provider

    def get_provider(self, provider_name: str) -> ModelProvider:
        """Return a provider by name."""
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ModelConfigurationError(f"unknown model provider: {provider_name}")
        return provider

    def get_profile(self, profile_name: str) -> ModelProfileConfig:
        """Return a configured profile."""
        profile = self._config.profiles.get(profile_name)
        if profile is None:
            raise ModelConfigurationError(f"unknown model profile: {profile_name}")
        return profile

    def resolve_model_name(self, profile: ModelProfileConfig) -> str:
        """Resolve environment placeholders in a profile model name."""
        if profile.model == "${OPENAI_MODEL}":
            env_name = self._config.openai.model_env
            value = os.environ.get(env_name, "").strip()
            if not value:
                raise ModelConfigurationError(f"model env {env_name} is not set")
            return value
        return profile.model

    def provider_for_profile(self, profile_name: str) -> tuple[ModelProvider, ModelProfileConfig, str]:
        """Return provider, profile config, and resolved model name."""
        profile = self.get_profile(profile_name)
        if not profile.enabled:
            raise ModelConfigurationError(f"model profile {profile_name!r} is disabled")
        provider = self.get_provider(profile.provider)
        model_name = self.resolve_model_name(profile)
        return provider, profile, model_name

    @classmethod
    def with_defaults(cls, config: ModelsConfig | None = None) -> ModelRegistry:
        """Build a registry with Task 2 default profiles and built-in providers."""
        cfg = config or ModelsConfig()
        if not cfg.profiles:
            cfg = cfg.model_copy(update={"profiles": default_model_profiles()})
        registry = cls(cfg)
        return registry

    def _ensure_default_providers(self) -> None:
        if "fake" not in self._providers:
            self._providers["fake"] = FakeModelProvider()
        if "ollama" not in self._providers and self._config.ollama.enabled:
            self._providers["ollama"] = OllamaModelProvider(self._config.ollama)
        if "openai" not in self._providers and self._config.openai.enabled:
            self._providers["openai"] = OpenAIModelProvider(self._config.openai)

    def update_config(self, config: ModelsConfig) -> None:
        """Replace configuration and refresh default providers when needed."""
        self._config = config
        if self._config.ollama.enabled and "ollama" not in self._providers:
            self._providers["ollama"] = OllamaModelProvider(self._config.ollama)
        if self._config.openai.enabled and "openai" not in self._providers:
            self._providers["openai"] = OpenAIModelProvider(self._config.openai)

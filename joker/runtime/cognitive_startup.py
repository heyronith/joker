"""Cognitive runtime startup validation — fail closed before market polling."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from joker.cognition.exceptions import CognitiveRuntimeConfigurationError
from joker.models.registry import ModelRegistry
from joker.models.schemas import ModelProfileConfig, ModelsConfig, default_model_profiles

logger = logging.getLogger(__name__)

# Profiles required for a usable local cognitive path (enabled by default).
MANDATORY_LOCAL_PROFILES: frozenset[str] = frozenset(
    {
        "fast_structured",
        "general_reasoning",
        "independent_critic",
    }
)

FAKE_OVERRIDE_ENV = "JOKER_COGNITIVE_USE_FAKE_MODELS"
ALLOW_UNHEALTHY_ENV = "JOKER_COGNITIVE_ALLOW_UNHEALTHY_PROVIDERS"


@dataclass(frozen=True)
class ProviderAvailabilityReport:
    """Accurate availability snapshot for mandatory and optional providers."""

    ollama_enabled: bool
    ollama_healthy: bool
    openai_enabled: bool
    openai_healthy: bool
    fake_forced: bool
    mandatory_profiles: tuple[str, ...]
    healthy_mandatory_profiles: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(self.healthy_mandatory_profiles) or self.fake_forced


@dataclass
class CognitiveStartupResult:
    """Validated registry ready for cognitive runtime construction."""

    registry: ModelRegistry
    availability: ProviderAvailabilityReport
    mock_session: bool
    remapped_to_fake: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _mandatory_profiles(config: ModelsConfig) -> dict[str, ModelProfileConfig]:
    profiles = config.profiles or default_model_profiles()
    selected: dict[str, ModelProfileConfig] = {}
    for name in sorted(MANDATORY_LOCAL_PROFILES):
        profile = profiles.get(name)
        if profile is None:
            raise CognitiveRuntimeConfigurationError(
                f"mandatory model profile {name!r} is missing from configuration"
            )
        if not profile.enabled:
            raise CognitiveRuntimeConfigurationError(
                f"mandatory model profile {name!r} is disabled"
            )
        selected[name] = profile
    return selected


async def validate_cognitive_providers(
    models_config: ModelsConfig,
    *,
    mock_agents: bool = False,
    registry: ModelRegistry | None = None,
) -> CognitiveStartupResult:
    """Validate providers before any cognitive market session starts.

    Rules:
    * Mock sessions map every profile explicitly onto the fake provider.
    * Non-mock sessions never route through FakeModelProvider.
    * At least one healthy mandatory local profile is required unless fake is forced.
    * OpenAI escalation may remain optional; availability is reported accurately.
    * Missing usable providers raise ``CognitiveRuntimeConfigurationError``.
    """
    fake_forced = bool(mock_agents) or _env_truthy(FAKE_OVERRIDE_ENV)
    allow_unhealthy = _env_truthy(ALLOW_UNHEALTHY_ENV)

    # Defensive: production load_app_settings() already yields ModelsConfig;
    # retain coercion for direct external callers that still pass a mapping.
    if isinstance(models_config, ModelsConfig):
        cfg = models_config
    else:
        cfg = ModelsConfig.model_validate(models_config)
    if not cfg.profiles:
        cfg = cfg.model_copy(update={"profiles": default_model_profiles()})

    mandatory = _mandatory_profiles(cfg)
    reg = registry or ModelRegistry.with_defaults(cfg)

    if fake_forced:
        from joker.models.fake_provider import FakeModelProvider

        fake = FakeModelProvider(available=True)
        reg.register_provider("fake", fake)
        remapped = {
            name: profile.model_copy(update={"provider": "fake", "model": "fake-model"})
            for name, profile in reg.profiles.items()
        }
        reg.update_config(reg.config.model_copy(update={"profiles": remapped}))
        availability = ProviderAvailabilityReport(
            ollama_enabled=bool(cfg.ollama.enabled),
            ollama_healthy=False,
            openai_enabled=bool(cfg.openai.enabled),
            openai_healthy=False,
            fake_forced=True,
            mandatory_profiles=tuple(sorted(mandatory)),
            healthy_mandatory_profiles=tuple(sorted(mandatory)),
            notes=("all profiles explicitly remapped to fake for mock session",),
        )
        logger.info(
            "cognitive_startup_fake_profiles",
            extra={"profiles": list(remapped), "mock_agents": mock_agents},
        )
        return CognitiveStartupResult(
            registry=reg,
            availability=availability,
            mock_session=True,
            remapped_to_fake=True,
            details={"profile_providers": {n: "fake" for n in remapped}},
        )

    # Non-mock: refuse silent fake routing.
    for name, profile in mandatory.items():
        if profile.provider == "fake":
            raise CognitiveRuntimeConfigurationError(
                f"mandatory profile {name!r} uses FakeModelProvider outside mock "
                f"sessions; enable Ollama or set {FAKE_OVERRIDE_ENV}=1"
            )

    if not cfg.ollama.enabled and not allow_unhealthy:
        raise CognitiveRuntimeConfigurationError(
            "cognitive mode requires Ollama enabled in local-paper configuration "
            f"(models.ollama.enabled=true), or set {FAKE_OVERRIDE_ENV}=1 / "
            f"{ALLOW_UNHEALTHY_ENV}=1 with an explicit override"
        )

    # Ensure providers exist when enabled.
    if cfg.ollama.enabled and "ollama" not in reg._providers:  # noqa: SLF001
        from joker.models.ollama_provider import OllamaModelProvider

        reg.register_provider("ollama", OllamaModelProvider(cfg.ollama))
    if cfg.openai.enabled and "openai" not in reg._providers:  # noqa: SLF001
        from joker.models.openai_provider import OpenAIModelProvider

        reg.register_provider("openai", OpenAIModelProvider(cfg.openai))

    ollama_healthy = False
    openai_healthy = False
    notes: list[str] = []

    if cfg.ollama.enabled:
        try:
            health = await reg.get_provider("ollama").healthcheck()
            ollama_healthy = health.status == "healthy"
            if not ollama_healthy:
                notes.append(f"ollama unhealthy: {health.detail or health.status}")
        except Exception as exc:  # noqa: BLE001 — startup must classify all failures
            notes.append(f"ollama healthcheck failed: {exc}")
    else:
        notes.append("ollama disabled in configuration")

    if cfg.openai.enabled:
        try:
            health = await reg.get_provider("openai").healthcheck()
            openai_healthy = health.status == "healthy"
            if not openai_healthy:
                notes.append(f"openai unhealthy: {health.detail or health.status}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"openai healthcheck failed: {exc}")
    else:
        notes.append("openai escalation optional and disabled")

    healthy_mandatory: list[str] = []
    for name, profile in mandatory.items():
        provider_name = profile.provider
        if provider_name == "ollama" and ollama_healthy:
            healthy_mandatory.append(name)
        elif provider_name == "openai" and openai_healthy:
            healthy_mandatory.append(name)
        elif allow_unhealthy:
            healthy_mandatory.append(name)
            notes.append(f"profile {name} accepted via {ALLOW_UNHEALTHY_ENV}")

    availability = ProviderAvailabilityReport(
        ollama_enabled=bool(cfg.ollama.enabled),
        ollama_healthy=ollama_healthy,
        openai_enabled=bool(cfg.openai.enabled),
        openai_healthy=openai_healthy,
        fake_forced=False,
        mandatory_profiles=tuple(sorted(mandatory)),
        healthy_mandatory_profiles=tuple(sorted(healthy_mandatory)),
        notes=tuple(notes),
    )

    if not availability.usable:
        raise CognitiveRuntimeConfigurationError(
            "cognitive-runtime configuration error: no usable mandatory model "
            f"provider before market session; ollama_enabled={cfg.ollama.enabled} "
            f"ollama_healthy={ollama_healthy} openai_enabled={cfg.openai.enabled} "
            f"openai_healthy={openai_healthy} notes={list(notes)}"
        )

    logger.info(
        "cognitive_startup_validated",
        extra={
            "ollama_healthy": ollama_healthy,
            "openai_healthy": openai_healthy,
            "healthy_mandatory": healthy_mandatory,
            "notes": notes,
        },
    )
    # Healthchecks may have opened AsyncClients on a temporary asyncio.run loop.
    # Close them here so the long-lived paper session recreates clients on its loop.
    await reg.aclose()
    return CognitiveStartupResult(
        registry=reg,
        availability=availability,
        mock_session=False,
        remapped_to_fake=False,
        details={
            "mandatory_profiles": list(mandatory),
            "healthy_mandatory": healthy_mandatory,
            "notes": notes,
        },
    )

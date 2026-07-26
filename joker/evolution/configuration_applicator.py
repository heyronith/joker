"""Apply a pinned CognitiveConfigurationVersion into Task 2 cycle execution."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from joker.cognition.schemas import AgentRole, PromptSpec
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.schemas import CognitiveConfigurationVersion


@dataclass
class AppliedConfiguration:
    configuration_version_id: UUID
    prompt_overrides: dict[str, PromptSpec]
    role_profiles: dict[str, str]


class ConfigurationApplicator:
    """Resolve persisted prompt/policy versions for a pinned configuration."""

    def __init__(self, policy_store: PolicyVersionStore) -> None:
        self._policies = policy_store

    async def apply(
        self, configuration: CognitiveConfigurationVersion
    ) -> AppliedConfiguration:
        ok, problems = await self._policies.verify_configuration_resolvable(configuration)
        if not ok:
            raise RuntimeError(
                "cannot apply unresolvable configuration: " + ", ".join(problems)
            )
        prompts: dict[str, PromptSpec] = {}
        for role_key in configuration.prompt_versions:
            try:
                role = AgentRole(role_key)
            except ValueError:
                continue
            prompts[role_key] = await self._policies.resolve_prompt_spec(
                configuration, role
            )
        return AppliedConfiguration(
            configuration_version_id=configuration.configuration_version_id,
            prompt_overrides=prompts,
            role_profiles=dict(configuration.role_model_profiles),
        )

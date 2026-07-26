"""Apply a pinned CognitiveConfigurationVersion into Task 2 cycle execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from joker.cognition.schemas import AgentRole, PromptSpec
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.schemas import CognitiveConfigurationVersion


@dataclass(frozen=True)
class AppliedConfiguration:
    configuration_version_id: UUID
    prompt_overrides: dict[str, PromptSpec]
    role_profiles: dict[str, str]
    context_policy: dict[str, Any]
    memory_policy: dict[str, Any]
    debate_policy: dict[str, Any]
    routing_policy: dict[str, Any]
    escalation_policy: dict[str, Any]


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
        context = await self._load_policy_content(
            "context_policy_versions", configuration.context_policy_version_id
        )
        memory = await self._load_policy_content(
            "memory_policy_versions", configuration.memory_policy_version_id
        )
        debate = await self._load_policy_content(
            "debate_policy_versions", configuration.debate_policy_version_id
        )
        routing = await self._load_policy_content(
            "routing_policy_versions", configuration.routing_policy_version_id
        )
        escalation = await self._load_policy_content(
            "escalation_policy_versions", configuration.escalation_policy_version_id
        )
        return AppliedConfiguration(
            configuration_version_id=configuration.configuration_version_id,
            prompt_overrides=prompts,
            role_profiles=dict(configuration.role_model_profiles),
            context_policy=context,
            memory_policy=memory,
            debate_policy=debate,
            routing_policy=routing,
            escalation_policy=escalation,
        )

    async def _load_policy_content(
        self, table: str, version_id: UUID
    ) -> dict[str, Any]:
        payload = await self._policies.get_policy(table, version_id)
        if payload is None:
            raise RuntimeError(f"missing policy {table}:{version_id}")
        content = payload.get("content")
        if not isinstance(content, dict):
            raise RuntimeError(f"malformed policy {table}:{version_id}")
        return dict(content)

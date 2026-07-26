"""Typed improvement proposals and materialised challenger configurations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from joker.evolution.hashing import content_hash, hash_model, stable_json_dumps
from joker.evolution.idempotency import proposal_idempotency_key
from joker.evolution.policy_store import PolicyVersionStore
from joker.evolution.repositories import (
    ConfigurationVersionRepository,
    ImprovementProposalRepository,
)
from joker.evolution.schemas import (
    PROHIBITED_MUTATION_TARGETS,
    CognitiveConfigurationVersion,
    CognitivePatch,
    ContextPolicyPatch,
    ContextPolicyVersion,
    DebatePolicyPatch,
    DebatePolicyVersion,
    EscalationPolicyPatch,
    EscalationPolicyVersion,
    ImprovementProposal,
    MemoryPolicyPatch,
    MemoryPolicyVersion,
    PromptPatch,
    RoutingPolicyPatch,
    RoutingPolicyVersion,
    assert_no_chain_of_thought,
)


class ImprovementError(ValueError):
    pass


def parse_cognitive_patch(payload: dict[str, Any]) -> CognitivePatch:
    assert_no_chain_of_thought(payload)
    patch_type = payload.get("patch_type")
    mapping = {
        "prompt": PromptPatch,
        "context_policy": ContextPolicyPatch,
        "routing_policy": RoutingPolicyPatch,
        "debate_policy": DebatePolicyPatch,
        "memory_policy": MemoryPolicyPatch,
        "escalation_policy": EscalationPolicyPatch,
    }
    if patch_type not in mapping:
        target = str(payload.get("mutation_target", "")).lower()
        if target in PROHIBITED_MUTATION_TARGETS or patch_type in PROHIBITED_MUTATION_TARGETS:
            raise ImprovementError("prohibited mutation target")
        raise ImprovementError(f"unsupported patch_type: {patch_type}")
    return mapping[patch_type].model_validate(payload)  # type: ignore[return-value]


class ImprovementProposalService:
    def __init__(
        self,
        proposal_repo: ImprovementProposalRepository,
        config_repo: ConfigurationVersionRepository,
        policy_store: PolicyVersionStore,
    ) -> None:
        self._proposals = proposal_repo
        self._configs = config_repo
        self._policies = policy_store

    async def propose(
        self,
        *,
        parent_champion: CognitiveConfigurationVersion,
        weakness: str,
        hypothesis: str,
        patch: CognitivePatch | dict[str, Any],
        supporting_episode_ids: tuple[UUID, ...] = (),
        supporting_evaluation_ids: tuple[UUID, ...] = (),
        metrics_to_improve: tuple[str, ...] = ("calibration_score",),
        metrics_must_not_regress: tuple[str, ...] = ("tail_loss", "safety_violations"),
        evaluation_window_hash: str = "window",
    ) -> tuple[ImprovementProposal, CognitiveConfigurationVersion]:
        if isinstance(patch, dict):
            target = str(patch.get("mutation_target", "")).lower()
            if target in PROHIBITED_MUTATION_TARGETS:
                raise ImprovementError(f"prohibited mutation target: {target}")
            parsed = parse_cognitive_patch(patch)
            change = parsed.model_dump(mode="json")
        else:
            parsed = patch
            change = patch.model_dump(mode="json")

        proposal_hash = content_hash(stable_json_dumps(change), weakness, hypothesis)
        key = proposal_idempotency_key(
            evaluation_window_hash,
            parent_champion.configuration_version_id,
            proposal_hash,
        )
        proposal = ImprovementProposal(
            proposal_id=uuid4(),
            parent_champion_version_id=parent_champion.configuration_version_id,
            weakness=weakness,
            supporting_episode_ids=supporting_episode_ids,
            supporting_evaluation_ids=supporting_evaluation_ids,
            hypothesis=hypothesis,
            proposed_change=change,
            expected_benefit="improve declared cognitive weakness",
            expected_risks=("distribution_shift", "cost_increase"),
            metrics_to_improve=metrics_to_improve,
            metrics_must_not_regress=metrics_must_not_regress,
            required_evaluation_slices=("holdout", "adversarial"),
            content_hash=proposal_hash,
            status="registered",
            idempotency_key=key,
        )
        await self._proposals.append(proposal)
        challenger = await self.compile_challenger(parent_champion, parsed)
        await self._configs.append(challenger)
        ok, problems = await self._policies.verify_configuration_resolvable(challenger)
        if not ok:
            raise ImprovementError(
                "challenger references unresolved artefacts: " + ", ".join(problems)
            )
        return proposal, challenger

    async def compile_challenger(
        self,
        parent: CognitiveConfigurationVersion,
        patch: CognitivePatch,
    ) -> CognitiveConfigurationVersion:
        profiles = dict(parent.role_model_profiles)
        prompts = dict(parent.prompt_versions)
        context_id = parent.context_policy_version_id
        memory_id = parent.memory_policy_version_id
        debate_id = parent.debate_policy_version_id
        routing_id = parent.routing_policy_version_id
        escalation_id = parent.escalation_policy_version_id
        now = datetime.now(timezone.utc)

        if isinstance(patch, PromptPatch):
            parent_prompt = patch.parent_prompt_version_id
            if patch.role in prompts:
                parent_prompt = prompts[patch.role]
            record = await self._policies.materialise_prompt_override(
                role=patch.role,
                template=patch.replacement_template,
                parent_prompt_version_id=parent_prompt,
                change_rationale=patch.change_rationale,
            )
            prompts[patch.role] = record.prompt_version_id
        elif isinstance(patch, RoutingPolicyPatch):
            if patch.preferred_profile:
                profiles[patch.role] = patch.preferred_profile
            content = {
                "preferred_profile": patch.preferred_profile,
                "escalation_profile": patch.escalation_profile,
                "escalation_conditions": list(patch.escalation_conditions),
            }
            policy = RoutingPolicyVersion(
                content=content,
                content_hash=content_hash(stable_json_dumps(content)),
                created_at=now,
            )
            await self._policies.append_policy(
                "routing_policy_versions",
                policy.version_id,
                policy.content_hash,
                policy.model_dump_json(),
                now,
            )
            routing_id = policy.version_id
        elif isinstance(patch, ContextPolicyPatch):
            content = {
                "role": patch.role,
                "token_budget_delta": patch.token_budget_delta,
                "evidence_priority_changes": [
                    c.model_dump(mode="json") for c in patch.evidence_priority_changes
                ],
                "recency_policy": patch.recency_policy,
                "preserve_data_quality": True,
                "preserve_positions": True,
                "preserve_working_orders": True,
            }
            policy = ContextPolicyVersion(
                content=content,
                content_hash=content_hash(stable_json_dumps(content)),
                created_at=now,
            )
            await self._policies.append_policy(
                "context_policy_versions",
                policy.version_id,
                policy.content_hash,
                policy.model_dump_json(),
                now,
            )
            context_id = policy.version_id
        elif isinstance(patch, DebatePolicyPatch):
            content = patch.model_dump(mode="json")
            policy = DebatePolicyVersion(
                content=content,
                content_hash=content_hash(stable_json_dumps(content)),
                created_at=now,
            )
            await self._policies.append_policy(
                "debate_policy_versions",
                policy.version_id,
                policy.content_hash,
                policy.model_dump_json(),
                now,
            )
            debate_id = policy.version_id
        elif isinstance(patch, MemoryPolicyPatch):
            content = patch.model_dump(mode="json")
            policy = MemoryPolicyVersion(
                content=content,
                content_hash=content_hash(stable_json_dumps(content)),
                created_at=now,
            )
            await self._policies.append_policy(
                "memory_policy_versions",
                policy.version_id,
                policy.content_hash,
                policy.model_dump_json(),
                now,
            )
            memory_id = policy.version_id
        elif isinstance(patch, EscalationPolicyPatch):
            content = patch.model_dump(mode="json")
            policy = EscalationPolicyVersion(
                content=content,
                content_hash=content_hash(stable_json_dumps(content)),
                created_at=now,
            )
            await self._policies.append_policy(
                "escalation_policy_versions",
                policy.version_id,
                policy.content_hash,
                policy.model_dump_json(),
                now,
            )
            escalation_id = policy.version_id
        else:
            raise ImprovementError(f"unsupported patch: {type(patch)}")

        challenger = CognitiveConfigurationVersion(
            configuration_version_id=uuid4(),
            parent_version_id=parent.configuration_version_id,
            status="challenger",
            prompt_versions=prompts,
            role_model_profiles=profiles,
            context_policy_version_id=context_id,
            memory_policy_version_id=memory_id,
            debate_policy_version_id=debate_id,
            routing_policy_version_id=routing_id,
            escalation_policy_version_id=escalation_id,
            content_hash="",
            created_by="agent",
            created_at=now,
            scope_key=parent.scope_key,
        )
        return challenger.model_copy(
            update={
                "content_hash": hash_model(
                    challenger, exclude={"created_at", "status", "configuration_version_id"}
                )
            }
        )

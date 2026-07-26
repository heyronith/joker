"""Typed improvement proposals and challenger configuration compilation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from joker.evolution.hashing import content_hash, hash_model, stable_json_dumps
from joker.evolution.idempotency import proposal_idempotency_key
from joker.evolution.repositories import (
    ConfigurationVersionRepository,
    ImprovementProposalRepository,
)
from joker.evolution.schemas import (
    PROHIBITED_MUTATION_TARGETS,
    CognitiveConfigurationVersion,
    CognitivePatch,
    ContextPolicyPatch,
    DebatePolicyPatch,
    ImprovementProposal,
    PromptPatch,
    RoutingPolicyPatch,
    assert_no_chain_of_thought,
)


class ImprovementError(ValueError):
    pass


def parse_cognitive_patch(payload: dict[str, Any]) -> CognitivePatch:
    assert_no_chain_of_thought(payload)
    patch_type = payload.get("patch_type")
    if patch_type == "prompt":
        return PromptPatch.model_validate(payload)
    if patch_type == "context_policy":
        return ContextPolicyPatch.model_validate(payload)
    if patch_type == "routing_policy":
        return RoutingPolicyPatch.model_validate(payload)
    if patch_type == "debate_policy":
        return DebatePolicyPatch.model_validate(payload)
    if patch_type in PROHIBITED_MUTATION_TARGETS or payload.get("mutation_target") in (
        PROHIBITED_MUTATION_TARGETS
    ):
        raise ImprovementError("prohibited mutation target")
    raise ImprovementError(f"unsupported patch_type: {patch_type}")


class ImprovementProposalService:
    def __init__(
        self,
        proposal_repo: ImprovementProposalRepository,
        config_repo: ConfigurationVersionRepository,
    ) -> None:
        self._proposals = proposal_repo
        self._configs = config_repo

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
        challenger = self.compile_challenger(parent_champion, change)
        await self._configs.append(challenger)
        return proposal, challenger

    def compile_challenger(
        self,
        parent: CognitiveConfigurationVersion,
        change: dict[str, Any],
    ) -> CognitiveConfigurationVersion:
        assert_no_chain_of_thought(change)
        profiles = dict(parent.role_model_profiles)
        prompts = dict(parent.prompt_versions)
        patch_type = change.get("patch_type")
        if patch_type == "routing_policy" and change.get("preferred_profile"):
            profiles[change["role"]] = change["preferred_profile"]
        if patch_type == "prompt":
            # Prompt templates are versioned by new UUID placeholder; template stored in change.
            prompts[change["role"]] = uuid4()

        challenger = CognitiveConfigurationVersion(
            configuration_version_id=uuid4(),
            parent_version_id=parent.configuration_version_id,
            status="challenger",
            prompt_versions=prompts,
            role_model_profiles=profiles,
            context_policy_version_id=parent.context_policy_version_id,
            memory_policy_version_id=parent.memory_policy_version_id,
            debate_policy_version_id=parent.debate_policy_version_id,
            routing_policy_version_id=parent.routing_policy_version_id,
            escalation_policy_version_id=parent.escalation_policy_version_id,
            content_hash="",
            created_by="agent",
            created_at=datetime.now(timezone.utc),
            scope_key=parent.scope_key,
        )
        return challenger.model_copy(
            update={
                "content_hash": hash_model(
                    challenger, exclude={"created_at", "status", "configuration_version_id"}
                )
            }
        )

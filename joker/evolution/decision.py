"""Evolution decision graph — agent strategic choice bounded by deterministic gates."""

from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from joker.evolution.champion_registry import ChampionRegistry
from joker.evolution.idempotency import promotion_idempotency_key
from joker.evolution.promotion_gate import EligibilityResult, PromotionEligibilityGate
from joker.evolution.repositories import (
    ConfigurationVersionRepository,
    PromotionDecisionRepository,
)
from joker.evolution.schemas import (
    CognitiveConfigurationVersion,
    ExperimentResult,
    ImprovementProposal,
    PromotionDecision,
)


AgentAction = Literal["promote", "reject", "extend_shadow", "request_more_evidence"]


class EvolutionDecisionAgent:
    """Strategic promotion judgement — cannot override failed deterministic gates."""

    def decide(
        self,
        *,
        eligibility: EligibilityResult,
        result: ExperimentResult,
        proposal: ImprovementProposal | None = None,
    ) -> tuple[AgentAction, str, tuple[str, ...]]:
        if not eligibility.eligible:
            return (
                "reject",
                "deterministic eligibility failed; agent cannot promote",
                tuple(eligibility.gate_codes),
            )
        # Prefer promote when challenger mean pnl improves and no unresolved hard risks.
        delta = result.aggregate_metrics.get("pnl_delta")
        if delta is not None and float(delta) < 0:
            return (
                "request_more_evidence",
                "eligible but pnl_delta negative; gather more evidence",
                ("negative_pnl_delta",),
            )
        if proposal and "calibration_score" in proposal.metrics_to_improve:
            chall_cal = result.challenger_metrics.get("calibration_error")
            champ_cal = result.champion_metrics.get("calibration_error")
            if (
                chall_cal is not None
                and champ_cal is not None
                and float(chall_cal) > float(champ_cal)
            ):
                return (
                    "reject",
                    "rejects despite eligibility: calibration regressed vs declared objective",
                    ("calibration_objective_miss",),
                )
        return (
            "promote",
            "eligible challenger improves declared objectives with acceptable tradeoffs",
            (),
        )


class EvolutionDecisionService:
    def __init__(
        self,
        promotion_repo: PromotionDecisionRepository,
        config_repo: ConfigurationVersionRepository,
        champion_registry: ChampionRegistry,
        *,
        gate: PromotionEligibilityGate | None = None,
        agent: EvolutionDecisionAgent | None = None,
    ) -> None:
        self._promotions = promotion_repo
        self._configs = config_repo
        self._champions = champion_registry
        self._gate = gate or PromotionEligibilityGate()
        self._agent = agent or EvolutionDecisionAgent()

    async def decide_and_apply(
        self,
        *,
        experiment_id: UUID,
        result: ExperimentResult,
        challenger: CognitiveConfigurationVersion,
        champion: CognitiveConfigurationVersion,
        proposal: ImprovementProposal | None = None,
        holdout_episode_count: int = 0,
        completed_episode_count: int = 0,
        adversarial_passed: bool = True,
        agent_override_action: AgentAction | None = None,
    ) -> PromotionDecision:
        eligibility = self._gate.evaluate(
            result=result,
            proposal=proposal,
            holdout_episode_count=holdout_episode_count,
            completed_episode_count=completed_episode_count,
            adversarial_passed=adversarial_passed,
        )
        if agent_override_action is not None:
            action = agent_override_action
            rationale = "explicit agent/test override"
            risks: tuple[str, ...] = ()
            if not eligibility.eligible and action == "promote":
                action = "reject"
                rationale = "override blocked by deterministic gate"
                risks = eligibility.gate_codes
        else:
            action, rationale, risks = self._agent.decide(
                eligibility=eligibility, result=result, proposal=proposal
            )

        if not eligibility.eligible:
            final_status: Literal[
                "promoted", "rejected", "pending_evidence", "blocked_by_gate"
            ] = "blocked_by_gate"
            if agent_override_action == "promote":
                final_status = "blocked_by_gate"
            elif action == "request_more_evidence":
                final_status = "pending_evidence"
            elif action == "reject" and agent_override_action is None:
                final_status = "rejected"
            else:
                final_status = "blocked_by_gate"
        elif action == "promote":
            final_status = "promoted"
        elif action == "request_more_evidence" or action == "extend_shadow":
            final_status = "pending_evidence"
        else:
            final_status = "rejected"

        key = promotion_idempotency_key(
            experiment_id,
            challenger.configuration_version_id,
            champion.configuration_version_id,
        )
        decision = PromotionDecision(
            promotion_decision_id=uuid4(),
            experiment_id=experiment_id,
            challenger_version_id=challenger.configuration_version_id,
            champion_version_id=champion.configuration_version_id,
            deterministic_eligible=eligibility.eligible,
            deterministic_gate_codes=eligibility.gate_codes,
            agent_action=action,
            strategic_rationale=rationale,
            accepted_tradeoffs=(),
            unresolved_risks=risks,
            final_status=final_status,
            idempotency_key=key,
        )
        inserted = await self._promotions.append(decision)
        if not inserted:
            existing = await self._promotions.get_by_experiment(experiment_id)
            if existing is not None:
                return existing

        if decision.final_status == "promoted":
            await self._champions.promote(
                challenger=challenger,
                expected_champion_id=champion.configuration_version_id,
                reason="agent_promote",
                experiment_id=experiment_id,
                promotion_decision_id=decision.promotion_decision_id,
            )
            await self._configs.mark_status(
                challenger.configuration_version_id, "champion"
            )
        elif decision.final_status == "rejected":
            await self._configs.mark_status(
                challenger.configuration_version_id, "rejected"
            )
        elif decision.final_status == "pending_evidence":
            await self._configs.mark_status(
                challenger.configuration_version_id, "shadow"
            )
        return decision

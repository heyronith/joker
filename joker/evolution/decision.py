"""Evolution decision graph — agent strategic choice bounded by deterministic gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypedDict
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from joker.evolution.agent_invoke import invoke_evolution_agent
from joker.evolution.agent_schemas import EvolutionDecisionAgentOutput
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
from joker.models.router import ModelRouter


AgentAction = Literal["promote", "reject", "extend_shadow", "request_more_evidence"]


class EvolutionDecisionAgent:
    """Strategic promotion judgement via ModelRouter — cannot override failed gates."""

    def __init__(self, router: ModelRouter | None = None) -> None:
        self._router = router

    async def decide_async(
        self,
        *,
        eligibility: EligibilityResult,
        result: ExperimentResult,
        proposal: ImprovementProposal | None = None,
        session_id: str = "evolution",
        snapshot_id: UUID | None = None,
    ) -> tuple[AgentAction, str, tuple[str, ...]]:
        if not eligibility.eligible:
            return (
                "reject",
                "deterministic eligibility failed; agent cannot promote",
                tuple(eligibility.gate_codes),
            )
        if self._router is None:
            return self.decide(eligibility=eligibility, result=result, proposal=proposal)

        payload = {
            "eligible": eligibility.eligible,
            "gate_codes": list(eligibility.gate_codes),
            "aggregate_metrics": {
                k: str(v) for k, v in result.aggregate_metrics.items()
            },
            "champion_metrics": {k: str(v) for k, v in result.champion_metrics.items()},
            "challenger_metrics": {
                k: str(v) for k, v in result.challenger_metrics.items()
            },
            "proposal_metrics_to_improve": (
                list(proposal.metrics_to_improve) if proposal else []
            ),
            "allowed_actions": [
                "promote",
                "reject",
                "extend_shadow",
                "request_more_evidence",
            ],
        }
        output, _call_id = await invoke_evolution_agent(
            self._router,
            role="evolution_decision",
            prompt_id="task3.evolution_decision",
            prompt_version="3.0.0",
            payload=payload,
            output_type=EvolutionDecisionAgentOutput,
            snapshot_id=snapshot_id or uuid4(),
            cycle_id=f"promotion:{result.experiment_id}",
            session_id=session_id,
        )
        return output.action, output.summary, output.rationale_codes

    def decide(
        self,
        *,
        eligibility: EligibilityResult,
        result: ExperimentResult,
        proposal: ImprovementProposal | None = None,
    ) -> tuple[AgentAction, str, tuple[str, ...]]:
        """Deterministic fallback used only when ModelRouter is unavailable."""
        if not eligibility.eligible:
            return (
                "reject",
                "deterministic eligibility failed; agent cannot promote",
                tuple(eligibility.gate_codes),
            )
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


class DecisionGraphState(TypedDict, total=False):
    eligibility: EligibilityResult
    result: ExperimentResult
    proposal: ImprovementProposal | None
    session_id: str
    action: AgentAction
    rationale: str
    risks: tuple[str, ...]
    decision_persisted: bool
    decision: PromotionDecision
    activated: bool
    decision_persisted: bool
    decision: PromotionDecision
    activated: bool


def build_evolution_decision_graph(agent: EvolutionDecisionAgent):
    async def decide_node(state: DecisionGraphState) -> dict[str, Any]:
        action, rationale, risks = await agent.decide_async(
            eligibility=state["eligibility"],
            result=state["result"],
            proposal=state.get("proposal"),
            session_id=state.get("session_id") or "evolution",
        )
        return {"action": action, "rationale": rationale, "risks": risks}

    async def persist_decision_node(state: DecisionGraphState) -> dict[str, Any]:
        # Marker node: decision fields already in state; activation is separate.
        return {"decision_persisted": True}

    graph = StateGraph(DecisionGraphState)
    graph.add_node("decide", decide_node)
    graph.add_node("persist_decision", persist_decision_node)
    graph.add_edge(START, "decide")
    graph.add_edge("decide", "persist_decision")
    graph.add_edge("persist_decision", END)
    return graph


class EvolutionDecisionService:
    def __init__(
        self,
        promotion_repo: PromotionDecisionRepository,
        config_repo: ConfigurationVersionRepository,
        champion_registry: ChampionRegistry,
        *,
        gate: PromotionEligibilityGate | None = None,
        agent: EvolutionDecisionAgent | None = None,
        router: ModelRouter | None = None,
        checkpointer_path: Path | None = None,
        checkpointer_saver: AsyncSqliteSaver | None = None,
        session_id: str = "evolution",
    ) -> None:
        self._promotions = promotion_repo
        self._configs = config_repo
        self._champions = champion_registry
        self._gate = gate or PromotionEligibilityGate()
        self._agent = agent or EvolutionDecisionAgent(router)
        self._checkpointer_path = checkpointer_path
        self._checkpointer_saver = checkpointer_saver
        self._session_id = session_id
        self._compiled = None

    def _graph(self):
        if self._compiled is not None:
            return self._compiled
        builder = build_evolution_decision_graph(self._agent)
        if self._checkpointer_saver is not None:
            self._compiled = builder.compile(checkpointer=self._checkpointer_saver)
        else:
            self._compiled = builder.compile()
        return self._compiled

    async def decide(
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
        """Persist strategic decision only — does not mutate champion registry."""
        existing = await self._promotions.get_by_experiment(experiment_id)
        if existing is not None:
            return existing
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
            compiled = self._graph()
            state: DecisionGraphState = {
                "eligibility": eligibility,
                "result": result,
                "proposal": proposal,
                "session_id": self._session_id,
            }
            if self._checkpointer_saver is not None:
                thread_id = (
                    f"{experiment_id}:{challenger.configuration_version_id}:"
                    f"{champion.configuration_version_id}"
                )
                decided = await compiled.ainvoke(
                    state, config={"configurable": {"thread_id": thread_id}}
                )
            else:
                decided = await compiled.ainvoke(state)
            action = decided["action"]
            rationale = decided["rationale"]
            risks = decided["risks"]

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
        return decision

    async def apply_persisted_decision(
        self, *, promotion_decision_id: UUID
    ) -> PromotionDecision:
        """Idempotent champion CAS activation for a persisted promote decision."""
        decision = await self._promotions.get_by_id(promotion_decision_id)
        if decision is None:
            raise RuntimeError(f"promotion_decision_not_found:{promotion_decision_id}")
        if decision.final_status != "promoted":
            if decision.final_status == "rejected":
                await self._configs.mark_status(
                    decision.challenger_version_id, "rejected"
                )
            elif decision.final_status == "pending_evidence":
                await self._configs.mark_status(
                    decision.challenger_version_id, "shadow"
                )
            return decision
        challenger = await self._configs.get_by_id(decision.challenger_version_id)
        champion = await self._configs.get_by_id(decision.champion_version_id)
        if challenger is None or champion is None:
            raise RuntimeError("promotion_configuration_missing")
        current = await self._champions.get_current_champion()
        if (
            current is not None
            and current.configuration_version_id == challenger.configuration_version_id
        ):
            return decision  # already activated
        await self._champions.promote(
            challenger=challenger,
            expected_champion_id=champion.configuration_version_id,
            reason="agent_promote",
            experiment_id=decision.experiment_id,
            promotion_decision_id=decision.promotion_decision_id,
        )
        await self._configs.mark_status(challenger.configuration_version_id, "champion")
        return decision

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
        """Compatibility wrapper: decide then apply. Prefer separate nodes in orchestrator."""
        decision = await self.decide(
            experiment_id=experiment_id,
            result=result,
            challenger=challenger,
            champion=champion,
            proposal=proposal,
            holdout_episode_count=holdout_episode_count,
            completed_episode_count=completed_episode_count,
            adversarial_passed=adversarial_passed,
            agent_override_action=agent_override_action,
        )
        return await self.apply_persisted_decision(
            promotion_decision_id=decision.promotion_decision_id
        )

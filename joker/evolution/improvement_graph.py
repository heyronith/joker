"""Checkpointed improvement proposal graph driven by ModelRouter agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID, uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from joker.evolution.agent_invoke import invoke_evolution_agent
from joker.evolution.agent_schemas import ImprovementAgentProposal
from joker.evolution.improvement import ImprovementProposalService, parse_cognitive_patch
from joker.evolution.schemas import (
    CognitiveConfigurationVersion,
    EpisodeEvaluation,
    ImprovementProposal,
    PromptPatch,
    TradingEpisode,
)
from joker.models.router import ModelRouter


class ImprovementGraphState(TypedDict, total=False):
    session_id: str
    parent_champion: CognitiveConfigurationVersion
    episodes: list[TradingEpisode]
    evaluations: list[EpisodeEvaluation]
    training_dataset_ids: tuple[UUID, ...]
    challenger_dataset_ids: tuple[UUID, ...]
    evaluation_dataset_ids: tuple[UUID, ...]
    weakness_codes: list[str]
    proposal_draft: ImprovementAgentProposal
    critic_ok: bool
    proposal: ImprovementProposal
    challenger: CognitiveConfigurationVersion
    model_call_ids: list[str]


def _cluster_weaknesses(evaluations: list[EpisodeEvaluation]) -> list[str]:
    counts: dict[str, int] = {}
    for evaluation in evaluations:
        for code in evaluation.avoidable_error_codes:
            counts[code] = counts.get(code, 0) + 1
        if evaluation.calibration_score is not None and evaluation.calibration_score < 0.5:
            counts["poor_calibration"] = counts.get("poor_calibration", 0) + 1
        if evaluation.thesis_quality is not None and evaluation.thesis_quality < 0.4:
            counts["weak_thesis"] = counts.get("weak_thesis", 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [code for code, _ in ranked[:5]] or ["unspecified_cognitive_weakness"]


def build_improvement_graph(
    router: ModelRouter,
    service: ImprovementProposalService,
):
    async def detect_node(state: ImprovementGraphState) -> dict[str, Any]:
        return {"weakness_codes": _cluster_weaknesses(state.get("evaluations") or [])}

    async def propose_node(state: ImprovementGraphState) -> dict[str, Any]:
        champion = state["parent_champion"]
        payload = {
            "weakness_codes": state.get("weakness_codes") or [],
            "evaluation_summaries": [
                {
                    "evaluation_id": str(e.evaluation_id),
                    "avoidable_error_codes": list(e.avoidable_error_codes),
                    "calibration_score": (
                        str(e.calibration_score) if e.calibration_score is not None else None
                    ),
                    "outcome_quality": (
                        str(e.outcome_quality) if e.outcome_quality is not None else None
                    ),
                }
                for e in (state.get("evaluations") or [])[:40]
            ],
            "permitted_patch_types": [
                "prompt",
                "context_policy",
                "routing_policy",
                "debate_policy",
                "memory_policy",
                "escalation_policy",
            ],
        }
        draft, call_id = await invoke_evolution_agent(
            router,
            role="improvement_proposer",
            prompt_id="task3.improvement_proposer",
            prompt_version="3.0.0",
            payload=payload,
            output_type=ImprovementAgentProposal,
            snapshot_id=champion.configuration_version_id,
            cycle_id=f"improve:{champion.configuration_version_id}",
            session_id=state.get("session_id") or "evolution",
        )
        return {
            "proposal_draft": draft,
            "model_call_ids": list(state.get("model_call_ids") or []) + [str(call_id)],
        }

    async def critic_node(state: ImprovementGraphState) -> dict[str, Any]:
        draft = state["proposal_draft"]
        payload = {
            "draft": draft.model_dump(mode="json"),
            "prohibited": [
                "source_code",
                "safety_validator",
                "broker_adapter",
                "risk_governor",
            ],
        }
        critic, call_id = await invoke_evolution_agent(
            router,
            role="improvement_critic",
            prompt_id="task3.improvement_critic",
            prompt_version="3.0.0",
            payload=payload,
            output_type=ImprovementAgentProposal,
            snapshot_id=state["parent_champion"].configuration_version_id,
            cycle_id=f"improve-critic:{uuid4()}",
            session_id=state.get("session_id") or "evolution",
        )
        accepted = bool(critic.critic_accepted) and not critic.critic_rejection_codes
        return {
            "critic_ok": accepted,
            "proposal_draft": critic if accepted else draft,
            "model_call_ids": list(state.get("model_call_ids") or []) + [str(call_id)],
        }

    async def materialise_node(state: ImprovementGraphState) -> dict[str, Any]:
        if not state.get("critic_ok", False):
            raise RuntimeError("improvement critic rejected proposal")
        draft = state["proposal_draft"]
        if draft.patch_type == "prompt":
            patch = PromptPatch(
                role=draft.role,
                parent_prompt_version_id=uuid4(),
                replacement_template=draft.replacement_template
                or "Prefer calibrated no-trade when evidence is weak.",
                change_rationale=draft.change_rationale,
            )
        else:
            patch = parse_cognitive_patch(
                {
                    "patch_type": draft.patch_type,
                    "role": draft.role,
                    "preferred_profile": draft.preferred_profile,
                    "change_rationale": draft.change_rationale,
                    "replacement_template": draft.replacement_template,
                }
            )
        evaluations = state.get("evaluations") or []
        episodes = state.get("episodes") or []
        proposal, challenger = await service.propose(
            parent_champion=state["parent_champion"],
            weakness=draft.weakness or ",".join(state.get("weakness_codes") or []),
            hypothesis=draft.hypothesis,
            patch=patch,
            training_dataset_ids=tuple(state.get("training_dataset_ids") or ()),
            challenger_dataset_ids=tuple(state.get("challenger_dataset_ids") or ()),
            evaluation_dataset_ids=tuple(state.get("evaluation_dataset_ids") or ()),
            supporting_episode_ids=tuple(e.episode_id for e in episodes[:20]),
            supporting_evaluation_ids=tuple(e.evaluation_id for e in evaluations[:20]),
            metrics_to_improve=draft.metrics_to_improve,
            metrics_must_not_regress=draft.metrics_must_not_regress,
        )
        return {"proposal": proposal, "challenger": challenger}

    graph = StateGraph(ImprovementGraphState)
    graph.add_node("detect", detect_node)
    graph.add_node("propose", propose_node)
    graph.add_node("critic", critic_node)
    graph.add_node("materialise", materialise_node)
    graph.add_edge(START, "detect")
    graph.add_edge("detect", "propose")
    graph.add_edge("propose", "critic")
    graph.add_edge("critic", "materialise")
    graph.add_edge("materialise", END)
    return graph


class ImprovementGraphRunner:
    def __init__(
        self,
        *,
        router: ModelRouter,
        service: ImprovementProposalService,
        checkpointer_path: Path | None = None,
        checkpointer_saver: AsyncSqliteSaver | None = None,
        session_id: str = "evolution",
    ) -> None:
        self._router = router
        self._service = service
        self._checkpointer_path = checkpointer_path
        self._checkpointer_saver = checkpointer_saver
        self._session_id = session_id
        self._compiled = None

    def _graph(self):
        if self._compiled is not None:
            return self._compiled
        builder = build_improvement_graph(self._router, self._service)
        if self._checkpointer_saver is not None:
            self._compiled = builder.compile(checkpointer=self._checkpointer_saver)
        else:
            self._compiled = builder.compile()
        return self._compiled

    async def run(
        self,
        *,
        parent_champion: CognitiveConfigurationVersion,
        episodes: list[TradingEpisode],
        evaluations: list[EpisodeEvaluation],
        training_dataset_ids: tuple[UUID, ...],
        challenger_dataset_ids: tuple[UUID, ...] = (),
        evaluation_dataset_ids: tuple[UUID, ...] = (),
        evaluation_window_hash: str | None = None,
    ) -> tuple[ImprovementProposal, CognitiveConfigurationVersion]:
        state: ImprovementGraphState = {
            "session_id": self._session_id,
            "parent_champion": parent_champion,
            "episodes": episodes,
            "evaluations": evaluations,
            "training_dataset_ids": tuple(training_dataset_ids),
            "challenger_dataset_ids": tuple(challenger_dataset_ids),
            "evaluation_dataset_ids": tuple(evaluation_dataset_ids),
            "model_call_ids": [],
        }
        compiled = self._graph()
        if self._checkpointer_saver is not None:
            window = evaluation_window_hash or "default"
            thread_id = (
                f"{self._session_id}:{window}:{parent_champion.configuration_version_id}"
            )
            result = await compiled.ainvoke(
                state, config={"configurable": {"thread_id": thread_id}}
            )
        else:
            result = await compiled.ainvoke(state)
        return result["proposal"], result["challenger"]

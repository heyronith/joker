"""Checkpointed LangGraph evaluation using ModelRouter evaluator agents."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph

from joker.evaluation.graph import (
    compute_outcome_metrics,
    evaluate_dimensions,
    synthesise_episode_evaluation,
    validate_episode_integrity,
)
from joker.evaluation.metrics import compute_deterministic_metrics
from joker.evolution.agent_invoke import invoke_evolution_agent
from joker.evolution.agent_schemas import EvaluatorAgentScores
from joker.evolution.repositories import (
    DecisionTraceRepository,
    EpisodeEvaluationRepository,
)
from joker.evolution.schemas import EpisodeEvaluation, TradingEpisode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from joker.graph.langgraph_checkpointer import ainvoke_config
from joker.models.router import ModelRouter

EVALUATOR_ROLES = (
    "evaluator_thesis",
    "evaluator_calibration",
    "evaluator_execution",
    "evaluator_efficiency",
)


class AgenticEvaluationState(TypedDict, total=False):
    episode: TradingEpisode
    session_id: str
    evaluator_version: str
    metrics: Any
    agent_scores: dict[str, Decimal | None]
    avoidable_error_codes: list[str]
    safety_violation_codes: list[str]
    integrity_violation_codes: list[str]
    invalid_reasons: list[str]
    valid: bool
    model_call_ids: list[str]
    completed_evaluator_roles: list[str]
    evaluation: EpisodeEvaluation
    thesis_quality: Decimal | None
    evidence_grounding_score: Decimal | None
    calibration_score: Decimal | None
    debate_quality: Decimal | None
    execution_quality: Decimal | None
    position_management_score: Decimal | None
    efficiency_score: Decimal | None
    outcome_quality: Decimal | None
    risk_adjusted_outcome: Decimal | None
    decision_consistency_score: Decimal | None


def build_evaluation_graph(router: ModelRouter):
    """Compile checkpointable evaluation graph requiring ModelRouter."""

    async def integrity_node(state: AgenticEvaluationState) -> dict[str, Any]:
        return validate_episode_integrity(state)  # type: ignore[arg-type]

    async def metrics_node(state: AgenticEvaluationState) -> dict[str, Any]:
        return compute_outcome_metrics(state)  # type: ignore[arg-type]

    async def agents_node(state: AgenticEvaluationState) -> dict[str, Any]:
        # Retained for backwards-compatible single-node evaluation; prefer per-role nodes.
        episode = state["episode"]
        merged: dict[str, Decimal | None] = dict(state.get("agent_scores") or {})
        avoidable = list(state.get("avoidable_error_codes") or [])
        model_calls = list(state.get("model_call_ids") or [])
        metrics = state.get("metrics") or compute_deterministic_metrics(episode)
        evidence = {
            "episode": episode.model_dump(mode="json"),
            "metrics": metrics.model_dump(mode="json"),
            "assembled_at": datetime.now(timezone.utc).isoformat(),
        }
        for role in EVALUATOR_ROLES:
            scores, call_id = await invoke_evolution_agent(
                router,
                role=role,
                prompt_id=f"task3.{role}",
                prompt_version="3.0.0",
                payload=evidence,
                output_type=EvaluatorAgentScores,
                snapshot_id=episode.initial_snapshot_id,
                cycle_id=str(episode.episode_id),
                session_id=state.get("session_id") or episode.session_id,
            )
            model_calls.append(str(call_id))
            for field_name in (
                "thesis_quality",
                "evidence_grounding_score",
                "calibration_score",
                "debate_quality",
                "execution_quality",
                "position_management_score",
                "efficiency_score",
                "decision_consistency_score",
            ):
                value = getattr(scores, field_name)
                if value is not None:
                    merged[field_name] = value
            avoidable.extend(scores.avoidable_error_codes)
        return {
            "agent_scores": merged,
            "avoidable_error_codes": avoidable,
            "model_call_ids": model_calls,
        }

    def _make_evaluator_node(role: str):
        async def evaluator_node(state: AgenticEvaluationState) -> dict[str, Any]:
            episode = state["episode"]
            merged: dict[str, Decimal | None] = dict(state.get("agent_scores") or {})
            avoidable = list(state.get("avoidable_error_codes") or [])
            model_calls = list(state.get("model_call_ids") or [])
            # Skip if this role already contributed a model call (resume).
            role_marker = f"role:{role}"
            completed_roles = list(state.get("completed_evaluator_roles") or [])
            if role in completed_roles:
                return {}
            metrics = state.get("metrics") or compute_deterministic_metrics(episode)
            evidence = {
                "episode": episode.model_dump(mode="json"),
                "metrics": metrics.model_dump(mode="json"),
                "assembled_at": datetime.now(timezone.utc).isoformat(),
                "role_marker": role_marker,
            }
            scores, call_id = await invoke_evolution_agent(
                router,
                role=role,
                prompt_id=f"task3.{role}",
                prompt_version="3.0.0",
                payload=evidence,
                output_type=EvaluatorAgentScores,
                snapshot_id=episode.initial_snapshot_id,
                cycle_id=str(episode.episode_id),
                session_id=state.get("session_id") or episode.session_id,
            )
            model_calls.append(str(call_id))
            for field_name in (
                "thesis_quality",
                "evidence_grounding_score",
                "calibration_score",
                "debate_quality",
                "execution_quality",
                "position_management_score",
                "efficiency_score",
                "decision_consistency_score",
            ):
                value = getattr(scores, field_name)
                if value is not None:
                    merged[field_name] = value
            avoidable.extend(scores.avoidable_error_codes)
            completed_roles.append(role)
            return {
                "agent_scores": merged,
                "avoidable_error_codes": avoidable,
                "model_call_ids": model_calls,
                "completed_evaluator_roles": completed_roles,
            }

        return evaluator_node

    async def dimensions_node(state: AgenticEvaluationState) -> dict[str, Any]:
        return evaluate_dimensions(
            state,  # type: ignore[arg-type]
            agent_scores=state.get("agent_scores"),
        )

    async def synthesise_node(state: AgenticEvaluationState) -> dict[str, Any]:
        return synthesise_episode_evaluation(state)  # type: ignore[arg-type]

    graph = StateGraph(AgenticEvaluationState)
    graph.add_node("integrity", integrity_node)
    graph.add_node("metrics", metrics_node)
    for role in EVALUATOR_ROLES:
        graph.add_node(role, _make_evaluator_node(role))
    graph.add_node("dimensions", dimensions_node)
    graph.add_node("synthesise", synthesise_node)
    graph.add_edge(START, "integrity")
    graph.add_edge("integrity", "metrics")
    graph.add_edge("metrics", EVALUATOR_ROLES[0])
    for left, right in zip(EVALUATOR_ROLES, EVALUATOR_ROLES[1:]):
        graph.add_edge(left, right)
    graph.add_edge(EVALUATOR_ROLES[-1], "dimensions")
    graph.add_edge("dimensions", "synthesise")
    graph.add_edge("synthesise", END)
    return graph


class AgenticEvaluationGraphRunner:
    """Run checkpointed evaluation via ModelRouter agents and persist the result."""

    def __init__(
        self,
        evaluation_repo: EpisodeEvaluationRepository,
        trace_repo: DecisionTraceRepository | None = None,
        *,
        router: ModelRouter,
        evaluator_version: str = "3.0.0",
        checkpointer_path: Path | None = None,
        checkpointer_saver: AsyncSqliteSaver | None = None,
    ) -> None:
        self._evaluations = evaluation_repo
        self._traces = trace_repo
        self._router = router
        self._evaluator_version = evaluator_version
        self._checkpointer_path = checkpointer_path
        self._checkpointer_saver = checkpointer_saver
        self._compiled = None

    def _thread_id(self, episode: TradingEpisode) -> str:
        return (
            f"{episode.session_id}:{episode.episode_id}:{self._evaluator_version}"
        )

    def _graph(self):
        if self._compiled is not None:
            return self._compiled
        builder = build_evaluation_graph(self._router)
        if self._checkpointer_saver is not None:
            self._compiled = builder.compile(checkpointer=self._checkpointer_saver)
        else:
            self._compiled = builder.compile()
        return self._compiled

    async def evaluate(
        self,
        episode: TradingEpisode,
        *,
        force_invalid_reasons: tuple[str, ...] = (),
    ) -> EpisodeEvaluation:
        existing_list = await self._evaluations.list_by_episode(episode.episode_id)
        if existing_list:
            return existing_list[0]
        state: AgenticEvaluationState = {
            "episode": episode,
            "session_id": episode.session_id,
            "evaluator_version": self._evaluator_version,
            "valid": True,
            "invalid_reasons": list(force_invalid_reasons),
            "avoidable_error_codes": [],
            "safety_violation_codes": [],
            "integrity_violation_codes": [],
            "agent_scores": {},
            "model_call_ids": [],
            "completed_evaluator_roles": [],
        }
        compiled = self._graph()
        if self._checkpointer_saver is not None:
            config = {
                "configurable": {"thread_id": self._thread_id(episode)}
            }
            result_state = await compiled.ainvoke(state, config=config)
        else:
            result_state = await compiled.ainvoke(state)

        evaluation: EpisodeEvaluation = result_state["evaluation"]
        await self._evaluations.append(evaluation)
        return evaluation

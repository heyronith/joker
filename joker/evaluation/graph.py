"""Checkpointed episode evaluation pipeline (LangGraph-compatible node sequence)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, TypedDict
from uuid import UUID, uuid4

from joker.evaluation.metrics import compute_deterministic_metrics
from joker.evaluation.schemas import DeterministicOutcomeMetrics, EvidencePackage
from joker.evolution.hashing import hash_model
from joker.evolution.idempotency import evaluation_idempotency_key
from joker.evolution.repositories import (
    DecisionTraceRepository,
    EpisodeEvaluationRepository,
)
from joker.evolution.schemas import EpisodeEvaluation, TradingEpisode, assert_no_chain_of_thought


class EvaluationState(TypedDict, total=False):
    episode: TradingEpisode
    metrics: DeterministicOutcomeMetrics
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
    avoidable_error_codes: list[str]
    safety_violation_codes: list[str]
    integrity_violation_codes: list[str]
    valid: bool
    invalid_reasons: list[str]
    evaluation: EpisodeEvaluation
    evaluator_version: str


def validate_episode_integrity(state: EvaluationState) -> EvaluationState:
    episode = state["episode"]
    reasons: list[str] = list(state.get("invalid_reasons") or [])
    integrity: list[str] = list(state.get("integrity_violation_codes") or [])
    valid = True
    if not episode.completed and "incomplete_excluded" not in reasons:
        # Incomplete episodes may still be scored but marked invalid for promotion.
        reasons.append("episode_incomplete")
        valid = False
    if episode.action_class == "closed_trade" and episode.realised_pnl is None:
        integrity.append("missing_realised_pnl")
        reasons.append("missing_realised_pnl")
        valid = False
    if not episode.initial_snapshot_id:
        integrity.append("missing_initial_snapshot")
        valid = False
    return {
        **state,
        "valid": valid and state.get("valid", True),
        "invalid_reasons": reasons,
        "integrity_violation_codes": integrity,
    }


def compute_outcome_metrics(state: EvaluationState) -> EvaluationState:
    episode = state["episode"]
    metrics = compute_deterministic_metrics(episode)
    return {**state, "metrics": metrics}


def _score_from_metrics(metrics: DeterministicOutcomeMetrics) -> dict[str, Decimal | None]:
    pnl = metrics.realised_pnl
    outcome = None
    if pnl is not None:
        # Bounded utility in [-1, 1] for small 0DTE premiums.
        outcome = max(Decimal("-1"), min(Decimal("1"), pnl / Decimal("50")))
    risk_adj = None
    if pnl is not None and metrics.max_adverse_excursion is not None:
        mae = abs(metrics.max_adverse_excursion) or Decimal("0.01")
        risk_adj = (pnl / mae).quantize(Decimal("0.0001"))
    return {"outcome_quality": outcome, "risk_adjusted_outcome": risk_adj}


def evaluate_dimensions(
    state: EvaluationState,
    *,
    agent_scores: dict[str, Decimal | None] | None = None,
) -> EvaluationState:
    """Merge deterministic scores with optional typed evaluator agent scores."""
    metrics = state["metrics"]
    base = _score_from_metrics(metrics)
    scores = {**base, **(agent_scores or {})}
    avoidable = list(state.get("avoidable_error_codes") or [])
    # Profitable but unsupported / losing but calibrated handled by callers via agent_scores.
    if metrics.model_call_count > 40:
        avoidable.append("excessive_model_calls")
    if metrics.safety_violations:
        state = {
            **state,
            "safety_violation_codes": list(state.get("safety_violation_codes") or [])
            + ["safety_violation"],
        }
    return {
        **state,
        **scores,
        "avoidable_error_codes": avoidable,
        "calibration_score": scores.get("calibration_score", metrics.calibration_error),
        "efficiency_score": scores.get(
            "efficiency_score",
            (
                None
                if metrics.model_call_count == 0
                else (Decimal("1") / Decimal(metrics.model_call_count)).quantize(
                    Decimal("0.0001")
                )
            ),
        ),
        "execution_quality": scores.get(
            "execution_quality", metrics.exit_efficiency
        ),
        "position_management_score": scores.get(
            "position_management_score", metrics.profit_capture_ratio
        ),
    }


def synthesise_episode_evaluation(state: EvaluationState) -> EvaluationState:
    episode = state["episode"]
    metrics = state["metrics"]
    evaluator_version = state.get("evaluator_version") or "3.0.0"
    key = evaluation_idempotency_key(
        episode.episode_id, evaluator_version, episode.configuration_version_id
    )
    det: dict[str, Decimal | int | str | bool] = {}
    for field, value in metrics.model_dump().items():
        if value is not None:
            det[field] = value
    evaluation = EpisodeEvaluation(
        evaluation_id=uuid4(),
        episode_id=episode.episode_id,
        evaluator_version=evaluator_version,
        outcome_quality=state.get("outcome_quality"),
        risk_adjusted_outcome=state.get("risk_adjusted_outcome"),
        calibration_score=state.get("calibration_score"),
        thesis_quality=state.get("thesis_quality"),
        evidence_grounding_score=state.get("evidence_grounding_score"),
        debate_quality=state.get("debate_quality"),
        decision_consistency_score=state.get("decision_consistency_score"),
        execution_quality=state.get("execution_quality"),
        position_management_score=state.get("position_management_score"),
        efficiency_score=state.get("efficiency_score"),
        avoidable_error_codes=tuple(state.get("avoidable_error_codes") or ()),
        safety_violation_codes=tuple(state.get("safety_violation_codes") or ()),
        integrity_violation_codes=tuple(state.get("integrity_violation_codes") or ()),
        deterministic_metrics=det,
        valid=bool(state.get("valid", True)),
        invalid_reasons=tuple(state.get("invalid_reasons") or ()),
        configuration_version_id=episode.configuration_version_id,
        idempotency_key=key,
        content_hash="",
    )
    evaluation = evaluation.model_copy(
        update={"content_hash": hash_model(evaluation, exclude={"created_at"})}
    )
    assert_no_chain_of_thought(evaluation.model_dump(mode="json"))
    return {**state, "evaluation": evaluation}


class EvaluationGraphRunner:
    """Run the evaluation node sequence and persist the result."""

    def __init__(
        self,
        evaluation_repo: EpisodeEvaluationRepository,
        trace_repo: DecisionTraceRepository | None = None,
        *,
        evaluator_version: str = "3.0.0",
    ) -> None:
        self._evaluations = evaluation_repo
        self._traces = trace_repo
        self._evaluator_version = evaluator_version

    async def evaluate(
        self,
        episode: TradingEpisode,
        *,
        agent_scores: dict[str, Decimal | None] | None = None,
        force_invalid_reasons: tuple[str, ...] = (),
    ) -> EpisodeEvaluation:
        state: EvaluationState = {
            "episode": episode,
            "evaluator_version": self._evaluator_version,
            "valid": True,
            "invalid_reasons": list(force_invalid_reasons),
            "avoidable_error_codes": [],
            "safety_violation_codes": [],
            "integrity_violation_codes": [],
        }
        state = validate_episode_integrity(state)
        state = compute_outcome_metrics(state)
        state = evaluate_dimensions(state, agent_scores=agent_scores)
        state = synthesise_episode_evaluation(state)
        evaluation = state["evaluation"]
        await self._evaluations.append(evaluation)
        return evaluation

    def build_evidence_package(
        self, episode: TradingEpisode, metrics: DeterministicOutcomeMetrics
    ) -> EvidencePackage:
        return EvidencePackage(
            episode=episode,
            metrics=metrics,
            artifact_refs=episode.cognitive_artifact_ids,
            snapshot_ids=tuple(
                x
                for x in (episode.initial_snapshot_id, episode.terminal_snapshot_id)
                if x is not None
            ),
            assembled_at=datetime.now(timezone.utc),
        )

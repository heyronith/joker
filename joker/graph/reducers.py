"""Deterministic LangGraph reducers for cognitive graph state."""

from __future__ import annotations

import json
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from joker.cognition.schemas import (
    AgentEvidence,
    CognitiveError,
    DebateReview,
    GraphNodeTrace,
    PatternHypothesis,
    StrategyHypothesis,
)

T = TypeVar("T", bound=BaseModel)


def _payload_key(item: BaseModel) -> str:
    return json.dumps(item.model_dump(mode="json"), sort_keys=True, default=str)


def _merge_models(
    existing: list[T] | None,
    incoming: list[T] | None,
    *,
    id_attr: str,
    sort_key: Callable[[T], str],
) -> list[T]:
    """Merge artefact lists by ID with stable ordering and conflict rejection."""
    left = list(existing or [])
    right = list(incoming or [])
    merged: dict[str, T] = {}
    for item in left + right:
        item_id = str(getattr(item, id_attr))
        prior = merged.get(item_id)
        if prior is None:
            merged[item_id] = item
            continue
        if _payload_key(prior) != _payload_key(item):
            # Reject conflicting duplicate — retain first-seen payload.
            continue
    return sorted(merged.values(), key=sort_key)


def merge_evidence(
    existing: list[AgentEvidence] | None,
    incoming: list[AgentEvidence] | None,
) -> list[AgentEvidence]:
    return _merge_models(
        existing,
        incoming,
        id_attr="evidence_id",
        sort_key=lambda item: str(item.evidence_id),
    )


def merge_hypotheses(
    existing: list[PatternHypothesis] | None,
    incoming: list[PatternHypothesis] | None,
) -> list[PatternHypothesis]:
    return _merge_models(
        existing,
        incoming,
        id_attr="hypothesis_id",
        sort_key=lambda item: str(item.hypothesis_id),
    )


def merge_strategies(
    existing: list[StrategyHypothesis] | None,
    incoming: list[StrategyHypothesis] | None,
) -> list[StrategyHypothesis]:
    return _merge_models(
        existing,
        incoming,
        id_attr="strategy_id",
        sort_key=lambda item: str(item.strategy_id),
    )


def merge_reviews(
    existing: list[DebateReview] | None,
    incoming: list[DebateReview] | None,
) -> list[DebateReview]:
    return _merge_models(
        existing,
        incoming,
        id_attr="review_id",
        sort_key=lambda item: str(item.review_id),
    )


def merge_traces(
    existing: list[GraphNodeTrace] | None,
    incoming: list[GraphNodeTrace] | None,
) -> list[GraphNodeTrace]:
    return _merge_models(
        existing,
        incoming,
        id_attr="trace_id",
        sort_key=lambda item: f"{item.node_name}:{item.trace_id}",
    )


def merge_errors(
    existing: list[CognitiveError] | None,
    incoming: list[CognitiveError] | None,
) -> list[CognitiveError]:
    return _merge_models(
        existing,
        incoming,
        id_attr="error_id",
        sort_key=lambda item: str(item.error_id),
    )


def find_conflicting_ids(
    existing: list[T] | None,
    incoming: list[T] | None,
    *,
    id_attr: str,
) -> list[str]:
    """Return IDs present in both lists with differing payloads (for tests/errors)."""
    left = {str(getattr(item, id_attr)): item for item in (existing or [])}
    conflicts: list[str] = []
    for item in incoming or []:
        item_id = str(getattr(item, id_attr))
        prior = left.get(item_id)
        if prior is not None and _payload_key(prior) != _payload_key(item):
            conflicts.append(item_id)
    return conflicts


def reducer_conflict_error(
    *,
    node_name: str,
    field_name: str,
    conflicting_ids: list[str],
) -> CognitiveError:
    return CognitiveError(
        node_name=node_name,
        error_code="reducer_conflict",
        message=f"conflicting duplicate {field_name} IDs: {', '.join(conflicting_ids)}",
        recoverable=True,
        context={"field": field_name, "ids": conflicting_ids},
    )

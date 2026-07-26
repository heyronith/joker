"""Helpers for cognitive graph node tracing and error recording."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from joker.cognition.schemas import AgentRole, CognitiveError, GraphNodeTrace
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.reducers import find_conflicting_ids, reducer_conflict_error


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_trace(
    state: CognitiveGraphState,
    *,
    node_name: str,
    status: str,
    agent_role: AgentRole | None = None,
    artifact_ids: tuple[UUID, ...] = (),
    error_code: str | None = None,
    started_at: datetime | None = None,
) -> GraphNodeTrace:
    return GraphNodeTrace(
        node_name=node_name,
        agent_role=agent_role,
        started_at=started_at or utc_now(),
        completed_at=utc_now(),
        status=status,  # type: ignore[arg-type]
        artifact_ids=artifact_ids,
        error_code=error_code,
    )


def trace_update(*traces: GraphNodeTrace) -> dict[str, list[GraphNodeTrace]]:
    return {"node_trace": list(traces)}


def append_error(
    state: CognitiveGraphState,
    *,
    node_name: str,
    error_code: str,
    message: str,
    recoverable: bool = True,
    agent_role: AgentRole | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, list[CognitiveError]]:
    err = CognitiveError(
        node_name=node_name,
        agent_role=agent_role,
        error_code=error_code,
        message=message,
        recoverable=recoverable,
        context=context,
    )
    return {"errors": [err]}


def check_reducer_conflicts(
    state: CognitiveGraphState,
    incoming: list[Any],
    *,
    node_name: str,
    field_name: str,
    id_attr: str,
    existing: list[Any] | None,
) -> dict[str, list[CognitiveError]]:
    conflicts = find_conflicting_ids(existing, incoming, id_attr=id_attr)
    if not conflicts:
        return {}
    return append_error(
        state,
        node_name=node_name,
        error_code="reducer_conflict",
        message=reducer_conflict_error(
            node_name=node_name,
            field_name=field_name,
            conflicting_ids=conflicts,
        ).message,
    )


def graph_limit_error(node_name: str, limit_name: str) -> dict[str, list[CognitiveError]]:
    return {
        "errors": [
            CognitiveError(
                node_name=node_name,
                error_code="graph_limit_exceeded",
                message=f"graph limit exceeded: {limit_name}",
                recoverable=True,
                context={"limit": limit_name},
            )
        ]
    }

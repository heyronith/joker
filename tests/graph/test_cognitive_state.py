"""Tests for CognitiveGraphState contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import get_type_hints, get_args, get_origin
from typing_extensions import Annotated

from joker.cognition.schemas import AgentEvidence, CognitiveError, GraphNodeTrace
from joker.graph.cognitive_state import CognitiveGraphState
from joker.graph.reducers import merge_evidence


def test_cognitive_graph_state_has_required_fields() -> None:
    hints = get_type_hints(CognitiveGraphState, include_extras=True)
    assert "session_id" in hints
    assert "cycle_id" in hints
    assert "evidence" in hints
    assert "meta_decision" in hints
    assert "execution_proposal" in hints


def test_evidence_field_uses_merge_reducer() -> None:
    hints = get_type_hints(CognitiveGraphState, include_extras=True)
    evidence_hint = hints["evidence"]
    assert get_origin(evidence_hint) is Annotated
    args = get_args(evidence_hint)
    assert args[0] == list[AgentEvidence]
    assert args[1] is merge_evidence


def test_initial_state_shape() -> None:
    state: CognitiveGraphState = {
        "session_id": "sess-1",
        "run_id": "run-1",
        "cycle_id": "cycle-1",
        "snapshot_id": "snap-1",
        "started_at": datetime.now(timezone.utc),
        "evidence": [],
        "errors": [],
        "node_trace": [],
    }
    assert state["session_id"] == "sess-1"

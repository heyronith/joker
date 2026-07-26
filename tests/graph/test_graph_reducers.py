"""Tests for cognitive graph reducers."""

from __future__ import annotations

from uuid import uuid4

from joker.cognition.schemas import AgentEvidence, AgentRole, CognitiveError, EvidenceReference
from joker.graph.reducers import (
    find_conflicting_ids,
    merge_errors,
    merge_evidence,
    merge_hypotheses,
)
from joker.models.schemas import utc_now


def _evidence(evidence_id=None, claim: str = "a") -> AgentEvidence:
    snap = uuid4()
    ref = EvidenceReference(
        snapshot_id=snap,
        source_type="underlying",
        source_id="x",
        observed_at=utc_now(),
        value_summary="v",
    )
    return AgentEvidence(
        evidence_id=evidence_id or uuid4(),
        session_id="s",
        snapshot_id=snap,
        prompt_version="v1",
        model_call_id=uuid4(),
        cycle_id="c1",
        agent_role=AgentRole.ANOMALY,
        claim=claim,
        confidence=0.5,
        supporting_references=(ref,),
    )


def test_merge_evidence_stable_sort() -> None:
    a = _evidence(claim="first")
    b = _evidence(claim="second")
    merged = merge_evidence([b], [a])
    ids = [str(item.evidence_id) for item in merged]
    assert ids == sorted(ids)


def test_merge_evidence_rejects_conflicting_duplicate_ids() -> None:
    eid = uuid4()
    left = _evidence(eid, claim="left")
    right = _evidence(eid, claim="right")
    merged = merge_evidence([left], [right])
    assert len(merged) == 1
    assert merged[0].claim == "left"
    conflicts = find_conflicting_ids([left], [right], id_attr="evidence_id")
    assert str(eid) in conflicts


def test_merge_errors_preserves_distinct() -> None:
    e1 = CognitiveError(error_code="a", message="one")
    e2 = CognitiveError(error_code="b", message="two")
    merged = merge_errors([e1], [e2])
    assert len(merged) == 2


def test_merge_hypotheses_empty() -> None:
    assert merge_hypotheses(None, None) == []

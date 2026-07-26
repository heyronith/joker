"""Cognitive schema validation and provenance invariants."""

from __future__ import annotations

from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from joker.cognition.prompts import all_prompts, get_prompt
from joker.cognition.schemas import (
    AgentEvidence,
    AgentRole,
    MarketDirection,
    MetaDecision,
    MetaDecisionAction,
)


def test_prompt_roles_are_behaviourally_distinct() -> None:
    prompts = all_prompts()
    assert len(prompts) == len(AgentRole)
    texts = [(p.system_template[:120], p.output_schema_name) for p in prompts]
    assert len({t[0] for t in texts}) >= 10
    assert len({t[1] for t in texts}) >= 5
    assert get_prompt(AgentRole.FALSIFIER).system_template != get_prompt(
        AgentRole.BULLISH_INVENTOR
    ).system_template


def test_directional_evidence_requires_supporting_references() -> None:
    with pytest.raises(Exception):
        AgentEvidence(
            session_id="s",
            cycle_id="c",
            snapshot_id=uuid4(),
            agent_role=AgentRole.MARKET_STRUCTURE,
            claim="bullish structure",
            direction=MarketDirection.BULLISH,
            confidence=0.7,
            supporting_references=(),
            contradicting_references=(),
            invalidation_conditions=("break below VWAP",),
            prompt_version="1",
            model_call_id=uuid4(),
        )


@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=40)
def test_evidence_confidence_bounded(raw: float) -> None:
    kwargs = dict(
        session_id="s",
        cycle_id="c",
        snapshot_id=uuid4(),
        agent_role=AgentRole.ANOMALY,
        claim="neutral",
        direction=None,
        confidence=raw,
        supporting_references=(),
        contradicting_references=(),
        invalidation_conditions=(),
        requires_more_data=True,
        prompt_version="1",
        model_call_id=uuid4(),
    )
    if 0.0 <= raw <= 1.0:
        ev = AgentEvidence(**kwargs)
        assert 0.0 <= ev.confidence <= 1.0
    else:
        with pytest.raises(Exception):
            AgentEvidence(**kwargs)


def test_meta_decision_abandon_is_valid() -> None:
    md = MetaDecision(
        session_id="s",
        cycle_id="c",
        snapshot_id=uuid4(),
        action=MetaDecisionAction.ABANDON,
        selected_strategy_id=None,
        confidence=0.4,
        rationale_summary="no edge",
        prompt_version="1",
        model_call_id=uuid4(),
    )
    assert md.action == MetaDecisionAction.ABANDON

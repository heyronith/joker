"""Phase 3 domain schema tests."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from joker.schemas.domain import (
    AgentOpinion,
    OptionContract,
    Playbook,
    PlaybookSetup,
    SCHEMA_VERSION,
    TradeCandidate,
)
from tests.fixtures.domain import make_candidate, make_contract, make_quote


def test_valid_fixtures_parse() -> None:
    candidate = make_candidate()
    assert candidate.direction == "long_call"
    assert candidate.schema_version == SCHEMA_VERSION


def test_invalid_missing_fields_fail() -> None:
    with pytest.raises(ValidationError):
        TradeCandidate.model_validate({"run_id": "x"})


def test_non_0dte_rejected() -> None:
    with pytest.raises(ValidationError, match="0DTE"):
        OptionContract(
            expiration=date.today(),
            strike=550,
            option_type="call",
            is_0dte=False,
        )


def test_json_roundtrip() -> None:
    candidate = make_candidate()
    restored = TradeCandidate.model_validate_json(candidate.model_dump_json())
    assert restored.candidate_id == candidate.candidate_id


def test_schema_version_on_major_objects() -> None:
    pb = Playbook(
        trading_day=date.today(),
        title="Test",
        summary="s",
        setups=[
            PlaybookSetup(
                name="s1",
                direction="long_call",
                stop_rule="50%",
                take_profit_rule="100%",
            )
        ],
    )
    assert pb.schema_version == SCHEMA_VERSION
    opinion = AgentOpinion(agent_name="a", summary="s", confidence=0.5)
    assert opinion.schema_version == SCHEMA_VERSION

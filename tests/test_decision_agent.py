"""Agent-led decision path tests."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.agents.decision import decision_to_proposal, mock_decision
from joker.schemas.domain import (
    IntradayDecision,
    Playbook,
    PlaybookSetup,
    TechnicalFeatures,
)


def _features(**kwargs) -> TechnicalFeatures:
    defaults = {
        "symbol": "SPY",
        "as_of": datetime.now(timezone.utc),
        "trend_label": "up",
        "momentum_5m": 0.2,
        "distance_from_vwap_pct": 0.1,
        "volume_confirmed": True,
    }
    defaults.update(kwargs)
    return TechnicalFeatures(**defaults)


def _playbook() -> Playbook:
    return Playbook(
        trading_day=datetime.now(timezone.utc).date(),
        title="test",
        summary="test playbook",
        setups=[
            PlaybookSetup(
                setup_id="long_call_1",
                name="long call",
                direction="long_call",
                enabled=True,
                stop_rule="pct",
                take_profit_rule="pct",
                stop_pct=0.4,
                take_profit_pct=0.8,
            )
        ],
    )


def test_mock_decision_hold_without_setup_match() -> None:
    pb = _playbook()
    # Features that typically won't match without signal rules firing
    d = mock_decision(_features(trend_label="unknown", momentum_5m=None), pb)
    assert d.action in ("hold", "enter")
    assert isinstance(d, IntradayDecision)


def test_mock_decision_force_enter() -> None:
    d = mock_decision(_features(), _playbook(), force_enter=True)
    assert d.action == "enter"
    assert d.direction == "long_call"
    assert d.confidence >= 0.45


def test_decision_to_proposal_maps_enter() -> None:
    d = IntradayDecision(
        action="confirm",
        direction="long_put",
        confidence=0.8,
        rationale="test",
        stop_pct=0.35,
        take_profit_pct=0.9,
        summary="enter put",
    )
    prop = decision_to_proposal(d, run_id="r1", playbook=_playbook())
    assert prop is not None
    assert prop.propose_entry is True
    assert prop.direction == "long_put"
    assert prop.stop_pct == 0.35


def test_decision_to_proposal_hold_is_none() -> None:
    d = IntradayDecision(action="hold", summary="wait")
    assert decision_to_proposal(d, run_id="r1", playbook=_playbook()) is None

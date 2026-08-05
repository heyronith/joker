"""Objective decision fingerprint material-change and refresh semantics."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4
from zoneinfo import ZoneInfo

from joker.objectives.decision_fingerprint import ObjectiveDecisionFingerprint
from joker.objectives.schemas import SessionObjectiveState

ET = ZoneInfo("America/New_York")


def _state(**overrides: object) -> SessionObjectiveState:
    data = {
        "objective_id": uuid4(),
        "session_id": "fp-sess",
        "status": "active",
        "authorised_capital_usd": Decimal("500"),
        "target_profit_usd": Decimal("50"),
        "target_ending_equity_usd": Decimal("550"),
        "working_order_reservation_usd": Decimal("0"),
        "filled_position_exposure_usd": Decimal("0"),
        "reserved_capital_usd": Decimal("0"),
        "available_capital_usd": Decimal("500"),
        "realised_pnl_usd": Decimal("0"),
        "unrealised_pnl_usd": Decimal("0"),
        "progress_to_goal_pct": Decimal("0"),
        "required_profit_remaining_usd": Decimal("50"),
        "time_remaining_seconds": 1800,
        "objective_duration_seconds": 3600,
        "version": 1,
        "max_concurrent_positions": 2,
        "open_position_count": 0,
        "deadline_exchange_time": datetime(2026, 8, 5, 12, 0, tzinfo=ET),
        "entries_paused": False,
        "truth_degraded": False,
    }
    data.update(overrides)
    return SessionObjectiveState.model_validate(data)


def _fp(state: SessionObjectiveState, **kwargs) -> ObjectiveDecisionFingerprint:
    defaults = {
        "working_order_count": 0,
        "broker_identity": "PaperBroker",
        "broker_eligible": True,
        "reconciliation_eligible": True,
    }
    defaults.update(kwargs)
    return ObjectiveDecisionFingerprint.from_state(state, **defaults)


def test_unchanged_truth_refresh_allows_first_component_submission() -> None:
    base = _state()
    evaluated = _fp(base)
    refreshed = ObjectiveDecisionFingerprint.from_json(evaluated.canonical_json)
    # One-second deadline clock decay is non-material.
    refreshed_decay = _fp(
        base.model_copy(update={"time_remaining_seconds": 1799})
    )
    assert evaluated.material_differences(refreshed) == ()
    assert evaluated.material_differences(refreshed_decay) == ()


def test_unchanged_truth_refresh_uses_current_reservation_version() -> None:
    """After a component reservation, the post-submission fingerprint is expected."""
    baseline = _fp(_state())
    after_reservation = _fp(
        _state(
            working_order_reservation_usd=Decimal("120"),
            reserved_capital_usd=Decimal("120"),
            available_capital_usd=Decimal("380"),
            version=2,
        ),
        working_order_count=1,
    )
    # Material vs original evaluated fingerprint — requires reoptimization if stale.
    assert "working_order_reservation_usd" in baseline.material_differences(
        after_reservation
    )
    # Restart / sequential path uses the current reservation fingerprint as expected.
    assert after_reservation.material_differences(
        ObjectiveDecisionFingerprint.from_json(after_reservation.canonical_json)
    ) == ()


def test_material_capital_change_requires_reoptimization() -> None:
    evaluated = _fp(_state(available_capital_usd=Decimal("500")))
    current = _fp(_state(available_capital_usd=Decimal("350")))
    diffs = evaluated.material_differences(current)
    assert "available_capital_usd" in diffs


def test_material_goal_gap_change_requires_reoptimization() -> None:
    evaluated = _fp(_state(required_profit_remaining_usd=Decimal("50")))
    current = _fp(_state(required_profit_remaining_usd=Decimal("80")))
    diffs = evaluated.material_differences(current)
    assert "remaining_profit_gap_usd" in diffs


def test_deadline_change_requires_reoptimization() -> None:
    evaluated = _fp(
        _state(deadline_exchange_time=datetime(2026, 8, 5, 12, 0, tzinfo=ET))
    )
    current = _fp(
        _state(deadline_exchange_time=datetime(2026, 8, 5, 11, 0, tzinfo=ET))
    )
    diffs = evaluated.material_differences(current)
    assert "deadline_exchange_time" in diffs


def test_position_slot_change_requires_reoptimization() -> None:
    evaluated = _fp(
        _state(max_concurrent_positions=2, open_position_count=0),
        working_order_count=0,
    )
    current = _fp(
        _state(max_concurrent_positions=2, open_position_count=1),
        working_order_count=0,
    )
    diffs = evaluated.material_differences(current)
    assert "open_position_count" in diffs
    slot_shrink = _fp(_state(max_concurrent_positions=1))
    assert "max_concurrent_positions" in evaluated.material_differences(slot_shrink)

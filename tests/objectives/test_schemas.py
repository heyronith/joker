"""Objective schema and deadline validation tests."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from joker.objectives.deadline import DeadlineParseError, resolve_deadline
from joker.objectives.schemas import build_definition


ET = ZoneInfo("America/New_York")


def test_build_definition_derives_profit_and_ending_equity() -> None:
    deadline = datetime(2026, 7, 30, 15, 30, tzinfo=ET)
    d = build_definition(
        session_id="s1",
        authorised_capital_usd=500,
        target_profit_pct=20,
        deadline_exchange_time=deadline,
        max_concurrent_positions=1,
        accepted_total_loss_risk=True,
    )
    assert d.target_profit_usd == Decimal("100.00")
    assert d.target_ending_equity_usd == Decimal("600.00")


def test_definition_rejects_naive_deadline() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_definition(
            session_id="s1",
            authorised_capital_usd=100,
            target_profit_pct=10,
            deadline_exchange_time=datetime(2026, 7, 30, 15, 30),
            max_concurrent_positions=1,
            accepted_total_loss_risk=True,
        )


def test_definition_requires_total_loss_ack() -> None:
    deadline = datetime(2026, 7, 30, 15, 30, tzinfo=ET)
    with pytest.raises(ValueError, match="accepted_total_loss_risk"):
        build_definition(
            session_id="s1",
            authorised_capital_usd=100,
            target_profit_pct=10,
            deadline_exchange_time=deadline,
            max_concurrent_positions=1,
            accepted_total_loss_risk=False,
        )


def test_resolve_same_day_deadline() -> None:
    now = datetime(2026, 7, 30, 10, 0, tzinfo=ET)
    d = resolve_deadline("15:30 ET", now=now)
    assert d.hour == 15 and d.minute == 30
    assert d.tzinfo is not None


def test_resolve_rejects_past_deadline() -> None:
    now = datetime(2026, 7, 30, 16, 0, tzinfo=ET)
    with pytest.raises(DeadlineParseError, match="already passed"):
        resolve_deadline("15:30 ET", now=now)


def test_resolve_rejects_naive_iso() -> None:
    with pytest.raises(DeadlineParseError, match="timezone"):
        resolve_deadline("2026-07-30T15:30:00")

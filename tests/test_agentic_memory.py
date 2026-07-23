"""Tests for structured signals, memory, and agentic paper helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from joker.agents.intraday import mock_intraday_result, mock_session_lesson
from joker.memory import build_day_memory, load_session_lessons, save_session_lesson
from joker.schemas.domain import (
    Playbook,
    PlaybookSetup,
    SessionLesson,
    TechnicalFeatures,
    TradeProposal,
)
from joker.strategy.signal_rules import detect_setup_from_playbook, setup_matches_features


def _features(**kwargs) -> TechnicalFeatures:
    defaults = {
        "symbol": "SPY",
        "as_of": datetime.now(timezone.utc),
        "trend_label": "trend_up",
        "momentum_5m": 0.2,
        "distance_from_vwap_pct": 0.05,
    }
    defaults.update(kwargs)
    return TechnicalFeatures(**defaults)


def _setup(**kwargs) -> PlaybookSetup:
    defaults = {
        "name": "Call",
        "direction": "long_call",
        "entry_conditions": ["test"],
        "stop_rule": "50%",
        "take_profit_rule": "100%",
        "require_trend": "trend_up",
        "vwap_side": "above",
        "min_vwap_distance_pct": 0.02,
        "min_momentum_pct": 0.1,
    }
    defaults.update(kwargs)
    return PlaybookSetup(**defaults)


def test_structured_setup_matches_features() -> None:
    setup = _setup()
    assert setup_matches_features(setup, _features()) is True
    assert setup_matches_features(setup, _features(trend_label="trend_down")) is False
    assert setup_matches_features(setup, _features(momentum_5m=0.01)) is False


def test_detect_setup_from_playbook() -> None:
    call = _setup()
    put = _setup(
        name="Put",
        direction="long_put",
        require_trend="trend_down",
        vwap_side="below",
        min_momentum_pct=-0.1,
    )
    pb = Playbook(
        trading_day=date.today(),
        title="SPY plan",
        summary="s",
        setups=[call, put],
    )
    assert detect_setup_from_playbook(pb, _features()).setup_id == call.setup_id
    assert (
        detect_setup_from_playbook(
            pb, _features(trend_label="trend_down", momentum_5m=-0.2, distance_from_vwap_pct=-0.05)
        ).setup_id
        == put.setup_id
    )


def test_memory_roundtrip(tmp_path: Path) -> None:
    lesson = SessionLesson(
        trading_day=date.today().replace(day=max(1, date.today().day - 1))
        if date.today().day > 1
        else date(2026, 7, 1),
        summary="Test lesson",
        what_worked=["a"],
        what_failed=["b"],
        next_day_hints=["c"],
        final_pnl_usd=12.5,
        trades_entered=1,
    )
    # Force a prior day
    lesson = lesson.model_copy(update={"trading_day": date(2026, 7, 1)})
    save_session_lesson(tmp_path, lesson)
    loaded = load_session_lessons(tmp_path, lookback_days=30, as_of=date(2026, 7, 10))
    assert len(loaded) == 1
    assert loaded[0].summary == "Test lesson"
    bundle = build_day_memory(data_dir=tmp_path, as_of=date(2026, 7, 10), lookback_days=30)
    assert bundle.memory_available is True
    assert bundle.recent_pnl_usd == 12.5


def test_mock_intraday_proposes_when_setup_matches() -> None:
    setup = _setup()
    pb = Playbook(
        trading_day=date.today(),
        title="SPY",
        summary="s",
        setups=[setup],
        approved=True,
    )
    result = mock_intraday_result(pb, _features(), run_id="r1")
    assert result.proposal is not None
    assert result.proposal.propose_entry is True
    assert result.proposal.setup_id == setup.setup_id


def test_trade_proposal_schema_strict() -> None:
    schema = TradeProposal.model_json_schema()
    assert schema.get("additionalProperties") is False


def test_mock_session_lesson() -> None:
    lesson = mock_session_lesson(date.today(), {"trades_entered": 0, "final_pnl_usd": 0})
    assert "No entries" in lesson.what_failed

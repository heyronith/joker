"""Tests for enriched features, propose/confirm, and session micro-memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from joker.agents.decision import confirm_gate, decision_from_pending, mock_decision, pending_from_decision
from joker.agents.session_memory import PendingProposal, SessionMicroMemory, score_trade_quality
from joker.features.engine import FeatureEngine
from joker.schemas.domain import (
    Candle,
    IntradayDecision,
    MarketSnapshot,
    Playbook,
    PlaybookSetup,
    TechnicalFeatures,
)


def _candle(i: int, price: float) -> Candle:
    ts = datetime.now(timezone.utc) - timedelta(minutes=20 - i)
    return Candle(
        symbol="SPY",
        timestamp=ts,
        open=price,
        high=price + 0.4,
        low=price - 0.3,
        close=price + 0.1,
        volume=1000 + i,
    )


def test_feature_engine_richer_fields() -> None:
    candles = [_candle(i, 550 + i * 0.2) for i in range(16)]
    snap = MarketSnapshot(
        symbol="SPY",
        timestamp=candles[-1].timestamp,
        price=candles[-1].close,
        candles=candles,
    )
    feat = FeatureEngine(max_age_seconds=99999).compute(snap)
    assert feat.candle_count == 16
    assert feat.opening_range_high is not None
    assert feat.opening_range_low is not None
    assert feat.momentum_15m is not None
    assert feat.range_15m_pct is not None
    assert feat.extension_label in {"near_vwap", "extended_up", "extended_down", "unknown"}
    assert feat.day_part


def test_confirm_gate_rejects_expired() -> None:
    pending = PendingProposal(
        direction="long_call",
        setup_id="s1",
        confidence=0.7,
        stop_pct=0.4,
        take_profit_pct=0.8,
        rationale="test",
        spy_price=550.0,
        proposed_at=datetime.now(timezone.utc) - timedelta(seconds=300),
        atm_call_mid=1.0,
    )
    ok, reason = confirm_gate(
        pending,
        spy_price=550.0,
        option_context={"atm_call": {"mid": 1.0}},
        ttl_seconds=120,
    )
    assert ok is False
    assert "expired" in reason


def test_confirm_gate_rejects_spy_drift() -> None:
    pending = PendingProposal(
        direction="long_call",
        setup_id="s1",
        confidence=0.7,
        stop_pct=0.4,
        take_profit_pct=0.8,
        rationale="test",
        spy_price=550.0,
        proposed_at=datetime.now(timezone.utc),
        atm_call_mid=1.0,
    )
    ok, reason = confirm_gate(
        pending,
        spy_price=552.0,  # ~0.36% drift
        option_context={"atm_call": {"mid": 1.0}},
        max_spy_drift_pct=0.20,
    )
    assert ok is False
    assert "spy_drift" in reason


def test_session_memory_prompt_and_outcome() -> None:
    mem = SessionMicroMemory()
    mem.record_decision(action="propose", direction="long_call", confidence=0.6, summary="edge")
    mem.note_entry(direction="long_call", entry_price=1.0)
    rec = mem.record_outcome(
        exit_reason="stop",
        exit_price=0.7,
        mae=0.4,
        mfe=0.1,
        duration_minutes=5.0,
        realized_pnl_usd=-30.0,
    )
    assert "poor_timing" in rec.quality_note or "loss" in rec.quality_note
    blob = mem.prompt_dict()
    assert len(blob["recent_decisions"]) == 1
    assert len(blob["recent_outcomes"]) == 1


def test_mock_propose_then_confirm() -> None:
    pb = Playbook(
        trading_day=datetime.now(timezone.utc).date(),
        title="t",
        summary="s",
        setups=[
            PlaybookSetup(
                setup_id="c1",
                name="call",
                direction="long_call",
                enabled=True,
                stop_rule="pct",
                take_profit_rule="pct",
            )
        ],
    )
    feat = TechnicalFeatures(
        symbol="SPY",
        as_of=datetime.now(timezone.utc),
        trend_label="up",
        momentum_5m=0.2,
    )
    mem = SessionMicroMemory()
    first = mock_decision(feat, pb, force_enter=True, session_memory=mem)
    assert first.action == "propose"
    mem.set_pending(
        pending_from_decision(
            first,
            spy_price=550.0,
            option_context={"atm_call": {"mid": 1.05}},
        )
    )
    second = mock_decision(feat, pb, session_memory=mem)
    assert second.action == "confirm"
    assert second.direction == "long_call"


def test_score_trade_quality_gave_back() -> None:
    note = score_trade_quality(
        entry_price=1.0,
        exit_price=0.9,
        mae=0.2,
        mfe=0.25,
        exit_reason="time_stop",
    )
    assert "gave_back_edge" in note


def test_decision_from_pending_maps_fields() -> None:
    pending = PendingProposal(
        direction="long_put",
        setup_id="p1",
        confidence=0.8,
        stop_pct=0.35,
        take_profit_pct=0.9,
        rationale="fade",
        spy_price=550.0,
        proposed_at=datetime.now(timezone.utc),
    )
    d = decision_from_pending(pending)
    assert isinstance(d, IntradayDecision)
    assert d.action == "confirm"
    assert d.direction == "long_put"
    assert d.stop_pct == 0.35

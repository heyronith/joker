"""EV gates, adaptive aggression, edge prefilter, OptionSelector advisory."""

from __future__ import annotations

from datetime import datetime, timezone

from joker.agents.decision import ev_entry_allowed
from joker.agents.session_memory import SessionMicroMemory
from joker.execution.option_selector import OptionSelector, OptionSelectorConfig
from joker.risk.capital import CapitalBudget, CapitalPlan
from joker.runtime.live_cli import format_live_event, should_stream_event
from joker.schemas.domain import IntradayDecision, TechnicalFeatures
from joker.schemas.replay import OptionQuoteEvent
from joker.strategy.edge_prefilter import edge_prefilter


def test_allocate_rejects_non_positive_ev() -> None:
    budget = CapitalBudget(plan=CapitalPlan(authorized_usd=500.0, aggression_mode="fixed"))
    result = budget.allocate(
        premium_per_contract=1.0,
        capital_fraction=0.5,
        confidence=0.9,
        win_probability=0.7,
        expected_value_usd=0.0,
    )
    assert result.quantity == 0
    assert result.reason == "ev_non_positive"


def test_allocate_rejects_low_win_probability() -> None:
    budget = CapitalBudget(
        plan=CapitalPlan(authorized_usd=500.0, aggression_mode="fixed", min_win_probability=0.45)
    )
    result = budget.allocate(
        premium_per_contract=1.0,
        capital_fraction=0.5,
        confidence=0.9,
        win_probability=0.30,
        expected_value_usd=1.0,
    )
    assert result.quantity == 0
    assert result.reason == "win_probability_low"


def test_allocate_respects_authorized_with_fixed_aggression() -> None:
    budget = CapitalBudget(
        plan=CapitalPlan(
            authorized_usd=500.0,
            aggression_mode="fixed",
            max_kelly_fraction=1.0,
            max_contracts_per_trade=50,
        )
    )
    result = budget.allocate(
        premium_per_contract=2.0,
        target_contracts=2,
        confidence=1.0,
        win_probability=0.6,
        expected_value_usd=1.0,
        expected_r=1.5,
    )
    assert result.quantity == 2
    assert result.notional_usd == 400.0
    assert result.ev_gate == "ok"


def test_aggression_cap_boosts_when_behind_goal() -> None:
    budget = CapitalBudget(
        plan=CapitalPlan(
            authorized_usd=500.0,
            target_profit_pct=20.0,
            aggression_mode="goal_adaptive",
            max_kelly_fraction=0.35,
            behind_goal_boost=0.15,
        )
    )
    # No progress → behind → higher than base
    behind = budget.aggression_cap(minutes_to_close=180.0)
    assert behind > 0.35
    budget.realized_pnl_usd = 90.0  # 90% of $100 target
    ahead = budget.aggression_cap(minutes_to_close=180.0)
    assert ahead < behind
    budget.realized_pnl_usd = 100.0
    assert budget.aggression_cap(minutes_to_close=180.0) <= 0.15


def test_ev_entry_allowed_vetoes_reckless() -> None:
    ok, _ = ev_entry_allowed(
        IntradayDecision(
            action="confirm",
            direction="long_call",
            confidence=0.9,
            win_probability=0.2,
            expected_r=2.0,
            expected_value_usd=1.0,
            rationale="x",
        ),
        min_win_probability=0.45,
    )
    assert ok is False
    ok2, _ = ev_entry_allowed(
        IntradayDecision(
            action="confirm",
            direction="long_call",
            confidence=0.7,
            win_probability=0.55,
            expected_r=1.5,
            expected_value_usd=0.4,
            rationale="x",
        ),
        min_win_probability=0.45,
    )
    assert ok2 is True


def test_edge_prefilter_skips_goal_met_and_no_edge() -> None:
    feat = TechnicalFeatures(
        symbol="SPY",
        as_of=datetime(2026, 7, 1, 15, 0, tzinfo=timezone.utc),
        momentum_5m=0.01,
        distance_from_vwap_pct=0.01,
        minutes_to_close=120.0,
        is_stale=False,
    )
    assert edge_prefilter(feat, goal_met=True).candidate is False
    assert edge_prefilter(feat, goal_met=False).candidate is False

    strong = feat.model_copy(
        update={
            "momentum_5m": 0.20,
            "distance_from_vwap_pct": 0.15,
            "trend_label": "trend_up",
            "extension_label": "extended_up",
        }
    )
    pre = edge_prefilter(strong, goal_met=False)
    assert pre.candidate is True
    assert pre.direction == "long_call"


def test_option_selector_advisory_does_not_hard_reject() -> None:
    from datetime import date

    ts = datetime(2026, 7, 1, 14, 30, tzinfo=timezone.utc)
    quotes = [
        OptionQuoteEvent(
            timestamp=ts,
            source="test",
            contract_id="c550",
            expiration=date(2026, 7, 1),
            strike=550.0,
            option_type="call",
            bid=1.0,
            ask=1.5,
            mid=1.25,
            spread_pct=40.0,
            quote_timestamp=ts,
        ),
    ]
    hard = OptionSelector(OptionSelectorConfig(max_spread_pct=15.0))
    try:
        hard.select_from_events(quotes, "long_call", 550.0, ts)
        assert False, "expected hard reject"
    except Exception as exc:
        assert "WIDE_SPREAD" in str(exc)

    soft = OptionSelector(
        OptionSelectorConfig(max_spread_pct=15.0, soft_liquidity_advisory=True)
    )
    selected = soft.select_from_events(quotes, "long_call", 550.0, ts)
    assert selected is not None
    assert any("WIDE_SPREAD" in a for a in soft.last_advisories)


def test_session_expectancy_stats() -> None:
    mem = SessionMicroMemory()
    mem.note_entry(direction="long_call", entry_price=1.0)
    mem.record_outcome(
        exit_reason="take_profit",
        exit_price=1.5,
        realized_pnl_usd=50.0,
        mae=0.1,
        mfe=0.6,
        duration_minutes=10.0,
    )
    mem.note_entry(direction="long_put", entry_price=2.0)
    mem.record_outcome(
        exit_reason="stop_loss",
        exit_price=1.0,
        realized_pnl_usd=-100.0,
        mae=0.5,
        mfe=0.1,
        duration_minutes=8.0,
    )
    stats = mem.expectancy_stats()
    assert stats["n"] == 2
    assert stats["win_rate"] == 0.5
    assert stats["avg_r"] is not None
    prompt = mem.prompt_dict()
    assert "session_expectancy" in prompt


def test_normalize_stock_timespan_and_parse_candle_row() -> None:
    from joker.data.webull_api import normalize_stock_timespan, _parse_candle_row

    assert normalize_stock_timespan("1m") == "M1"
    assert normalize_stock_timespan("M5") == "M5"
    row = {
        "symbol": "SPY",
        "time": "2026-07-24T19:29:00.000+0000",
        "open": "737.69",
        "close": "737.64",
        "high": "737.87",
        "low": "737.58",
        "volume": "116984",
    }
    c = _parse_candle_row(row)
    assert c.volume == 116984.0
    assert c.close == 737.64


def test_stock_bars_endpoint_verified_uses_symbol_param() -> None:
    from joker.data.webull_endpoints import get_endpoint

    ep = get_endpoint("stock_bars")
    assert ep.verified is True
    assert "symbol" in ep.required_params
    assert "symbols" not in ep.required_params


def test_cli_streams_ev_and_goal_fields() -> None:
    assert should_stream_event("agent.prefilter_skip")
    assert should_stream_event("capital.sized")
    assert should_stream_event("option.advisory")
    line = format_live_event(
        "agent.decision",
        {
            "action": "propose",
            "direction": "long_call",
            "confidence": 0.7,
            "win_probability": 0.58,
            "expected_value_usd": 0.4,
            "aggression_cap": 0.42,
            "goal_gap_pct": 80.0,
            "pending": False,
            "summary": "test",
        },
    )
    assert "p_win=0.58" in line
    assert "ev=0.4" in line
    assert "gap=80.0" in line

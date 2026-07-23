"""Strategy quality metrics tests."""

from __future__ import annotations

from joker.reporting.metrics import StrategyQualityMetrics, compute_quality_metrics
from joker.runtime.market_handler import MarketHandlerState


def test_metrics_from_handler_state() -> None:
    state = MarketHandlerState(
        setups_armed=2,
        signals_detected=3,
        risk_rejections=1,
        option_selector_rejections=2,
        trades_entered=1,
        trades_exited=1,
        exits_by_reason={"stop_loss": 1},
        stale_quote_events=1,
        wide_spread_rejects=1,
        trade_durations_minutes=[15.0],
        max_adverse_excursion=0.5,
        max_favorable_excursion=0.3,
    )
    m = compute_quality_metrics(state)
    assert m.signals_detected == 3
    assert m.candidates_risk_rejected == 1
    assert m.candidates_option_rejected == 2
    assert m.exits_by_reason["stop_loss"] == 1
    assert m.avg_time_in_trade_minutes == 15.0


def test_no_trade_day_metrics() -> None:
    m = compute_quality_metrics(MarketHandlerState())
    assert m.trades_entered == 0
    assert m.signals_detected == 0


def test_rejected_only_metrics() -> None:
    state = MarketHandlerState(signals_detected=2, risk_rejections=2, option_selector_rejections=1)
    m = compute_quality_metrics(state)
    assert m.skipped_trades == 3

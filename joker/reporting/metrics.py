"""Strategy-quality metrics (non profit-only)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrategyQualityMetrics:
    setups_armed: int = 0
    signals_detected: int = 0
    candidates_risk_rejected: int = 0
    candidates_option_rejected: int = 0
    trades_entered: int = 0
    exits_by_reason: dict[str, int] = field(default_factory=dict)
    avg_time_in_trade_minutes: float | None = None
    max_adverse_excursion: float | None = None
    max_favorable_excursion: float | None = None
    rule_violations: int = 0
    missing_data_events: int = 0
    stale_quote_events: int = 0
    wide_spread_rejects: int = 0
    skipped_trades: int = 0

    def to_dict(self) -> dict:
        return {
            "setups_armed": self.setups_armed,
            "signals_detected": self.signals_detected,
            "candidates_risk_rejected": self.candidates_risk_rejected,
            "candidates_option_rejected": self.candidates_option_rejected,
            "trades_entered": self.trades_entered,
            "exits_by_reason": dict(self.exits_by_reason),
            "avg_time_in_trade_minutes": self.avg_time_in_trade_minutes,
            "max_adverse_excursion": self.max_adverse_excursion,
            "max_favorable_excursion": self.max_favorable_excursion,
            "rule_violations": self.rule_violations,
            "missing_data_events": self.missing_data_events,
            "stale_quote_events": self.stale_quote_events,
            "wide_spread_rejects": self.wide_spread_rejects,
            "skipped_trades": self.skipped_trades,
        }


def compute_quality_metrics(
    handler_state,
    *,
    trade_durations_minutes: list[float] | None = None,
) -> StrategyQualityMetrics:
    exits = dict(getattr(handler_state, "exits_by_reason", {}) or {})
    durations = trade_durations_minutes or getattr(
        handler_state, "trade_durations_minutes", None
    )
    avg_time = None
    if durations:
        avg_time = sum(durations) / len(durations)

    risk_rej = getattr(handler_state, "risk_rejections", 0)
    option_rej = getattr(handler_state, "option_selector_rejections", 0)

    return StrategyQualityMetrics(
        setups_armed=getattr(handler_state, "setups_armed", 0),
        signals_detected=getattr(handler_state, "signals_detected", 0),
        candidates_risk_rejected=risk_rej,
        candidates_option_rejected=option_rej,
        trades_entered=getattr(handler_state, "trades_entered", 0),
        exits_by_reason=exits,
        avg_time_in_trade_minutes=avg_time,
        max_adverse_excursion=getattr(handler_state, "max_adverse_excursion", None),
        max_favorable_excursion=getattr(handler_state, "max_favorable_excursion", None),
        rule_violations=getattr(handler_state, "rule_violations", 0),
        missing_data_events=getattr(handler_state, "missing_data_events", 0),
        stale_quote_events=getattr(handler_state, "stale_quote_events", 0),
        wide_spread_rejects=getattr(handler_state, "wide_spread_rejects", 0),
        skipped_trades=risk_rej + option_rej,
    )

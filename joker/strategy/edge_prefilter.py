"""Cheap deterministic prefilter — skip LLM when no plausible 0DTE edge."""

from __future__ import annotations

from dataclasses import dataclass

from joker.schemas.domain import TechnicalFeatures


@dataclass(frozen=True)
class EdgePrefilterResult:
    candidate: bool
    direction: str | None  # long_call | long_put | None
    reason: str


def edge_prefilter(
    features: TechnicalFeatures,
    *,
    goal_met: bool = False,
    min_abs_momentum_pct: float = 0.08,
    min_vwap_distance_pct: float = 0.05,
) -> EdgePrefilterResult:
    """
    Return whether DecisionAgent should be invoked this tick.

    Looks for directional microstructure: momentum + VWAP side / OR break alignment.
    """
    if goal_met:
        return EdgePrefilterResult(False, None, "goal_met")
    if features.is_stale:
        return EdgePrefilterResult(False, None, "stale_features")
    if (features.minutes_to_close is not None) and features.minutes_to_close < 10:
        return EdgePrefilterResult(False, None, "too_close_to_eod")

    mom = features.momentum_5m
    if mom is None:
        mom = features.momentum_15m
    dist = features.distance_from_vwap_pct
    if mom is None or dist is None:
        if features.candle_count < 6:
            return EdgePrefilterResult(False, None, "warming_up_features")
        if mom is None and dist is None:
            return EdgePrefilterResult(False, None, "missing_momentum_and_vwap")
        if mom is None:
            return EdgePrefilterResult(False, None, "missing_momentum")
        return EdgePrefilterResult(False, None, "missing_vwap")

    # Long call: positive momentum above VWAP or OR high break
    or_high_break = (
        features.distance_from_or_high_pct is not None
        and features.distance_from_or_high_pct >= 0.0
        and mom > 0
    )
    or_low_break = (
        features.distance_from_or_low_pct is not None
        and features.distance_from_or_low_pct <= 0.0
        and mom < 0
    )

    if mom >= min_abs_momentum_pct and (
        dist >= min_vwap_distance_pct
        or or_high_break
        or features.extension_label in {"near_vwap", "extended_up"}
        and features.trend_label == "trend_up"
    ):
        return EdgePrefilterResult(True, "long_call", "momentum_up_vwap_or_or")

    if mom <= -min_abs_momentum_pct and (
        dist <= -min_vwap_distance_pct
        or or_low_break
        or features.extension_label in {"near_vwap", "extended_down"}
        and features.trend_label == "trend_down"
    ):
        return EdgePrefilterResult(True, "long_put", "momentum_down_vwap_or_or")

    # Chop with strong extension — still worth an LLM look for mean-reversion / continuation
    if abs(mom) >= min_abs_momentum_pct * 1.5 and abs(dist) >= min_vwap_distance_pct * 2:
        direction = "long_call" if mom > 0 else "long_put"
        return EdgePrefilterResult(True, direction, "strong_extension")

    return EdgePrefilterResult(False, None, "no_edge")

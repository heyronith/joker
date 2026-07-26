"""Deterministic episode metrics (Decimal-safe, no agent judgement)."""

from __future__ import annotations

from decimal import Decimal

from joker.evaluation.schemas import DeterministicOutcomeMetrics
from joker.evolution.schemas import TradingEpisode


def _ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return (numerator / denominator).quantize(Decimal("0.0001"))


def compute_deterministic_metrics(
    episode: TradingEpisode,
    *,
    model_call_count: int = 0,
    model_call_latency_ms_total: int = 0,
    approximate_token_usage: int = 0,
    provider_failures: int = 0,
    escalation_frequency: int = 0,
    evidence_request_count: int = 0,
    debate_iterations: int = 0,
    rejected_order_count: int = 0,
    duplicate_action_count: int = 0,
    declared_confidence: Decimal | None = None,
    outcome_hit: bool | None = None,
    safety_violations: int = 0,
    integrity_violations: int = 0,
    data_quality_exposure: int = 0,
) -> DeterministicOutcomeMetrics:
    premium = episode.entry_price
    pnl = episode.realised_pnl
    mfe = episode.max_favourable_excursion
    mae = episode.max_adverse_excursion

    profit_capture = None
    if pnl is not None and mfe is not None and mfe != 0:
        profit_capture = _ratio(pnl, mfe)

    exit_efficiency = None
    if pnl is not None and mfe is not None and mae is not None:
        span = mfe - mae
        if span != 0:
            exit_efficiency = _ratio(pnl - mae, span)

    calibration_error = None
    brier = None
    if declared_confidence is not None and outcome_hit is not None:
        y = Decimal("1") if outcome_hit else Decimal("0")
        calibration_error = abs(declared_confidence - y)
        brier = (declared_confidence - y) ** 2

    return DeterministicOutcomeMetrics(
        realised_pnl=pnl,
        return_on_premium=_ratio(pnl, premium),
        max_favourable_excursion=mfe,
        max_adverse_excursion=mae,
        profit_capture_ratio=profit_capture,
        exit_efficiency=exit_efficiency,
        entry_slippage=episode.entry_slippage,
        exit_slippage=episode.exit_slippage,
        fill_ratio=Decimal("1") if episode.action_class == "closed_trade" else None,
        holding_seconds=episode.holding_seconds,
        model_call_latency_ms_total=model_call_latency_ms_total,
        model_call_count=model_call_count,
        approximate_token_usage=approximate_token_usage,
        provider_failures=provider_failures,
        escalation_frequency=escalation_frequency,
        evidence_request_count=evidence_request_count,
        debate_iterations=debate_iterations,
        rejected_order_count=rejected_order_count,
        duplicate_action_count=duplicate_action_count,
        calibration_error=calibration_error,
        brier_score=brier,
        data_quality_exposure=data_quality_exposure or len(episode.data_quality_ids),
        safety_violations=safety_violations,
        integrity_violations=integrity_violations,
    )

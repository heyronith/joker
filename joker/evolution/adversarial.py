"""Permanent adversarial evaluation corpus for challenger eligibility."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdversarialScenario:
    scenario_id: str
    title: str
    description: str
    required: bool = True


ADVERSARIAL_CORPUS: tuple[AdversarialScenario, ...] = (
    AdversarialScenario("adv_01", "stale_option_quotes", "Strong-looking but stale option quotes"),
    AdversarialScenario("adv_02", "conflicting_evidence", "Conflicting structure vs volatility evidence"),
    AdversarialScenario("adv_03", "hallucinated_contract", "Agent invents nonexistent contract"),
    AdversarialScenario("adv_04", "false_consensus", "Repeated evidence causes false consensus"),
    AdversarialScenario("adv_05", "bullish_thin_liquidity", "Bullish price with deteriorating liquidity"),
    AdversarialScenario("adv_06", "thesis_invalidated", "Bearish thesis invalidated after entry"),
    AdversarialScenario("adv_07", "partial_fill_spread", "Partial fills then spread expansion"),
    AdversarialScenario("adv_08", "reduce_then_exit", "Position reduction followed by full exit"),
    AdversarialScenario("adv_09", "replace_deterioration", "Replace working order as quotes worsen"),
    AdversarialScenario("adv_10", "provider_timeout_debate", "Provider timeout during debate"),
    AdversarialScenario("adv_11", "local_model_unavailable", "Mandatory local model unavailable"),
    AdversarialScenario("adv_12", "escalation_unavailable", "Escalation model unavailable"),
    AdversarialScenario("adv_13", "duplicate_snapshot", "Duplicate snapshot event"),
    AdversarialScenario("adv_14", "duplicate_position", "Duplicate position event"),
    AdversarialScenario("adv_15", "crash_after_model", "Crash after model response persistence"),
    AdversarialScenario("adv_16", "crash_after_accept", "Crash after order acceptance"),
    AdversarialScenario("adv_17", "missing_data_quality", "Missing data-quality report"),
    AdversarialScenario("adv_18", "partial_option_surface", "Partial option surface"),
    AdversarialScenario("adv_19", "zero_contract_surface", "Zero-contract option surface"),
    AdversarialScenario("adv_20", "justified_no_trade_move", "Favourable move after justified no-trade"),
    AdversarialScenario("adv_21", "profit_unsupported", "Profitable trade with unsupported reasoning"),
    AdversarialScenario("adv_22", "loss_calibrated", "Losing trade with calibrated evidence"),
    AdversarialScenario("adv_23", "regime_shift", "Regime shift between entry and exit"),
    AdversarialScenario("adv_24", "concurrent_urgent_exit", "Entry cycle concurrent with urgent exit"),
    AdversarialScenario("adv_25", "narrow_period_overfit", "Config overfit to a narrow historical period"),
    # Objective / goal-driven adversarial corpus (unit-tested; not promotion-hard yet)
    AdversarialScenario("adv_obj_01", "high_target_insufficient_time", "Very high target with insufficient time", required=False),
    AdversarialScenario("adv_obj_02", "target_reached_early", "Target reached early pauses entries", required=False),
    AdversarialScenario("adv_obj_03", "large_initial_loss", "Large initial loss reduces available capital", required=False),
    AdversarialScenario("adv_obj_04", "consecutive_losses", "Multiple consecutive losses without martingale", required=False),
    AdversarialScenario("adv_obj_05", "high_ev_after_losses", "One high-EV opportunity after losses", required=False),
    AdversarialScenario("adv_obj_06", "no_valid_opportunity", "No valid opportunity all session", required=False),
    AdversarialScenario("adv_obj_07", "capital_almost_exhausted", "Capital almost exhausted", required=False),
    AdversarialScenario("adv_obj_08", "deadline_open_position", "Deadline passes with an open position", required=False),
    AdversarialScenario("adv_obj_09", "reservation_race", "Concurrent reservation race", required=False),
    AdversarialScenario("adv_obj_10", "restart_unfilled", "Restart with accepted but unfilled order", required=False),
    AdversarialScenario("adv_obj_11", "partial_fill_restart", "Partial fill before restart", required=False),
    AdversarialScenario("adv_obj_12", "broker_local_mismatch", "Broker position differs from reservation", required=False),
    AdversarialScenario("adv_obj_13", "model_oversize", "Model recommends more contracts than affordable", required=False),
    AdversarialScenario("adv_obj_14", "martingale_blocked", "Model recommends martingale-style sizing", required=False),
    AdversarialScenario("adv_obj_15", "negative_ev_high_upside", "High upside but negative expected value", required=False),
    AdversarialScenario("adv_obj_16", "probability_unavailable", "Target probability unavailable", required=False),
    AdversarialScenario("adv_obj_17", "no_trade_better", "No-trade higher objective value than trading", required=False),
    AdversarialScenario("adv_obj_18", "poor_calibration", "High-confidence model with poor calibration", required=False),
    AdversarialScenario("adv_obj_19", "zero_profit_target", "Profit target is zero", required=False),
    AdversarialScenario("adv_obj_20", "capital_below_premium", "Authorised capital smaller than one premium", required=False),
)


def required_scenario_ids() -> tuple[str, ...]:
    return tuple(s.scenario_id for s in ADVERSARIAL_CORPUS if s.required)


def evaluate_adversarial_subset(
    passed_ids: set[str],
    *,
    required_ids: tuple[str, ...] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    required = required_ids or required_scenario_ids()
    missing = tuple(sid for sid in required if sid not in passed_ids)
    return (not missing, missing)

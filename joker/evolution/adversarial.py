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

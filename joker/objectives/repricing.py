"""Execution-time EV repricing against the current option quote.

Method version: long_option_entry_cost_adjust_v1

historical_expected_gross_value =
    original_expected_value_usd + original_entry_cost + original_cost_assumptions

repriced_expected_value =
    historical_expected_gross_value - current_entry_cost - current_cost_assumptions

Entry cost is premium_per_contract * 100 * quantity (long option premium paid).
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID

from joker.objectives.historical_schemas import RepricedStrategyEstimate
from joker.objectives.schemas import StrategyObjectiveEstimate, premium_notional_usd

REPRICING_METHOD = "long_option_entry_cost_adjust_v1"


def reprice_long_option_estimate(
    estimate: StrategyObjectiveEstimate,
    *,
    current_premium_per_contract_usd: Decimal | float,
    quantity: int,
    request_snapshot_id: UUID | str,
    quote_timestamp: datetime | None = None,
    current_slippage_per_contract_usd: Decimal | float | None = None,
    max_premium_change_pct: Decimal | float = Decimal("25"),
    max_quote_age_seconds: int | None = None,
    quote_age_seconds: int | None = None,
    max_spread_pct: float | None = None,
    current_spread_pct: float | None = None,
) -> RepricedStrategyEstimate:
    """Recompute EV when entry premium changes; never invent EV if original missing."""
    ts = quote_timestamp or datetime.now(timezone.utc)
    invalidation: list[str] = []
    original_ev = estimate.expected_value_usd
    original_premium = Decimal(
        str(estimate.quote_inputs.get("premium_per_contract") or "0")
    )
    qty = max(1, int(quantity or estimate.quote_inputs.get("quantity") or 1))
    current_premium = Decimal(str(current_premium_per_contract_usd)).quantize(
        Decimal("0.01")
    )
    slip = Decimal(
        str(
            current_slippage_per_contract_usd
            if current_slippage_per_contract_usd is not None
            else estimate.quote_inputs.get("slippage_per_contract")
            or "0.02"
        )
    )

    original_entry = premium_notional_usd(original_premium + slip, qty)
    current_entry = premium_notional_usd(current_premium + slip, qty)
    change = (current_premium - original_premium).quantize(Decimal("0.01"))
    change_pct = None
    if original_premium > 0:
        change_pct = (
            (change / original_premium) * Decimal("100")
        ).quantize(Decimal("0.01"))

    assumptions_valid = True
    if original_ev is None:
        invalidation.append("original_expected_value_unavailable")
        assumptions_valid = False
    if original_premium <= 0:
        invalidation.append("original_premium_unavailable")
        assumptions_valid = False
    if not estimate.valid:
        invalidation.append("original_estimate_invalid")
        assumptions_valid = False
    if estimate.valid_until is not None and ts > estimate.valid_until:
        invalidation.append("estimate_expired")
        assumptions_valid = False
    if change_pct is not None and abs(change_pct) > Decimal(str(max_premium_change_pct)):
        invalidation.append("premium_change_exceeds_assumption")
        assumptions_valid = False
    if (
        max_quote_age_seconds is not None
        and quote_age_seconds is not None
        and quote_age_seconds > max_quote_age_seconds
    ):
        invalidation.append("quote_stale")
        assumptions_valid = False
    if (
        max_spread_pct is not None
        and current_spread_pct is not None
        and current_spread_pct > max_spread_pct
    ):
        invalidation.append("spread_unacceptable")
        assumptions_valid = False

    repriced_ev: Decimal | None = None
    if original_ev is not None and assumptions_valid:
        gross = original_ev + original_entry
        repriced_ev = (gross - current_entry).quantize(Decimal("0.01"))

    valid = (
        assumptions_valid
        and repriced_ev is not None
        and repriced_ev > 0
        and current_entry > 0
    )
    if repriced_ev is not None and repriced_ev <= 0:
        invalidation.append("repriced_expected_value_not_positive")

    return RepricedStrategyEstimate(
        original_estimate_id=estimate.estimate_id,
        request_snapshot_id=UUID(str(request_snapshot_id)),
        quote_timestamp=ts,
        original_premium_usd=original_premium,
        current_premium_usd=current_premium,
        premium_change_usd=change,
        premium_change_pct=change_pct,
        original_expected_value_usd=original_ev or Decimal("0.00"),
        repriced_expected_value_usd=repriced_ev,
        repricing_method=REPRICING_METHOD,
        original_maximum_loss_usd=estimate.maximum_loss_usd,
        repriced_maximum_loss_usd=current_entry,
        assumptions_still_valid=assumptions_valid,
        invalidation_reasons=tuple(invalidation),
        valid=valid,
    )

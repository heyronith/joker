"""Helpers to populate FeasibilityInputs from Task 1 snapshot truth."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID

from joker.objectives.feasibility import FeasibilityInputs


def build_feasibility_inputs_from_truth(
    *,
    snapshot_id: UUID,
    snapshot: Any | None = None,
    data_quality: Any | None = None,
    option_surface_slice: tuple[Any, ...] = (),
    projection: Any | None = None,
    available_capital_usd: Decimal | float | None = None,
    evidence_ids: tuple[UUID, ...] = (),
    comparable_outcome_samples: int = 0,
    historical_hit_rate: Decimal | float | None = None,
    estimated_opportunities_remaining: int | None = None,
    slippage_usd_estimate: Decimal | float | None = None,
    session_phase: str | None = None,
) -> FeasibilityInputs:
    """Construct rich FeasibilityInputs — never snapshot_id alone in production path."""
    premiums: list[Decimal] = []
    spreads: list[float] = []
    ages: list[float] = []
    ivs: list[float] = []
    affordable = 0
    available = (
        Decimal(str(available_capital_usd))
        if available_capital_usd is not None
        else None
    )

    for contract in option_surface_slice:
        bid = getattr(contract, "bid", None)
        ask = getattr(contract, "ask", None)
        if bid is None or ask is None:
            continue
        try:
            b = Decimal(str(bid))
            a = Decimal(str(ask))
        except Exception:
            continue
        if a <= 0 or b < 0 or a < b:
            continue
        mid = ((b + a) / Decimal("2")).quantize(Decimal("0.01"))
        premiums.append(mid)
        if float(b) > 0:
            spreads.append(float((a - b) / ((a + b) / Decimal("2"))))
        age = getattr(contract, "quote_age_seconds", None) or getattr(
            contract, "age_seconds", None
        )
        if age is not None:
            try:
                ages.append(float(age))
            except Exception:
                pass
        iv = getattr(contract, "implied_volatility", None) or getattr(
            contract, "iv", None
        )
        if iv is not None:
            try:
                ivs.append(float(iv))
            except Exception:
                pass
        notional = mid * Decimal("100")
        if available is None or notional <= available:
            affordable += 1

    median_premium = None
    if premiums:
        ordered = sorted(premiums)
        median_premium = ordered[len(ordered) // 2]

    typical_spread = None
    if spreads:
        typical_spread = sum(spreads) / len(spreads)

    quote_age = None
    if ages:
        quote_age = max(ages)
    elif data_quality is not None:
        quote_age = getattr(data_quality, "max_option_quote_age_seconds", None)
        if quote_age is None:
            quote_age = getattr(data_quality, "underlying_quote_age_seconds", None)

    realised_vol = None
    if snapshot is not None:
        underlying = getattr(snapshot, "underlying", None)
        if underlying is not None:
            realised_vol = getattr(underlying, "realised_volatility", None) or getattr(
                underlying, "realized_volatility", None
            )

    phase = session_phase
    if phase is None and snapshot is not None:
        phase = getattr(snapshot, "session_phase", None) or getattr(
            snapshot, "market_phase", None
        )

    open_positions = 0
    working_orders = 0
    if projection is not None:
        positions = getattr(projection, "positions", {}) or {}
        for pos in positions.values():
            qty = getattr(pos, "quantity", None)
            if qty is None and isinstance(pos, dict):
                qty = pos.get("quantity")
            try:
                if Decimal(str(qty or 0)) != 0:
                    open_positions += 1
            except Exception:
                continue
        orders = getattr(projection, "orders", {}) or {}
        for order in orders.values():
            status = getattr(order, "status", None)
            if status is None and isinstance(order, dict):
                status = order.get("status")
            if str(status or "").lower() in {
                "open",
                "accepted",
                "submitted",
                "partially_filled",
                "working",
            }:
                working_orders += 1

    return FeasibilityInputs(
        snapshot_id=snapshot_id,
        session_phase=str(phase) if phase is not None else None,
        median_premium_usd=median_premium,
        typical_spread_pct=typical_spread,
        quote_age_seconds=float(quote_age) if quote_age is not None else None,
        realised_vol=float(realised_vol) if realised_vol is not None else None,
        implied_vol=(sum(ivs) / len(ivs)) if ivs else None,
        estimated_opportunities_remaining=estimated_opportunities_remaining,
        comparable_outcome_samples=int(comparable_outcome_samples),
        historical_hit_rate=(
            Decimal(str(historical_hit_rate))
            if historical_hit_rate is not None
            else None
        ),
        slippage_usd_estimate=(
            Decimal(str(slippage_usd_estimate))
            if slippage_usd_estimate is not None
            else None
        ),
        open_positions=open_positions,
        working_orders=working_orders,
        valid_contract_count=affordable if option_surface_slice else None,
        evidence_ids=evidence_ids,
    )

"""Deterministic portfolio-level underlying scenario grids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence


@dataclass(frozen=True)
class SharedUnderlyingScenario:
    scenario_id: str
    probability: Decimal
    underlying_price: Decimal
    horizon_seconds: int
    generation_method: str
    assumptions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "probability": str(self.probability),
            "underlying_price": str(self.underlying_price),
            "horizon_seconds": self.horizon_seconds,
            "generation_method": self.generation_method,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class SharedUnderlyingScenarioGrid:
    grid_hash: str
    reference_underlying_price: Decimal
    evaluation_time: datetime
    horizon_seconds: int
    generation_method: str
    assumptions: tuple[str, ...]
    scenarios: tuple[SharedUnderlyingScenario, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid_hash": self.grid_hash,
            "reference_underlying_price": str(self.reference_underlying_price),
            "evaluation_time": self.evaluation_time.isoformat(),
            "horizon_seconds": self.horizon_seconds,
            "generation_method": self.generation_method,
            "assumptions": list(self.assumptions),
            "scenarios": [scenario.as_dict() for scenario in self.scenarios],
        }


def build_shared_underlying_scenario_grid(
    *,
    strategies: Sequence[Any],
    reference_underlying_price: Decimal,
    evaluation_time: datetime,
    horizon_seconds: int,
    base_move_pct: Decimal = Decimal("1.00"),
) -> SharedUnderlyingScenarioGrid:
    """Build one coherent distribution shared by every portfolio component.

    Directionally incompatible theses are combined as an equal-weight mixture at
    the underlying level. Contract IV and Greeks are deliberately excluded here;
    they affect each option's response to these common spots.
    """
    if evaluation_time.tzinfo is None:
        raise ValueError("shared scenario evaluation_time must be timezone-aware")
    if reference_underlying_price <= 0:
        raise ValueError("shared scenario underlying price must be positive")
    if horizon_seconds <= 0:
        raise ValueError("shared scenario horizon must be positive")
    if base_move_pct <= 0:
        raise ValueError("shared scenario move must be positive")

    signed_tilts: list[Decimal] = []
    directions: set[str] = set()
    for strategy in sorted(strategies, key=lambda item: str(item.strategy_id)):
        direction = str(
            getattr(getattr(strategy, "direction", None), "value", None)
            or getattr(strategy, "direction", "")
        ).lower()
        confidence = max(
            Decimal("0"),
            min(Decimal("1"), Decimal(str(getattr(strategy, "confidence", 0.5) or 0.5))),
        )
        if direction in {"bullish", "long", "up"}:
            sign = Decimal("1")
            directions.add("bullish")
        elif direction in {"bearish", "short", "down"}:
            sign = Decimal("-1")
            directions.add("bearish")
        else:
            sign = Decimal("0")
            directions.add("neutral")
        signed_tilts.append(sign * (confidence - Decimal("0.5")) * Decimal("0.20"))

    mixture_tilt = (
        sum(signed_tilts, Decimal("0")) / Decimal(len(signed_tilts))
        if signed_tilts
        else Decimal("0")
    )
    assumptions = [
        "one_snapshot_underlying_reference",
        "contract_iv_and_greeks_do_not_change_underlying_scenarios",
    ]
    method = "deterministic_symmetric_grid"
    if "bullish" in directions and "bearish" in directions:
        method = "deterministic_equal_weight_directional_mixture"
        assumptions.append("incompatible_directions_mixed_at_portfolio_level")
    elif mixture_tilt:
        method = "deterministic_directionally_tilted_grid"

    z_values = (-2, -1, 0, 1, 2)
    base_weights = (
        Decimal("0.08"),
        Decimal("0.22"),
        Decimal("0.40"),
        Decimal("0.22"),
        Decimal("0.08"),
    )
    raw_weights = [
        max(
            Decimal("0.001"),
            base * (Decimal("1") + mixture_tilt * Decimal(z)),
        )
        for z, base in zip(z_values, base_weights, strict=True)
    ]
    total = sum(raw_weights, Decimal("0"))
    probabilities = [(raw / total).quantize(Decimal("0.000001")) for raw in raw_weights]
    probabilities[2] += Decimal("1") - sum(probabilities, Decimal("0"))

    scenario_assumptions = tuple(assumptions)
    scenarios = tuple(
        SharedUnderlyingScenario(
            scenario_id=f"shared_underlying_z_{z:+d}",
            probability=probability,
            underlying_price=max(
                Decimal("0.01"),
                reference_underlying_price
                + (
                    reference_underlying_price
                    * base_move_pct
                    / Decimal("100")
                    * Decimal(z)
                ),
            ).quantize(Decimal("0.0001")),
            horizon_seconds=horizon_seconds,
            generation_method=method,
            assumptions=scenario_assumptions,
        )
        for z, probability in zip(z_values, probabilities, strict=True)
    )
    canonical = {
        "reference_underlying_price": str(reference_underlying_price),
        "evaluation_time": evaluation_time.isoformat(),
        "horizon_seconds": horizon_seconds,
        "generation_method": method,
        "assumptions": assumptions,
        "scenarios": [scenario.as_dict() for scenario in scenarios],
    }
    grid_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SharedUnderlyingScenarioGrid(
        grid_hash=grid_hash,
        reference_underlying_price=reference_underlying_price,
        evaluation_time=evaluation_time,
        horizon_seconds=horizon_seconds,
        generation_method=method,
        assumptions=scenario_assumptions,
        scenarios=scenarios,
    )

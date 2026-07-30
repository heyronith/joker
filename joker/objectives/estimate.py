"""Build StrategyObjectiveEstimate from observable Task 1 / Task 3 inputs.

Never accept model prose alone as a numeric EV calculation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence
from uuid import UUID

from joker.cognition.schemas import StrategyHypothesis
from joker.objectives.schemas import (
    SessionObjectiveState,
    StrategyObjectiveEstimate,
    premium_notional_usd,
)


class StrategyEstimateBuilder:
    """Deterministic estimate construction from quotes, plans, and calibrated samples."""

    def __init__(
        self,
        *,
        minimum_samples_for_calibrated_ev: int = 20,
        default_slippage_per_contract_usd: Decimal = Decimal("0.02"),
        require_positive_expected_value: bool = True,
    ) -> None:
        self.minimum_samples = minimum_samples_for_calibrated_ev
        self.default_slippage = default_slippage_per_contract_usd
        self.require_positive_ev = require_positive_expected_value

    def build(
        self,
        *,
        strategy: StrategyHypothesis,
        objective_state: SessionObjectiveState,
        snapshot_id: UUID,
        premium_per_contract_usd: Decimal | float | None = None,
        bid: Decimal | float | None = None,
        ask: Decimal | float | None = None,
        comparable_episode_count: int = 0,
        historical_avg_pnl_usd: Decimal | float | None = None,
        historical_hit_rate: Decimal | float | None = None,
        historical_payoff_ratio: Decimal | float | None = None,
        slippage_per_contract_usd: Decimal | float | None = None,
        evidence_ids: Sequence[UUID] = (),
    ) -> StrategyObjectiveEstimate:
        assumptions: list[str] = []
        uncertainty: list[str] = []
        quote_inputs: dict[str, Any] = {}

        qty = 1
        if strategy.candidate_legs:
            qty = max(1, int(strategy.candidate_legs[0].quantity))

        premium = None
        if premium_per_contract_usd is not None:
            premium = Decimal(str(premium_per_contract_usd))
            assumptions.append("premium_from_explicit_input")
        elif bid is not None and ask is not None:
            premium = (
                (Decimal(str(bid)) + Decimal(str(ask))) / Decimal("2")
            ).quantize(Decimal("0.01"))
            assumptions.append("premium_from_bid_ask_mid")
            quote_inputs["bid"] = str(bid)
            quote_inputs["ask"] = str(ask)
        else:
            uncertainty.append("premium_unavailable")

        slip = Decimal(
            str(
                slippage_per_contract_usd
                if slippage_per_contract_usd is not None
                else self.default_slippage
            )
        )
        assumptions.append(f"slippage_per_contract={slip}")

        capital = Decimal("0.00")
        max_loss = Decimal("0.00")
        if premium is not None:
            capital = premium_notional_usd(premium + slip, qty)
            max_loss = capital  # long options: max loss = premium paid
            quote_inputs["premium_per_contract"] = str(premium)
            quote_inputs["quantity"] = qty
            quote_inputs["capital_required_usd"] = str(capital)

        ev: Decimal | None = None
        win_p: Decimal | None = None
        payoff: Decimal | None = None
        method = "unsupported"

        resolution = int(strategy.expected_horizon_seconds)

        if (
            comparable_episode_count >= self.minimum_samples
            and historical_avg_pnl_usd is not None
        ):
            ev = Decimal(str(historical_avg_pnl_usd)).quantize(Decimal("0.01"))
            method = "calibrated_episode_average"
            assumptions.append(
                f"ev_from_{comparable_episode_count}_comparable_episodes"
            )
            if historical_hit_rate is not None:
                win_p = Decimal(str(historical_hit_rate)).quantize(Decimal("0.0001"))
            if historical_payoff_ratio is not None:
                payoff = Decimal(str(historical_payoff_ratio)).quantize(Decimal("0.01"))
        else:
            uncertainty.append("insufficient_calibrated_samples_for_ev")
            if comparable_episode_count > 0:
                assumptions.append(
                    f"only_{comparable_episode_count}_samples_below_minimum_"
                    f"{self.minimum_samples}"
                )
            method = "ev_unavailable"

        # Exit/stop plan presence is recorded but never turned into invented EV.
        if strategy.exit_plan is not None:
            assumptions.append("exit_plan_present")
        else:
            uncertainty.append("exit_plan_missing")

        valid = capital > 0 and premium is not None
        if self.require_positive_ev and ev is not None and ev <= 0:
            valid = False
            uncertainty.append("non_positive_expected_value")
        if self.require_positive_ev and ev is None:
            # Missing EV is not valid for entry when positive EV is required.
            valid = False
            uncertainty.append("expected_value_unavailable")

        if capital > objective_state.available_capital_usd:
            valid = False
            uncertainty.append("capital_required_exceeds_available")

        return StrategyObjectiveEstimate(
            strategy_id=strategy.strategy_id,
            objective_id=objective_state.objective_id,
            snapshot_id=snapshot_id,
            expected_value_usd=ev,
            estimated_win_probability=win_p,
            estimated_payoff_ratio=payoff,
            estimated_resolution_seconds=resolution,
            capital_required_usd=capital,
            maximum_loss_usd=max_loss,
            calculation_method=method,
            assumptions=tuple(assumptions),
            evidence_ids=tuple(evidence_ids),
            uncertainty_reasons=tuple(uncertainty),
            quote_inputs=quote_inputs,
            valid=valid,
        )

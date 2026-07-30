"""Deterministic objective capital sizer — agent quantity is advisory only."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.objectives.schemas import ObjectiveSizingDecision, SessionObjectiveState
from joker.risk.capital import CapitalBudget, CapitalPlan


class DeterministicObjectiveSizer:
    """Final quantity from authorised capital + EV gates; no martingale."""

    def __init__(
        self,
        *,
        max_capital_fraction: float = 0.85,
        max_probe_fraction: float = 0.15,
        prohibit_loss_multiplier: bool = True,
        minimum_win_probability: float = 0.45,
        require_positive_expected_value: bool = True,
        maximum_authorised_contracts: int = 20,
        min_contracts: int = 1,
        aggression_mode: str = "goal_adaptive",
        max_kelly_fraction: float = 0.35,
        behind_goal_boost: float = 0.15,
        ahead_goal_dampen: float = 0.15,
    ) -> None:
        self.max_capital_fraction = max_capital_fraction
        self.max_probe_fraction = max_probe_fraction
        self.prohibit_loss_multiplier = prohibit_loss_multiplier
        self.minimum_win_probability = minimum_win_probability
        self.require_positive_ev = require_positive_expected_value
        self.maximum_authorised_contracts = maximum_authorised_contracts
        self.min_contracts = min_contracts
        self.aggression_mode = aggression_mode
        self.max_kelly_fraction = max_kelly_fraction
        self.behind_goal_boost = behind_goal_boost
        self.ahead_goal_dampen = ahead_goal_dampen

    def size(
        self,
        state: SessionObjectiveState,
        *,
        strategy_id: UUID | None,
        premium_per_contract_usd: Decimal | float,
        requested_quantity: int | None = None,
        expected_value_usd: Decimal | float | None = None,
        estimated_win_probability: Decimal | float | None = None,
        expected_r: Decimal | float | None = None,
        is_probe: bool = False,
        confidence: float = 0.5,
        prior_loss_count: int = 0,
    ) -> ObjectiveSizingDecision:
        reasons: list[str] = []
        premium = Decimal(str(premium_per_contract_usd))
        available = state.available_capital_usd
        inputs: dict[str, Any] = {
            "requested_quantity": requested_quantity,
            "is_probe": is_probe,
            "prior_loss_count": prior_loss_count,
            "prohibit_loss_multiplier": self.prohibit_loss_multiplier,
        }

        if state.entries_paused or state.status in {
            "target_reached",
            "deadline_reached",
            "capital_exhausted",
            "stopped_by_user",
        }:
            reasons.append(f"status_blocks_sizing:{state.status}")
            return self._reject(
                state,
                strategy_id,
                premium,
                available,
                reasons,
                inputs,
                is_probe=is_probe,
                ev=expected_value_usd,
                win_p=estimated_win_probability,
                expected_r=expected_r,
            )

        if state.feasibility_classification == "infeasible":
            reasons.append("feasibility_infeasible")
            return self._reject(
                state,
                strategy_id,
                premium,
                available,
                reasons,
                inputs,
                is_probe=is_probe,
                ev=expected_value_usd,
                win_p=estimated_win_probability,
                expected_r=expected_r,
            )

        if state.open_position_count >= state.max_concurrent_positions:
            reasons.append("max_concurrent_positions")
            return self._reject(
                state,
                strategy_id,
                premium,
                available,
                reasons,
                inputs,
                is_probe=is_probe,
                ev=expected_value_usd,
                win_p=estimated_win_probability,
                expected_r=expected_r,
            )

        if self.require_positive_ev:
            if expected_value_usd is None:
                reasons.append("ev_unavailable")
                return self._reject(
                    state,
                    strategy_id,
                    premium,
                    available,
                    reasons,
                    inputs,
                    is_probe=is_probe,
                    ev=expected_value_usd,
                    win_p=estimated_win_probability,
                    expected_r=expected_r,
                )
            if Decimal(str(expected_value_usd)) <= 0:
                reasons.append("ev_non_positive")
                return self._reject(
                    state,
                    strategy_id,
                    premium,
                    available,
                    reasons,
                    inputs,
                    is_probe=is_probe,
                    ev=expected_value_usd,
                    win_p=estimated_win_probability,
                    expected_r=expected_r,
                )

        if estimated_win_probability is not None:
            if float(estimated_win_probability) < self.minimum_win_probability:
                reasons.append("win_probability_low")
                return self._reject(
                    state,
                    strategy_id,
                    premium,
                    available,
                    reasons,
                    inputs,
                    is_probe=is_probe,
                    ev=expected_value_usd,
                    win_p=estimated_win_probability,
                    expected_r=expected_r,
                )
        else:
            reasons.append("win_probability_unavailable")
            # Missing probability must not become a favourable default.

        if self.prohibit_loss_multiplier and prior_loss_count > 0:
            # Prior losses affect remaining capital/feasibility only — never multiply size.
            inputs["loss_multiplier_blocked"] = True

        plan = CapitalPlan(
            authorized_usd=float(state.authorised_capital_usd),
            target_profit_pct=float(
                (state.target_profit_usd / state.authorised_capital_usd * 100)
                if state.authorised_capital_usd > 0
                else 0
            ),
            max_concurrent_positions=state.max_concurrent_positions,
            max_contracts_per_trade=self.maximum_authorised_contracts,
            min_contracts_per_trade=self.min_contracts,
            aggression_mode=self.aggression_mode,
            max_kelly_fraction=self.max_kelly_fraction,
            min_win_probability=self.minimum_win_probability,
            behind_goal_boost=self.behind_goal_boost,
            ahead_goal_dampen=self.ahead_goal_dampen,
        )
        budget = CapitalBudget(
            plan=plan,
            reserved_usd=float(state.reserved_capital_usd),
            realized_pnl_usd=float(state.realised_pnl_usd),
            open_positions=state.open_position_count,
        )
        frac_cap = self.max_probe_fraction if is_probe else self.max_capital_fraction
        minutes = state.time_remaining_seconds / 60.0
        result = budget.allocate(
            premium_per_contract=float(premium),
            capital_fraction=frac_cap,
            target_contracts=requested_quantity,
            confidence=confidence,
            win_probability=(
                float(estimated_win_probability)
                if estimated_win_probability is not None
                else None
            ),
            expected_value_usd=(
                float(expected_value_usd) if expected_value_usd is not None else None
            ),
            expected_r=float(expected_r) if expected_r is not None else None,
            minutes_to_close=minutes,
        )
        qty = int(result.quantity)
        if qty < 1:
            reasons.append(result.reason or "allocate_rejected")
            return self._reject(
                state,
                strategy_id,
                premium,
                available,
                reasons,
                inputs,
                is_probe=is_probe,
                ev=expected_value_usd,
                win_p=estimated_win_probability,
                expected_r=expected_r,
                aggression=Decimal(str(result.aggression_cap)),
            )

        # Hard ceilings
        qty = min(qty, self.maximum_authorised_contracts)
        notional = (premium * Decimal("100") * Decimal(qty)).quantize(Decimal("0.01"))
        if notional > available:
            # shrink
            per = premium * Decimal("100")
            qty = int(available // per) if per > 0 else 0
            notional = (per * Decimal(qty)).quantize(Decimal("0.01")) if qty else Decimal("0.00")
            reasons.append("shrunk_to_available_capital")
        if qty < self.min_contracts:
            reasons.append("insufficient_capital_for_min_contract")
            return self._reject(
                state,
                strategy_id,
                premium,
                available,
                reasons,
                inputs,
                is_probe=is_probe,
                ev=expected_value_usd,
                win_p=estimated_win_probability,
                expected_r=expected_r,
                aggression=Decimal(str(result.aggression_cap)),
            )

        # Never apply loss-chasing multiplier
        if requested_quantity is not None and requested_quantity > qty:
            reasons.append("agent_quantity_capped")

        reasons.append("ok")
        return ObjectiveSizingDecision(
            sizing_id=uuid4(),
            objective_id=state.objective_id,
            strategy_id=strategy_id,
            requested_quantity=requested_quantity,
            approved_quantity=qty,
            premium_per_contract_usd=premium,
            approved_notional_usd=notional,
            available_before_usd=available,
            available_after_reservation_usd=(available - notional).quantize(Decimal("0.01")),
            aggression_cap=Decimal(str(result.aggression_cap)).quantize(Decimal("0.0001")),
            estimated_expected_value_usd=(
                Decimal(str(expected_value_usd)) if expected_value_usd is not None else None
            ),
            estimated_win_probability=(
                Decimal(str(estimated_win_probability))
                if estimated_win_probability is not None
                else None
            ),
            expected_r=Decimal(str(expected_r)) if expected_r is not None else None,
            approved=True,
            reason_codes=tuple(reasons),
            calculation_inputs=inputs,
            is_probe=is_probe,
        )

    def _reject(
        self,
        state: SessionObjectiveState,
        strategy_id: UUID | None,
        premium: Decimal,
        available: Decimal,
        reasons: list[str],
        inputs: dict[str, Any],
        *,
        is_probe: bool,
        ev: Decimal | float | None,
        win_p: Decimal | float | None,
        expected_r: Decimal | float | None,
        aggression: Decimal = Decimal("0"),
    ) -> ObjectiveSizingDecision:
        return ObjectiveSizingDecision(
            objective_id=state.objective_id,
            strategy_id=strategy_id,
            requested_quantity=None,
            approved_quantity=0,
            premium_per_contract_usd=premium,
            approved_notional_usd=Decimal("0.00"),
            available_before_usd=available,
            available_after_reservation_usd=available,
            aggression_cap=aggression,
            estimated_expected_value_usd=Decimal(str(ev)) if ev is not None else None,
            estimated_win_probability=Decimal(str(win_p)) if win_p is not None else None,
            expected_r=Decimal(str(expected_r)) if expected_r is not None else None,
            approved=False,
            reason_codes=tuple(reasons),
            calculation_inputs=inputs,
            is_probe=is_probe,
        )

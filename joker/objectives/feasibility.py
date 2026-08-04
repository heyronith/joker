"""Deterministic-first goal feasibility engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.objectives.schemas import GoalFeasibilityAssessment, SessionObjectiveState


@dataclass
class FeasibilityInputs:
    """Observable inputs — never invent historical certainty."""

    snapshot_id: UUID
    session_phase: str | None = None
    median_premium_usd: Decimal | None = None
    typical_spread_pct: float | None = None
    quote_age_seconds: float | None = None
    realised_vol: float | None = None
    implied_vol: float | None = None
    estimated_opportunities_remaining: int | None = None
    comparable_outcome_samples: int = 0
    historical_hit_rate: Decimal | None = None
    slippage_usd_estimate: Decimal | None = None
    open_positions: int = 0
    working_orders: int = 0
    valid_contract_count: int | None = None
    evidence_ids: tuple[UUID, ...] = ()


class GoalFeasibilityEngine:
    """Classify objective achievability without fabricating probabilities.

    Under ``target_attainment``, high/extreme required returns are ``low``
    (low_probability evidence), not hard ``infeasible``. Physical blocks
    (deadline, no capital, no contracts, closed session) remain ``infeasible``.
    """

    def __init__(
        self,
        *,
        minimum_samples_for_numeric_probability: int = 20,
        max_quote_age_seconds: float = 10.0,
        max_spread_pct: float = 0.35,
        policy: str = "positive_ev_baseline",
    ) -> None:
        self.minimum_samples = minimum_samples_for_numeric_probability
        self.max_quote_age_seconds = max_quote_age_seconds
        self.max_spread_pct = max_spread_pct
        self.policy = policy

    def assess(
        self,
        state: SessionObjectiveState,
        inputs: FeasibilityInputs,
    ) -> GoalFeasibilityAssessment:
        constraints: list[str] = []
        assumptions: list[str] = []
        uncertainty: list[str] = []

        if state.time_remaining_seconds <= 0:
            return GoalFeasibilityAssessment(
                objective_id=state.objective_id,
                snapshot_id=inputs.snapshot_id,
                classification="infeasible",
                estimated_success_probability=None,
                required_return_remaining_pct=self._required_return_pct(state),
                required_profit_remaining_usd=state.required_profit_remaining_usd,
                time_remaining_seconds=0,
                binding_constraints=("deadline_passed",),
                assumptions=(),
                uncertainty_reasons=("deadline_elapsed",),
                evidence_ids=inputs.evidence_ids,
            )

        if state.available_capital_usd <= 0 and state.open_position_count == 0:
            return GoalFeasibilityAssessment(
                objective_id=state.objective_id,
                snapshot_id=inputs.snapshot_id,
                classification="infeasible",
                estimated_success_probability=None,
                required_return_remaining_pct=self._required_return_pct(state),
                required_profit_remaining_usd=state.required_profit_remaining_usd,
                time_remaining_seconds=state.time_remaining_seconds,
                binding_constraints=("no_available_capital",),
                assumptions=(),
                uncertainty_reasons=(),
                evidence_ids=inputs.evidence_ids,
            )

        if (
            inputs.median_premium_usd is not None
            and inputs.median_premium_usd > state.available_capital_usd
        ):
            constraints.append("premium_exceeds_available_capital")

        if inputs.valid_contract_count is not None and inputs.valid_contract_count <= 0:
            constraints.append("no_affordable_contract")

        if inputs.quote_age_seconds is not None and inputs.quote_age_seconds > self.max_quote_age_seconds:
            constraints.append("stale_quotes")
            uncertainty.append("quote_age_elevated")

        if inputs.typical_spread_pct is not None and inputs.typical_spread_pct > self.max_spread_pct:
            constraints.append("wide_spreads")

        if inputs.session_phase in {"closed", "post_market"}:
            constraints.append("session_not_regular")

        required_pct = self._required_return_pct(state)
        # High required return with little time → low/infeasible without fabricating p
        minutes_left = state.time_remaining_seconds / 60.0
        if required_pct >= Decimal("50") and minutes_left < 30:
            constraints.append("high_target_insufficient_time")
        if required_pct >= Decimal("100") and minutes_left < 60:
            constraints.append("extreme_target_vs_time")

        target_attainment = self.policy == "target_attainment"
        if (
            "premium_exceeds_available_capital" in constraints
            or "no_affordable_contract" in constraints
        ) and state.open_position_count == 0:
            classification: str = "infeasible"
        elif "session_not_regular" in constraints:
            classification = "infeasible"
        elif (
            "extreme_target_vs_time" in constraints
            or "high_target_insufficient_time" in constraints
        ):
            # Low probability ≠ physical impossibility under target_attainment.
            if target_attainment:
                classification = "low"
            else:
                classification = (
                    "low" if required_pct < Decimal("100") else "infeasible"
                )
        elif "wide_spreads" in constraints or "stale_quotes" in constraints:
            classification = "low"
        elif required_pct <= Decimal("10") and minutes_left >= 60:
            classification = "high"
        elif required_pct <= Decimal("25"):
            classification = "medium"
        else:
            classification = "low"

        est_p: Decimal | None = None
        if inputs.comparable_outcome_samples >= self.minimum_samples and inputs.historical_hit_rate is not None:
            est_p = Decimal(str(inputs.historical_hit_rate)).quantize(Decimal("0.0001"))
            assumptions.append(
                f"hit_rate_from_{inputs.comparable_outcome_samples}_comparable_samples"
            )
        else:
            uncertainty.append("insufficient_samples_for_numeric_probability")
            assumptions.append("ordinal_classification_only")

        if classification == "infeasible":
            est_p = None  # never invent a numeric p for forced entries

        calc_inputs = {
            "session_phase": inputs.session_phase,
            "median_premium_usd": (
                str(inputs.median_premium_usd)
                if inputs.median_premium_usd is not None
                else None
            ),
            "typical_spread_pct": inputs.typical_spread_pct,
            "quote_age_seconds": inputs.quote_age_seconds,
            "realised_vol": inputs.realised_vol,
            "implied_vol": inputs.implied_vol,
            "valid_contract_count": inputs.valid_contract_count,
            "open_positions": inputs.open_positions,
            "working_orders": inputs.working_orders,
            "comparable_outcome_samples": inputs.comparable_outcome_samples,
            "estimated_opportunities_remaining": inputs.estimated_opportunities_remaining,
            "slippage_usd_estimate": (
                str(inputs.slippage_usd_estimate)
                if inputs.slippage_usd_estimate is not None
                else None
            ),
        }

        return GoalFeasibilityAssessment(
            assessment_id=uuid4(),
            objective_id=state.objective_id,
            snapshot_id=inputs.snapshot_id,
            classification=classification,  # type: ignore[arg-type]
            estimated_success_probability=est_p,
            required_return_remaining_pct=required_pct,
            required_profit_remaining_usd=state.required_profit_remaining_usd,
            time_remaining_seconds=state.time_remaining_seconds,
            estimated_opportunities_remaining=inputs.estimated_opportunities_remaining,
            minimum_required_expected_value_usd=None,
            minimum_required_win_probability=None,
            minimum_required_payoff_ratio=None,
            binding_constraints=tuple(constraints),
            assumptions=tuple(assumptions),
            uncertainty_reasons=tuple(uncertainty),
            evidence_ids=inputs.evidence_ids,
            calculation_inputs=calc_inputs,
            created_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _required_return_pct(state: SessionObjectiveState) -> Decimal:
        if state.authorised_capital_usd <= 0:
            return Decimal("0.00")
        return (
            state.required_profit_remaining_usd
            / state.authorised_capital_usd
            * Decimal("100")
        ).quantize(Decimal("0.01"))

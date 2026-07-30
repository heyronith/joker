"""Objective-aware strategy scoring including no-trade."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

from joker.objectives.schemas import ObjectiveStrategyScore, SessionObjectiveState


@dataclass
class StrategyScoreInput:
    strategy_id: UUID | None
    snapshot_id: UUID
    expected_value_usd: Decimal | float | None = None
    estimated_win_probability: Decimal | float | None = None
    estimated_payoff_ratio: Decimal | float | None = None
    estimated_resolution_seconds: int | None = None
    maximum_loss_usd: Decimal | float = 0
    capital_required_usd: Decimal | float = 0
    evidence_ids: tuple[UUID, ...] = ()
    assumptions: tuple[str, ...] = ()
    calculation_inputs: dict[str, Any] | None = None
    is_no_trade: bool = False


class ObjectiveStrategyScorer:
    """Prefer positive-EV strategies that improve target probability; always score no-trade."""

    def __init__(
        self,
        *,
        require_positive_expected_value: bool = True,
        minimum_win_probability: float = 0.45,
        allow_ordinal_when_probability_unavailable: bool = True,
    ) -> None:
        self.require_positive_ev = require_positive_expected_value
        self.min_win_p = minimum_win_probability
        self.allow_ordinal = allow_ordinal_when_probability_unavailable

    def score(
        self,
        state: SessionObjectiveState,
        candidate: StrategyScoreInput,
        *,
        target_probability_before: Decimal | None = None,
    ) -> ObjectiveStrategyScore:
        codes: list[str] = []
        ev = (
            Decimal(str(candidate.expected_value_usd))
            if candidate.expected_value_usd is not None
            else None
        )
        win_p = (
            Decimal(str(candidate.estimated_win_probability))
            if candidate.estimated_win_probability is not None
            else None
        )
        max_loss = Decimal(str(candidate.maximum_loss_usd)).quantize(Decimal("0.01"))
        capital = Decimal(str(candidate.capital_required_usd)).quantize(Decimal("0.01"))
        payoff = (
            Decimal(str(candidate.estimated_payoff_ratio))
            if candidate.estimated_payoff_ratio is not None
            else None
        )

        if candidate.is_no_trade:
            return ObjectiveStrategyScore(
                objective_id=state.objective_id,
                strategy_id=None,
                snapshot_id=candidate.snapshot_id,
                expected_value_usd=Decimal("0.00"),
                maximum_loss_usd=Decimal("0.00"),
                capital_required_usd=Decimal("0.00"),
                target_probability_before=target_probability_before,
                target_probability_after=target_probability_before,
                target_probability_delta=Decimal("0.00")
                if target_probability_before is not None
                else None,
                calculation_inputs={"kind": "no_trade"},
                assumptions=("no_trade_preserves_option_value",),
                evidence_ids=candidate.evidence_ids,
                valid=True,
                is_no_trade=True,
            )

        if capital > state.available_capital_usd:
            codes.append("capital_required_exceeds_available")
        if (
            candidate.estimated_resolution_seconds is not None
            and candidate.estimated_resolution_seconds > state.time_remaining_seconds
        ):
            codes.append("resolution_after_deadline")
        if self.require_positive_ev and ev is not None and ev <= 0:
            codes.append("non_positive_expected_value")
        if self.require_positive_ev and ev is None and not candidate.is_no_trade:
            codes.append("expected_value_unavailable")
        if win_p is not None and float(win_p) < self.min_win_p:
            codes.append("win_probability_below_minimum")
        if ev is None and win_p is None and not self.allow_ordinal and not candidate.is_no_trade:
            codes.append("probability_unavailable")

        p_after: Decimal | None = None
        p_delta: Decimal | None = None
        if target_probability_before is not None and ev is not None and state.target_profit_usd > 0:
            # Coarse ordinal bump: do not invent calibrated probabilities
            bump = Decimal("0.0")
            if ev > 0 and (win_p is None or win_p >= Decimal(str(self.min_win_p))):
                bump = min(Decimal("0.05"), ev / state.target_profit_usd)
            p_after = min(Decimal("0.99"), target_probability_before + bump)
            p_delta = (p_after - target_probability_before).quantize(Decimal("0.0001"))
        elif target_probability_before is None:
            codes.append("target_probability_unavailable")

        ret_pct = None
        if ev is not None and state.authorised_capital_usd > 0:
            ret_pct = (ev / state.authorised_capital_usd * Decimal("100")).quantize(
                Decimal("0.01")
            )

        valid = not any(
            c
            in {
                "capital_required_exceeds_available",
                "non_positive_expected_value",
                "expected_value_unavailable",
                "resolution_after_deadline",
                "win_probability_below_minimum",
            }
            for c in codes
        )

        return ObjectiveStrategyScore(
            score_id=uuid4(),
            objective_id=state.objective_id,
            strategy_id=candidate.strategy_id,
            snapshot_id=candidate.snapshot_id,
            expected_value_usd=ev,
            expected_return_on_authorised_capital_pct=ret_pct,
            estimated_win_probability=win_p,
            estimated_payoff_ratio=payoff,
            estimated_resolution_seconds=candidate.estimated_resolution_seconds,
            target_probability_before=target_probability_before,
            target_probability_after=p_after,
            target_probability_delta=p_delta,
            maximum_loss_usd=max_loss,
            capital_required_usd=capital,
            opportunity_cost_usd=None,
            calculation_inputs=dict(candidate.calculation_inputs or {}),
            assumptions=candidate.assumptions,
            evidence_ids=candidate.evidence_ids,
            valid=valid,
            invalidation_codes=tuple(codes),
            is_no_trade=False,
        )

    def score_all(
        self,
        state: SessionObjectiveState,
        candidates: list[StrategyScoreInput],
        *,
        snapshot_id: UUID,
        target_probability_before: Decimal | None = None,
    ) -> list[ObjectiveStrategyScore]:
        scores = [
            self.score(state, c, target_probability_before=target_probability_before)
            for c in candidates
        ]
        scores.append(
            self.score(
                state,
                StrategyScoreInput(
                    strategy_id=None,
                    snapshot_id=snapshot_id,
                    is_no_trade=True,
                ),
                target_probability_before=target_probability_before,
            )
        )
        return scores

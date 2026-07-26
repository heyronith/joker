"""Deterministic promotion eligibility gate (no strategic judgement)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal

from joker.evolution.config import PromotionSettings
from joker.evolution.schemas import (
    ExperimentResult,
    ImprovementProposal,
    PROHIBITED_MUTATION_TARGETS,
)


HARD_VETO_CODES = frozenset(
    {
        "safety_violation",
        "execution_integrity_violation",
        "invented_contract_accepted",
        "future_expiry_accepted",
        "duplicate_broker_action",
        "position_oversell",
        "task1_truth_bypass",
        "hidden_chain_of_thought",
        "holdout_leakage",
        "dataset_membership_corruption",
        "missing_experiment_provenance",
        "non_reproducible_experiment_state",
        "unresolved_checkpoint_failure",
        "material_data_quality_regression",
        "catastrophic_tail_loss_regression",
        "adversarial_scenario_failure",
        "starting_state_mismatch",
        "source_code_or_validator_mutation",
        "missing_critical_metric",
    }
)

# Metrics that must be present on both champion and challenger.
CRITICAL_METRICS: tuple[tuple[str, Literal["higher", "lower"]], ...] = (
    ("tail_loss", "lower"),  # more negative is worse
    ("calibration_error", "higher"),
    ("latency_ms", "higher"),
    ("cost_gbp", "higher"),
)


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    gate_codes: tuple[str, ...]
    details: dict[str, Any]


class PromotionEligibilityGate:
    """Calculate eligibility; never decides strategic promotion."""

    def __init__(self, settings: PromotionSettings | None = None) -> None:
        self._settings = settings or PromotionSettings()

    def evaluate(
        self,
        *,
        result: ExperimentResult,
        proposal: ImprovementProposal | None = None,
        holdout_episode_count: int = 0,
        completed_episode_count: int = 0,
        adversarial_passed: bool = True,
        prohibited_mutation_attempted: bool = False,
        holdout_leakage: bool = False,
    ) -> EligibilityResult:
        codes: list[str] = []

        if result.safety_failures:
            codes.append("safety_violation")
        if result.data_integrity_failures:
            codes.append("execution_integrity_violation")
        for failure in result.gate_rejection_codes:
            if failure in HARD_VETO_CODES or str(failure).startswith("missing_"):
                codes.append(str(failure))
        if not adversarial_passed:
            codes.append("adversarial_scenario_failure")
        if prohibited_mutation_attempted:
            codes.append("source_code_or_validator_mutation")
        if holdout_leakage:
            codes.append("holdout_leakage")
        if proposal is not None:
            target = str(proposal.proposed_change.get("mutation_target", "")).lower()
            if target in PROHIBITED_MUTATION_TARGETS:
                codes.append("source_code_or_validator_mutation")

        if completed_episode_count < self._settings.minimum_completed_episodes:
            codes.append("insufficient_completed_episodes")
        if holdout_episode_count < self._settings.minimum_holdout_episodes:
            codes.append("insufficient_holdout_episodes")

        champ = result.champion_metrics
        chall = result.challenger_metrics
        thresholds = {
            "tail_loss": self._settings.maximum_tail_loss_regression_pct,
            "calibration_error": self._settings.maximum_calibration_regression_pct,
            "latency_ms": self._settings.maximum_latency_regression_pct,
            "cost_gbp": self._settings.maximum_cost_regression_pct,
        }
        for metric, worse_direction in CRITICAL_METRICS:
            codes.extend(
                self._regression_codes(
                    champ,
                    chall,
                    metric,
                    thresholds[metric],
                    worse_direction=worse_direction,
                    required=True,
                )
            )

        seen: set[str] = set()
        ordered: list[str] = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                ordered.append(code)

        hard = [
            c
            for c in ordered
            if c in HARD_VETO_CODES
            or c.startswith("missing_critical_metric")
            or c.endswith("_regression")
        ]
        eligible = not hard and "insufficient_completed_episodes" not in ordered
        if "insufficient_holdout_episodes" in ordered:
            eligible = False

        return EligibilityResult(
            eligible=eligible,
            gate_codes=tuple(ordered),
            details={
                "hard_vetoes": hard,
                "completed_episode_count": completed_episode_count,
                "holdout_episode_count": holdout_episode_count,
            },
        )

    def _regression_codes(
        self,
        champion: dict[str, Any],
        challenger: dict[str, Any],
        metric: str,
        max_regression_pct: Decimal,
        *,
        worse_direction: Literal["higher", "lower"] = "higher",
        required: bool = False,
    ) -> list[str]:
        if metric not in champion or metric not in challenger:
            if required:
                return [f"missing_critical_metric:{metric}"]
            return []
        try:
            c0 = Decimal(str(champion[metric]))
            c1 = Decimal(str(challenger[metric]))
        except Exception:
            if required:
                return [f"missing_critical_metric:{metric}"]
            return []
        if c0 == 0:
            # Zero baseline: any movement in the worse direction is a veto when required.
            if worse_direction == "higher" and c1 > 0:
                return [f"{metric}_regression"]
            if worse_direction == "lower" and c1 < 0:
                return [f"{metric}_regression"]
            return []

        if worse_direction == "higher":
            # Larger values are worse (latency, cost, calibration error).
            if c1 <= c0:
                return []
            change_pct = ((c1 - c0) / abs(c0)) * Decimal("100")
        else:
            # Smaller / more-negative values are worse (tail_loss).
            # champion=-10, challenger=-40 → regression 300%.
            if c1 >= c0:
                return []
            change_pct = ((c0 - c1) / abs(c0)) * Decimal("100")

        if change_pct > max_regression_pct:
            code = f"{metric}_regression"
            if metric == "tail_loss" and change_pct > max_regression_pct:
                # Also emit the catastrophic veto alias when downside collapses.
                return [code, "catastrophic_tail_loss_regression"]
            return [code]
        return []

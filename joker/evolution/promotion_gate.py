"""Deterministic promotion eligibility gate (no strategic judgement)."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

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
    }
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
            if failure in HARD_VETO_CODES:
                codes.append(failure)
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

        # Non-inferiority / regression checks on aggregate metrics.
        champ = result.champion_metrics
        chall = result.challenger_metrics
        codes.extend(
            self._regression_codes(
                champ, chall, "tail_loss", self._settings.maximum_tail_loss_regression_pct
            )
        )
        codes.extend(
            self._regression_codes(
                champ,
                chall,
                "calibration_error",
                self._settings.maximum_calibration_regression_pct,
                higher_is_worse=True,
            )
        )
        codes.extend(
            self._regression_codes(
                champ,
                chall,
                "latency_ms",
                self._settings.maximum_latency_regression_pct,
                higher_is_worse=True,
            )
        )
        codes.extend(
            self._regression_codes(
                champ,
                chall,
                "cost_gbp",
                self._settings.maximum_cost_regression_pct,
                higher_is_worse=True,
            )
        )

        # Deduplicate while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                ordered.append(code)

        hard = [c for c in ordered if c in HARD_VETO_CODES]
        eligible = len(hard) == 0 and "insufficient_completed_episodes" not in ordered
        if "insufficient_holdout_episodes" in ordered:
            eligible = False
        if any(
            c.endswith("_regression") for c in ordered
        ) and self._settings.require_deterministic_eligibility:
            # Soft regressions block eligibility by default.
            eligible = False

        return EligibilityResult(
            eligible=eligible and not ordered[:1] == ["safety_violation"],
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
        higher_is_worse: bool = True,
    ) -> list[str]:
        if metric not in champion or metric not in challenger:
            return []
        try:
            c0 = Decimal(str(champion[metric]))
            c1 = Decimal(str(challenger[metric]))
        except Exception:
            return []
        if c0 == 0:
            return []
        if higher_is_worse:
            change_pct = ((c1 - c0) / abs(c0)) * Decimal("100")
            if change_pct > max_regression_pct:
                return [f"{metric}_regression"]
        else:
            change_pct = ((c0 - c1) / abs(c0)) * Decimal("100")
            if change_pct > max_regression_pct:
                return [f"{metric}_regression"]
        return []

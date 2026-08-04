"""Typed public construction of goal-driven objective engines."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from joker.objectives.config import HistoricalOutcomeSettings
from joker.objectives.feasibility import GoalFeasibilityEngine
from joker.objectives.historical_outcomes import HistoricalOutcomeService
from joker.objectives.scoring import ObjectiveStrategyScorer
from joker.objectives.sizing import DeterministicObjectiveSizer
from joker.objectives.target_attainment import TargetAttainmentPolicy


class HistoricalSourceDiagnostic(BaseModel):
    """Why historical EV sources are or are not configured."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_loader_configured: bool = False
    evaluation_loader_configured: bool = False
    dataset_loader_configured: bool = False
    objective_repository_attached: bool = False
    reason: str | None = None
    cold_start: bool = True


class ObjectiveEngineBundle(BaseModel):
    """Public dependency bundle for goal-driven cognitive control."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    feasibility_engine: GoalFeasibilityEngine
    objective_strategy_scorer: ObjectiveStrategyScorer
    capital_sizer: DeterministicObjectiveSizer
    historical_outcome_service: HistoricalOutcomeService
    historical_outcome_settings: HistoricalOutcomeSettings
    target_attainment_policy: TargetAttainmentPolicy | None = None
    objective_policy: str = "positive_ev_baseline"
    shadow_baseline_enabled: bool = False
    source_diagnostic: HistoricalSourceDiagnostic = Field(
        default_factory=HistoricalSourceDiagnostic
    )

    def as_deps_kwargs(self) -> dict[str, Any]:
        """Keyword args suitable for CognitiveGraphDeps construction."""
        return {
            "feasibility_engine": self.feasibility_engine,
            "objective_strategy_scorer": self.objective_strategy_scorer,
            "capital_sizer": self.capital_sizer,
            "historical_outcome_service": self.historical_outcome_service,
            "historical_outcome_settings": self.historical_outcome_settings,
            "target_attainment_policy": self.target_attainment_policy,
            "objective_policy": self.objective_policy,
            "shadow_baseline_enabled": self.shadow_baseline_enabled,
        }

"""Task 3 evolution configuration (paper-only, defaults disabled)."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class EpisodeCompilerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    no_trade_windows_minutes: list[int] = Field(default_factory=lambda: [5, 15, 30])
    finalise_at_session_end: bool = True


class EvaluationRuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_parallel_episodes: int = 4
    evaluator_profile: str = "reasoning_local"
    counterfactual_max_per_episode: int = 3
    evaluator_version: str = "3.0.0"


class DatasetSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_episode_count: int = 50
    minimum_holdout_count: int = 20
    minimum_regime_count: int = 3
    allow_incomplete_in_promotion: bool = False


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_parallel_replays: int = 4
    maximum_model_calls: int = 5000
    maximum_cost_gbp: Decimal = Decimal("25.00")
    repeated_samples: int = 3


class ShadowSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    maximum_challengers: int = 1
    queue_size: int = 128
    snapshot_coalescing: bool = True


class PromotionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto_promote_paper_only: bool = True
    require_agent_recommendation: bool = True
    require_deterministic_eligibility: bool = True
    maximum_tail_loss_regression_pct: Decimal = Decimal("5")
    maximum_calibration_regression_pct: Decimal = Decimal("3")
    maximum_latency_regression_pct: Decimal = Decimal("25")
    maximum_cost_regression_pct: Decimal = Decimal("50")
    minimum_completed_episodes: int = 20
    minimum_holdout_episodes: int = 10


class DriftSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    rolling_episode_window: int = 50
    safety_rollback_immediate: bool = True
    strategic_rollback_requires_agent: bool = True


class OrchestratorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    minimum_new_completed_episodes: int = 50
    minimum_new_evaluations: int = 50
    minimum_holdout_episodes: int = 20
    maximum_active_challengers: int = 1
    automatic_cycle_interval_minutes: int = 60
    pause_under_load: bool = True


class EvolutionSettings(BaseModel):
    """Task 3 evolution controls. Default ``enabled=False`` preserves Task 1/2."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    episode_compiler: EpisodeCompilerSettings = Field(
        default_factory=EpisodeCompilerSettings
    )
    evaluation: EvaluationRuntimeSettings = Field(
        default_factory=EvaluationRuntimeSettings
    )
    datasets: DatasetSettings = Field(default_factory=DatasetSettings)
    experiments: ExperimentSettings = Field(default_factory=ExperimentSettings)
    shadow: ShadowSettings = Field(default_factory=ShadowSettings)
    promotion: PromotionSettings = Field(default_factory=PromotionSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)

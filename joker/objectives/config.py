"""Objective / session-goal configuration (production fail-closed)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObjectiveFeasibilitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    minimum_samples_for_numeric_probability: int = 20
    allow_ordinal_scoring_when_probability_unavailable: bool = True

    @field_validator("minimum_samples_for_numeric_probability")
    @classmethod
    def _samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("minimum_samples_for_numeric_probability must be >= 1")
        return value


class ObjectiveSizingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "objective_adaptive"
    max_capital_fraction: float = 0.85
    max_probe_fraction: float = 0.15
    prohibit_loss_multiplier: bool = True

    @field_validator("max_capital_fraction", "max_probe_fraction")
    @classmethod
    def _frac(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("fraction must be in (0, 1]")
        return value

    @field_validator("mode")
    @classmethod
    def _mode(cls, value: str) -> str:
        allowed = {"objective_adaptive", "fixed", "target_attainment"}
        if value not in allowed:
            raise ValueError(f"sizing.mode must be one of {sorted(allowed)}")
        return value


class HistoricalOutcomeSettings(BaseModel):
    """Factual Task-3 historical analogue / EV eligibility thresholds."""

    model_config = ConfigDict(extra="forbid")

    minimum_samples_for_ev: int = 20
    maximum_samples: int = 200
    minimum_similarity: float = 0.65
    minimum_effective_sample_size: float = 15.0
    maximum_episode_age_days: int = 90
    confidence_level: float = 0.95
    require_lower_confidence_bound_positive: bool = True
    require_same_strategy_family: bool = True
    use_similarity_weighting: bool = True
    estimate_ttl_seconds: int = 300
    max_premium_change_pct_for_repricing: float = 25.0

    @field_validator(
        "minimum_samples_for_ev", "maximum_samples", "maximum_episode_age_days"
    )
    @classmethod
    def _pos_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("minimum_similarity", "confidence_level")
    @classmethod
    def _unit(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("must be in (0, 1]")
        return value


class ExplorationModeSettings(BaseModel):
    """Operator-approved research exploration — disabled by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    operator_confirmation_required: bool = True
    paper_only: bool = True
    maximum_capital_usd: float = 25.0
    maximum_concurrent_positions: int = 1
    maximum_trades_per_session: int = 1
    require_complete_truth: bool = True


class ObjectiveExecutionSettings(BaseModel):
    """Execution-time quote/limit reconciliation for long-option buys."""

    model_config = ConfigDict(extra="forbid")

    maximum_buy_limit_above_ask_pct: float = 5.0

    @field_validator("maximum_buy_limit_above_ask_pct")
    @classmethod
    def _pct(cls, value: float) -> float:
        if value < 0:
            raise ValueError("maximum_buy_limit_above_ask_pct must be >= 0")
        return value


class TargetAttainmentProbabilitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    empirical_enabled: bool = True
    ordinal_fallback_enabled: bool = True


class TargetAttainmentQuantitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_all_affordable_quantities: bool = True


class TargetAttainmentNoTradeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_opportunity_cost: bool = True


class TargetAttainmentTieBreakSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_lower_confidence_bound: bool = True


class TargetAttainmentSettings(BaseModel):
    """Maximize P(goal by deadline) under authorized-capital constraints."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_full_remaining_capital: bool = True
    maximum_capital_fraction: float = 1.0
    minimum_calibrated_samples: int = 20
    probability_estimation: TargetAttainmentProbabilitySettings = Field(
        default_factory=TargetAttainmentProbabilitySettings
    )
    quantity_search: TargetAttainmentQuantitySettings = Field(
        default_factory=TargetAttainmentQuantitySettings
    )
    no_trade: TargetAttainmentNoTradeSettings = Field(
        default_factory=TargetAttainmentNoTradeSettings
    )
    tie_breaking: TargetAttainmentTieBreakSettings = Field(
        default_factory=TargetAttainmentTieBreakSettings
    )

    @field_validator("maximum_capital_fraction")
    @classmethod
    def _cap(cls, value: float) -> float:
        if not 0 < value <= 1:
            raise ValueError("maximum_capital_fraction must be in (0, 1]")
        return value

    @field_validator("minimum_calibrated_samples")
    @classmethod
    def _samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("minimum_calibrated_samples must be >= 1")
        return value


class ObjectiveSettings(BaseModel):
    """Session objective gates — no default capital/target/deadline that silently arms."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    # positive_ev_baseline | target_attainment
    policy: str = "positive_ev_baseline"
    shadow_baseline_enabled: bool = False
    require_session_confirmation: bool = True
    require_total_loss_acknowledgement: bool = True
    require_deadline: bool = True
    pause_entries_when_goal_met: bool = True
    stop_new_entries_at_deadline: bool = True

    # Baseline-policy evidence thresholds (hard vetoes only under positive_ev_baseline).
    minimum_win_probability: float = 0.45
    require_positive_expected_value: bool = True

    default_max_concurrent_positions: int = 1
    maximum_authorised_contracts: int = 20

    feasibility: ObjectiveFeasibilitySettings = Field(
        default_factory=ObjectiveFeasibilitySettings
    )
    sizing: ObjectiveSizingSettings = Field(default_factory=ObjectiveSizingSettings)
    historical_outcomes: HistoricalOutcomeSettings = Field(
        default_factory=HistoricalOutcomeSettings
    )
    execution: ObjectiveExecutionSettings = Field(
        default_factory=ObjectiveExecutionSettings
    )
    exploration: ExplorationModeSettings = Field(
        default_factory=ExplorationModeSettings
    )
    target_attainment: TargetAttainmentSettings = Field(
        default_factory=TargetAttainmentSettings
    )
    operator_event_capacity: int = 256

    @field_validator("policy")
    @classmethod
    def _policy(cls, value: str) -> str:
        allowed = {"positive_ev_baseline", "target_attainment"}
        if value not in allowed:
            raise ValueError(f"objective.policy must be one of {sorted(allowed)}")
        return value

    @field_validator("minimum_win_probability")
    @classmethod
    def _win(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("minimum_win_probability must be in [0, 1]")
        return value

    @field_validator("default_max_concurrent_positions", "maximum_authorised_contracts")
    @classmethod
    def _pos(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @property
    def is_target_attainment(self) -> bool:
        return self.policy == "target_attainment"

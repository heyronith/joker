"""Objective / session-goal configuration (production fail-closed)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class FullChainOptimizerSettings(BaseModel):
    """Deterministic paper/replay full-chain and portfolio optimizer settings."""

    model_config = ConfigDict(extra="forbid")

    # Disabled by default globally. Paper/replay profiles must opt in.
    enabled: bool = False
    maximum_quote_age_seconds: int = 30
    maximum_relative_spread: float = 0.25
    maximum_contracts_evaluated: int = 200
    moneyness_buckets: tuple[float, ...] = (-2.0, -0.5, 0.5, 2.0)
    premium_buckets: tuple[float, ...] = (0.10, 0.20, 0.50, 1.0, 2.0, 5.0)
    delta_buckets: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    top_contracts_per_strategy: int = 50
    top_candidates_for_agent_review: int = 10
    portfolio_search_enabled: bool = True
    portfolio_beam_width: int = 50
    maximum_portfolio_candidates: int = 500
    allow_duplicate_contracts: bool = False
    minimum_probability_improvement_over_wait: float = 0.01
    cli_graph_view: str = "compact"
    cli_top_contract_rows: int = 10
    cli_top_portfolio_rows: int = 10

    @field_validator(
        "maximum_quote_age_seconds",
        "maximum_contracts_evaluated",
        "top_contracts_per_strategy",
        "top_candidates_for_agent_review",
        "portfolio_beam_width",
        "maximum_portfolio_candidates",
        "cli_top_contract_rows",
        "cli_top_portfolio_rows",
    )
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("full-chain integer limits must be >= 1")
        return value

    @field_validator("maximum_relative_spread")
    @classmethod
    def _spread(cls, value: float) -> float:
        if not 0 < value <= 2:
            raise ValueError("maximum_relative_spread must be in (0, 2]")
        return value

    @field_validator("minimum_probability_improvement_over_wait")
    @classmethod
    def _probability_delta(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError(
                "minimum_probability_improvement_over_wait must be in [0, 1]"
            )
        return value

    @field_validator("cli_graph_view")
    @classmethod
    def _graph_view(cls, value: str) -> str:
        if value not in {"compact", "verbose", "json"}:
            raise ValueError("cli_graph_view must be compact, verbose, or json")
        return value

    @model_validator(mode="after")
    def _ordered_buckets(self) -> FullChainOptimizerSettings:
        for name in ("moneyness_buckets", "premium_buckets", "delta_buckets"):
            values = tuple(getattr(self, name))
            if not values or any(b <= a for a, b in zip(values, values[1:])):
                raise ValueError(f"{name} must be non-empty and strictly increasing")
        if any(value <= 0 for value in self.premium_buckets):
            raise ValueError("premium_buckets must contain only positive values")
        if any(not 0 < value <= 1 for value in self.delta_buckets):
            raise ValueError("delta_buckets must be in (0, 1]")
        return self


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

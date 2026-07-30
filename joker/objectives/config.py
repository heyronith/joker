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
        allowed = {"objective_adaptive", "fixed"}
        if value not in allowed:
            raise ValueError(f"sizing.mode must be one of {sorted(allowed)}")
        return value


class ObjectiveSettings(BaseModel):
    """Session objective gates — no default capital/target/deadline that silently arms."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    require_session_confirmation: bool = True
    require_total_loss_acknowledgement: bool = True
    require_deadline: bool = True
    pause_entries_when_goal_met: bool = True
    stop_new_entries_at_deadline: bool = True

    minimum_win_probability: float = 0.45
    require_positive_expected_value: bool = True

    default_max_concurrent_positions: int = 1
    maximum_authorised_contracts: int = 20

    feasibility: ObjectiveFeasibilitySettings = Field(
        default_factory=ObjectiveFeasibilitySettings
    )
    sizing: ObjectiveSizingSettings = Field(default_factory=ObjectiveSizingSettings)
    operator_event_capacity: int = 256

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

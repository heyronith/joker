"""ObjectiveSettings config loading."""

from __future__ import annotations

from pathlib import Path

from joker.config.loader import load_app_settings


def test_paper_all_tasks_loads_objective_section() -> None:
    root = Path(__file__).resolve().parents[2]
    app, _env = load_app_settings(root / "config" / "paper-all-tasks.yaml")
    assert app.objective.enabled is True
    assert app.objective.policy == "target_attainment"
    assert app.objective.shadow_baseline_enabled is True
    assert app.objective.require_deadline is True
    assert app.objective.require_total_loss_acknowledgement is True
    assert app.objective.sizing.prohibit_loss_multiplier is True
    assert app.objective.sizing.max_capital_fraction == 1.0
    assert app.objective.feasibility.minimum_samples_for_numeric_probability == 20
    assert app.objective.historical_outcomes.minimum_samples_for_ev == 20
    assert app.objective.historical_outcomes.require_lower_confidence_bound_positive
    assert app.objective.target_attainment.allow_full_remaining_capital is True
    assert app.objective.exploration.enabled is False


def test_paper_yaml_loads_target_attainment_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    app, _env = load_app_settings(root / "config" / "paper.yaml")
    assert app.objective.policy == "target_attainment"
    assert app.objective.shadow_baseline_enabled is True
    assert app.objective.is_target_attainment is True
